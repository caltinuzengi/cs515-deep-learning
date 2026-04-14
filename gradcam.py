"""Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GradCAM:
    """Computes Grad-CAM saliency maps for a given model and target layer.

    The target layer must produce a 4-D feature map (B, C, H, W).
    A single forward+backward pass is used per image.

    Args:
        model: PyTorch model in eval mode.
        target_layer: Convolutional layer whose activations/gradients are used.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: Tensor | None = None
        self._gradients: Tensor | None = None

        self._fwd_handle = target_layer.register_forward_hook(self._save_activations)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradients)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _save_activations(
        self,
        module: nn.Module,
        input: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        """Forward hook — stores the layer output (detached)."""
        self._activations = output.detach()

    def _save_gradients(
        self,
        module: nn.Module,
        grad_input: tuple[Tensor, ...],
        grad_output: tuple[Tensor, ...],
    ) -> None:
        """Backward hook — stores the gradient w.r.t. the layer output."""
        self._gradients = grad_output[0].detach()

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def __call__(self, x: Tensor, class_idx: int | None = None) -> np.ndarray:
        """Compute the Grad-CAM map for a single input image.

        Args:
            x: Input tensor of shape (1, C, H, W), already on the model's device.
            class_idx: Target class index. If None, uses the predicted class.

        Returns:
            Normalised CAM of shape (H_in, W_in) with values in [0, 1].
        """
        assert x.shape[0] == 1, "GradCAM expects a single image (batch size 1)"
        h_in, w_in = x.shape[2], x.shape[3]

        self.model.zero_grad()
        x = x.requires_grad_(False)  # we only need grads on the activations

        logits: Tensor = self.model(x)  # (1, num_classes)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        # Scalar target — gradient flows back to the target layer
        score: Tensor = logits[0, class_idx]
        score.backward()

        # weights = global-average-pooled gradients  (C,)
        grads: Tensor = self._gradients  # (1, C, h, w)
        acts: Tensor = self._activations  # (1, C, h, w)

        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        cam: Tensor = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Upsample to input spatial size
        cam = F.interpolate(
            cam,
            size=(h_in, w_in),
            mode="bilinear",
            align_corners=False,
        )  # (1, 1, H_in, W_in)

        cam_np: np.ndarray = cam.squeeze().cpu().numpy()  # (H_in, W_in)

        # Normalise to [0, 1]
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

    # ------------------------------------------------------------------
    # Visualisation helper
    # ------------------------------------------------------------------

    @staticmethod
    def overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Alpha-blend a Grad-CAM heatmap onto an RGB image.

        Args:
            img: Original image as uint8 array of shape (H, W, 3).
            cam: Grad-CAM map of shape (H, W), values in [0, 1].
            alpha: Weight of the heatmap overlay (0 = image only, 1 = heatmap only).

        Returns:
            Blended image as uint8 array of shape (H, W, 3).
        """
        import matplotlib.pyplot as plt  # lazy import — optional dependency

        colormap = plt.get_cmap("jet")
        heatmap: np.ndarray = (colormap(cam)[:, :, :3] * 255).astype(np.uint8)

        img_f = img.astype(np.float32)
        heat_f = heatmap.astype(np.float32)
        blended = (1.0 - alpha) * img_f + alpha * heat_f
        return np.clip(blended, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def remove_hooks(self) -> None:
        """Remove the registered forward and backward hooks."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Return the canonical Grad-CAM target layer for a given model.

    Supported model names (case-insensitive prefix match):
    - "resnet"    → model.layer4[-1].conv2  (last residual conv)
    - "mobilenet" → model.conv2             (final expansion conv, 1280 ch)

    Args:
        model: Instantiated model.
        model_name: Architecture name string (e.g. "resnet", "mobilenet").

    Returns:
        The target nn.Module layer.

    Raises:
        ValueError: If the model name is not recognised.
    """
    name = model_name.lower()
    if name.startswith("resnet"):
        return model.layer4[-1].conv2
    if name.startswith("mobilenet"):
        return model.conv2
    raise ValueError(
        f"Unrecognised model name '{model_name}' for GradCAM target layer selection. "
        "Supported: 'resnet', 'mobilenet'."
    )
