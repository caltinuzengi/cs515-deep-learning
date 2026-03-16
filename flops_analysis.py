"""FLOPs and parameter count comparison for all model architectures.

Uses ``ptflops`` to measure computational complexity.  Run directly::

    python flops_analysis.py
"""

from ptflops import get_model_complexity_info

from models.CNN import SimpleCNN
from models.ResNet import ResNet, BasicBlock
from models.mobilenet import MobileNetV2


def main() -> None:
    models_cfg = [
        ("SimpleCNN",        SimpleCNN(num_classes=10),                     (3, 32, 32)),
        ("ResNet-18",        ResNet(BasicBlock, [2,2,2,2], num_classes=10), (3, 32, 32)),
        ("MobileNetV2",      MobileNetV2(num_classes=10),                   (3, 32, 32)),
    ]

    print(f"{'Model':<20} {'Params':>12} {'MACs':>14}")
    print("-" * 48)

    for name, model, input_size in models_cfg:
        macs, params = get_model_complexity_info(
            model, input_size, as_strings=True,
            print_per_layer_stat=False, verbose=False,
        )
        print(f"{name:<20} {params:>12} {macs:>14}")


if __name__ == "__main__":
    main()
