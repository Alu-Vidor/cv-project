"""
Download the Food-101 dataset to the data directory and print class labels.
"""
from pathlib import Path

from torchvision.datasets import Food101


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset = Food101(root=str(data_dir), download=True)

    print("Found", len(dataset.classes), "classes:")
    for label in dataset.classes:
        print(label)


if __name__ == "__main__":
    main()
