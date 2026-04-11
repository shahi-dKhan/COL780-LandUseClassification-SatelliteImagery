import argparse
import multiprocessing as mp
import torch
from SnE import ResNetSEClassifier


def main():
    parser = argparse.ArgumentParser(description="Test ResNet-18 + SE Blocks for Land Use Classification")
    parser.add_argument("--img_dir",    required=True, help="Path to image directory")
    parser.add_argument("--test_csv",   required=True, help="Path to test CSV")
    parser.add_argument("--model_path", required=True, help="Path to saved .pth weights")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # Pass test_csv for all splits — only test_loader is used here
    model = ResNetSEClassifier(
        args.img_dir, args.test_csv, args.test_csv, args.test_csv,
        batch_size=args.batch_size,
    )
    model.model.load_state_dict(torch.load(args.model_path, map_location=model.device))

    test_f1, test_auc, test_acc = model.evaluate(model.test_loader)
    print(f"Test AUC: {test_auc:.4f} | Test F1: {test_f1:.4f} | Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
