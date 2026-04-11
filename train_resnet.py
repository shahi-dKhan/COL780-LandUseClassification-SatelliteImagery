import argparse
import multiprocessing as mp
from Resnet import ResNetClassifier


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-18 for Land Use Classification")
    parser.add_argument("--img_dir",    required=True, help="Path to image directory")
    parser.add_argument("--train_csv",  required=True, help="Path to train CSV")
    parser.add_argument("--val_csv",    required=True, help="Path to validation CSV")
    parser.add_argument("--test_csv",   required=True, help="Path to test CSV")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--use_focal",  action="store_true", help="Use Focal Loss instead of CrossEntropy")
    parser.add_argument("--focal_alpha",type=float, default=1.0)
    parser.add_argument("--focal_gamma",type=float, default=2.0)
    parser.add_argument("--use_scheduler", action="store_true", help="Use LR scheduler")
    parser.add_argument("--model_path", default="best_resnet_model.pth", help="Where to save the best model")
    args = parser.parse_args()

    model = ResNetClassifier(
        args.img_dir, args.train_csv, args.val_csv, args.test_csv,
        batch_size=args.batch_size,
        lr=args.lr,
        use_Focal=args.use_focal,
    )
    model.best_model_path = args.model_path

    model.fit(num_epochs=args.epochs)
    print(f"\nBest model saved to: {args.model_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
