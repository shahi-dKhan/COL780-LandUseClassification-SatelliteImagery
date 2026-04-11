import argparse
import multiprocessing as mp
from ViT import DieTClassifier


def main():
    parser = argparse.ArgumentParser(description="Train DeiT-3 Small for Land Use Classification")
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
    parser.add_argument("--run_name",   default="deit3", help="Used to name the saved .pth file")
    args = parser.parse_args()

    model = DieTClassifier(
        args.img_dir, args.train_csv, args.val_csv, args.test_csv,
        batch_size=args.batch_size,
        lr=args.lr,
        use_focal=args.use_focal,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        run_name=args.run_name,
    )

    model.fit(num_epochs=args.epochs)
    print(f"\nBest model saved to: {model.best_model_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
