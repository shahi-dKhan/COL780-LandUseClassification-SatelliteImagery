"""
Run inference on all saved .pth models and print results in table format.
Fills in missing Accuracy values for Tables 3 & 4 in the report, and
resolves the DeiT-3 CE inconsistency between Tables 9 and 14.

Run from the COL780-LandUseClassification-SatelliteImagery/ directory:
    python test_all_models.py --img_dir A3_Dataset --test_csv A3_Dataset/test.csv
"""

import argparse
import torch
import os
from Resnet import ResNetClassifier
from SnE import ResNetSEClassifier
from ViT import DieTClassifier as DeiTClassifier
from nonormdeit import DieTClassifier as DyTClassifier


def test_model(model_obj, model_path, device):
    model_obj.model.load_state_dict(torch.load(model_path, map_location=device))
    f1, auc, acc = model_obj.evaluate(model_obj.test_loader)
    return f1, auc, acc


def print_row(label, f1, auc, acc):
    print(f"  {label:<45} AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir",  required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # Paths to model folders (relative to this script)
    base = os.path.dirname(os.path.abspath(__file__))
    resnet_dir    = os.path.join(base, "..", "Resnet")
    sne_dir       = os.path.join(base, "..", "SnE")
    deit_dir      = os.path.join(base, "..", "deit")
    deitnonorm_dir= os.path.join(base, "..", "deitnonorm")

    # ----------------------------------------------------------------
    # TABLE 3 & 4  —  ResNet-18 Focal Loss (missing: Accuracy column)
    # ----------------------------------------------------------------
    print("\n" + "="*70)
    print("RESNET-18 FOCAL LOSS ABLATION  (Tables 3 & 4)")
    print("="*70)

    resnet_configs = [
        ("CE Baseline",               "best_resnet_model.pth",             False, None, None),
        ("Focal α=0.25 γ=1.0",        "best_focal_a0.25_g1.0 (1).pth",     True, 0.25, 1.0),
        ("Focal α=0.25 γ=2.0",        "best_focal_a0.25_g2.0 (1).pth",     True, 0.25, 2.0),
        ("Focal α=0.25 γ=5.0",        "best_focal_a0.25_g5.0 (1).pth",     True, 0.25, 5.0),
        ("Focal α=0.50 γ=1.0",        "best_focal_a0.5_g1.0 (1).pth",      True, 0.50, 1.0),
        ("Focal α=0.50 γ=2.0",        "best_focal_a0.5_g2.0 (1).pth",      True, 0.50, 2.0),
        ("Focal α=0.50 γ=5.0",        "best_focal_a0.5_g5.0 (1).pth",      True, 0.50, 5.0),
        ("Focal α=1.00 γ=1.0",        "best_focal_a1.0_g1.0 (1).pth",      True, 1.00, 1.0),
        ("Focal α=1.00 γ=2.0",        "best_focal_a1.0_g2.0 (1).pth",      True, 1.00, 2.0),
        ("Focal α=1.00 γ=5.0",        "best_focal_a1.0_g5.0 (1).pth",      True, 1.00, 5.0),
        ("Focal α=0.25 γ=2.0 + Sched","best_focal_with_scheduler (1).pth", True, 0.25, 2.0),
    ]

    for label, fname, use_focal, alpha, gamma in resnet_configs:
        path = os.path.join(resnet_dir, fname)
        if not os.path.exists(path):
            print(f"  [SKIP] {label} — file not found: {path}")
            continue
        m = ResNetClassifier(
            args.img_dir, args.test_csv, args.test_csv, args.test_csv,
            batch_size=args.batch_size,
            use_Focal=use_focal,
        )
        f1, auc, acc = test_model(m, path, m.device)
        print_row(label, f1, auc, acc)

    # ----------------------------------------------------------------
    # SnE  —  CE Baseline (already reported, just verify)
    # ----------------------------------------------------------------
    print("\n" + "="*70)
    print("RESNET-18 + SE  (verification)")
    print("="*70)

    sne_path = os.path.join(sne_dir, "best_sne_model.pth")
    if os.path.exists(sne_path):
        m = ResNetSEClassifier(
            args.img_dir, args.test_csv, args.test_csv, args.test_csv,
            batch_size=args.batch_size,
        )
        f1, auc, acc = test_model(m, sne_path, m.device)
        print_row("CE Baseline", f1, auc, acc)

    # ----------------------------------------------------------------
    # DeiT-3  —  resolve Table 9 vs Table 14 inconsistency
    # ----------------------------------------------------------------
    print("\n" + "="*70)
    print("DEIT-3 SMALL  (resolving inconsistency between Tables 9 & 14)")
    print("="*70)

    deit_configs = [
        ("deit/ baseline CE",          os.path.join(deit_dir,       "best_deit3_baseline_ce.pth"),   False),
        ("deit/ focal a=1.0 g=2.0",   os.path.join(deit_dir,       "best_deit3_focal_a1.0_g2.0.pth"), True),
        ("deitnonorm/ deit_baseline",  os.path.join(deitnonorm_dir, "best_deit_baseline.pth"),        False),
    ]

    for label, path, use_focal in deit_configs:
        if not os.path.exists(path):
            print(f"  [SKIP] {label} — file not found")
            continue
        m = DeiTClassifier(
            args.img_dir, args.test_csv, args.test_csv, args.test_csv,
            batch_size=args.batch_size,
            use_focal=use_focal,
        )
        f1, auc, acc = test_model(m, path, m.device)
        print_row(label, f1, auc, acc)

    # ----------------------------------------------------------------
    # DeiT-3 + DyT  —  verify Table 13
    # ----------------------------------------------------------------
    print("\n" + "="*70)
    print("DEIT-3 + DyT  (verification)")
    print("="*70)

    dyt_path = os.path.join(deitnonorm_dir, "best_deit_dyt_ablation.pth")
    if os.path.exists(dyt_path):
        m = DyTClassifier(
            args.img_dir, args.test_csv, args.test_csv, args.test_csv,
            batch_size=args.batch_size,
            use_dyt=True,
        )
        f1, auc, acc = test_model(m, dyt_path, m.device)
        print_row("DyT CE", f1, auc, acc)

    print("\nDone.")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
