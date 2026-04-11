import argparse
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from load_data import CropData
from Resnet import ResNetClassifier


def load_checkpoint_state(checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict):
        for key in ["state_dict", "model_state_dict", "model"]:
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


def load_inverse_label_map(label_map_path):
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    return {idx: name for name, idx in label_map.items()}


def sample_per_class(dataset):
    df = dataset.df
    samples = []
    for label in sorted(df["Label"].unique()):
        idx = int(df.index[df["Label"] == label][0])
        filename = df.iloc[idx]["Filename"]
        samples.append((int(label), filename, idx))
    return samples


def read_display_image(img_path):
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def build_visualization_strip(image_rgb, cam_map):
    height, width = image_rgb.shape[:2]
    cam_resized = cv2.resize(cam_map, (width, height), interpolation=cv2.INTER_LINEAR)

    cam_uint8 = np.uint8(np.clip(cam_resized, 0.0, 1.0) * 255)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.addWeighted(image_rgb, 0.55, heatmap_rgb, 0.45, 0)

    strip_rgb = np.concatenate([image_rgb, heatmap_rgb, overlay_rgb], axis=1)
    return cv2.cvtColor(strip_rgb, cv2.COLOR_RGB2BGR)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        pred_class = int(logits.argmax(dim=1).item())

        score = logits[:, pred_class]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)
        cam = cam.squeeze(0)

        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam.cpu().numpy(), pred_class


class ResNetBonus(ResNetClassifier):
    def __init__(
        self,
        img_dir,
        train_csv,
        val_csv,
        test_csv,
        checkpoint_path,
        batch_size=32,
        num_classes=10,
    ):
        super().__init__(
            img_dir=img_dir,
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            batch_size=batch_size,
            num_classes=num_classes,
            lr=1e-4,
            use_Focal=False,
        )

        state_dict = load_checkpoint_state(checkpoint_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.grad_cam = GradCAM(self.model, self.model.layer4[-1].conv2)
        self.img_dir = img_dir
        self.test_dataset = CropData(img_dir_path=img_dir, csv_file_path=test_csv)

    def run_bonus(self, label_map_path, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        idx_to_name = load_inverse_label_map(label_map_path)
        selected_samples = sample_per_class(self.test_dataset)

        print("Generating Grad-CAM visualizations for ResNet bonus task...")
        for true_label, filename, dataset_idx in selected_samples:
            img_path = os.path.join(self.img_dir, filename)
            image_rgb = read_display_image(img_path)
            image_tensor, _ = self.test_dataset[dataset_idx]
            image_tensor = image_tensor.unsqueeze(0).to(self.device)

            cam_map, pred_label = self.grad_cam.generate(image_tensor)
            strip_bgr = build_visualization_strip(image_rgb, cam_map)

            true_name = idx_to_name.get(true_label, str(true_label))
            pred_name = idx_to_name.get(pred_label, str(pred_label))
            out_name = f"{true_label:02d}_{true_name}_pred_{pred_label:02d}_{pred_name}.jpg"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, strip_bgr)
            print(f"Saved: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Bonus Task 3.1: Grad-CAM for ResNet model")
    parser.add_argument("--img_dir", default="A3_Dataset", help="Root directory containing image folders")
    parser.add_argument("--train_csv", default="A3_Dataset/train.csv", help="Train CSV path")
    parser.add_argument("--val_csv", default="A3_Dataset/validation.csv", help="Validation CSV path")
    parser.add_argument("--test_csv", default="A3_Dataset/test.csv", help="Test CSV path")
    parser.add_argument("--label_map", default="A3_Dataset/label_map.json", help="Label map JSON path")
    parser.add_argument(
        "--checkpoint",
        default="best_resnet_model.pth",
        help="Path to trained Task 1.1 ResNet checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        default="bonus_outputs/resnet_gradcam",
        help="Directory to save Grad-CAM output images",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model = ResNetBonus(
        img_dir=args.img_dir,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        checkpoint_path=args.checkpoint,
    )
    model.run_bonus(label_map_path=args.label_map, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
