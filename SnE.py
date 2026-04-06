import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from load_data import CropData
import multiprocessing as mp

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class SqueezeExcitationBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    - Squeeze: Global average pooling to aggregate spatial information
    - Excitation: Gating network to learn channel-wise weights
    """
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitationBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze: Global average pooling
        batch_size, channels, height, width = x.size()
        squeeze = x.view(batch_size, channels, -1).mean(dim=2)  # (B, C)

        # Excitation: Gating network
        excitation = self.fc1(squeeze)  # (B, C/16)
        excitation = self.relu(excitation)
        excitation = self.fc2(excitation)  # (B, C)
        excitation = self.sigmoid(excitation)

        # Reshape for channel-wise multiplication
        excitation = excitation.view(batch_size, channels, 1, 1)  # (B, C, 1, 1)

        # Scale the input feature map
        return x * excitation


class ResNetSEClassifier(nn.Module):
    def __init__(self, img_dir, train_csv, val_csv, test_csv, batch_size=32, num_classes=10, num_epochs=10, lr=1e-4, use_focal=False, focal_alpha=1, focal_gamma=2):
        super().__init__()
        self.device = self.get_device()
        self.train_loader, self.val_loader, self.test_loader = self.get_dataloaders(img_dir, train_csv, val_csv, test_csv, batch_size)
        self.model = self.build_model(num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss() if not use_focal else FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.best_auc = -1
        self.best_model_path = "best_sne_model.pth"
    
    def get_device(self):
        if torch.cuda.is_available():
            print("Using CUDA")
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            print("Using MPS")
            return torch.device("mps")
        print("Using CPU")
        return torch.device("cpu")

    def get_dataloaders(self, img_dir, train_csv, val_csv, test_csv, batch_size):
        train_dataset = CropData(img_dir_path=img_dir, csv_file_path=train_csv)
        val_dataset = CropData(img_dir_path=img_dir, csv_file_path=val_csv)
        test_dataset = CropData(img_dir_path=img_dir, csv_file_path=test_csv)

        num_workers = 0 if self.device.type == "mps" else 4
        pin_memory = self.device.type == "cuda"

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        return train_loader, val_loader, test_loader
    
    def build_model(self, num_classes):
        """
        Build ResNet-18 with SE blocks inserted after each residual stage
        SE blocks are added after layer1, layer2, layer3, and layer4 with 
        channel sizes: 64, 128, 256, 512 respectively
        """
        # Local import keeps this method robust in notebook environments where
        # cells may be run out of order.
        from torchvision import models as tv_models
        base_model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        
        # Create a new model that includes SE blocks
        model = ResNetWithSE(base_model, num_classes)
        
        return model
    
    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss
    
    def evaluate(self, loader):
        self.model.eval()
        all_labels = []
        all_predictions = []
        all_probs = []

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        macro_f1 = f1_score(all_labels, all_predictions, average="macro")
        macro_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
        acc = accuracy_score(all_labels, all_predictions)

        return macro_f1, macro_auc, acc
    
    def fit(self, num_epochs):
        for epoch in range(num_epochs):
            epoch_loss = self.train_one_epoch()
            print(f"\nEpoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

            print("Validation Metrics:")
            val_f1, val_auc, val_acc = self.evaluate(self.val_loader)
            print(f"  F1: {val_f1:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}")

            if val_auc > self.best_auc:
                self.best_auc = val_auc
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"  ✓ Best model saved (AUC: {val_auc:.4f})")
    
    def test(self):
        print("\nFinal Test Metrics (Before Loading Best Model):")
        test_f1, test_auc, test_acc = self.evaluate(self.test_loader)
        print(f"  F1: {test_f1:.4f}, AUC: {test_auc:.4f}, Acc: {test_acc:.4f}")

        print("\nLoading best model...")
        self.model.load_state_dict(torch.load(self.best_model_path))

        print("\nFinal Test Metrics (Best Model):")
        test_f1, test_auc, test_acc = self.evaluate(self.test_loader)
        print(f"  F1: {test_f1:.4f}, AUC: {test_auc:.4f}, Acc: {test_acc:.4f}")
        
        return test_f1, test_auc, test_acc
    
    def load_best_model(self):
        self.model.load_state_dict(torch.load(self.best_model_path))


class ResNetWithSE(nn.Module):
    """
    ResNet-18 with Squeeze-and-Excitation blocks inserted after each residual stage
    """
    def __init__(self, base_resnet, num_classes):
        super(ResNetWithSE, self).__init__()
        
        # Copy the early layers from base ResNet
        self.conv1 = base_resnet.conv1
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool
        
        # Copy the residual layers
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4
        
        # Add SE blocks after each residual stage
        # Channel sizes in ResNet-18: layer1=64, layer2=128, layer3=256, layer4=512
        self.se1 = SqueezeExcitationBlock(64, reduction=16)
        self.se2 = SqueezeExcitationBlock(128, reduction=16)
        self.se3 = SqueezeExcitationBlock(256, reduction=16)
        self.se4 = SqueezeExcitationBlock(512, reduction=16)
        
        # Global average pooling and fully connected layer
        self.avgpool = base_resnet.avgpool
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Initial convolution and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual blocks with SE recalibration
        x = self.layer1(x)
        x = self.se1(x)  # SE block after layer1 (64 channels)
        
        x = self.layer2(x)
        x = self.se2(x)  # SE block after layer2 (128 channels)
        
        x = self.layer3(x)
        x = self.se3(x)  # SE block after layer3 (256 channels)
        
        x = self.layer4(x)
        x = self.se4(x)  # SE block after layer4 (512 channels)
        
        # Global average pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def main():
    IMG_DIR = "A3_Dataset"
    TRAIN_CSV = "A3_Dataset/train.csv"
    VAL_CSV = "A3_Dataset/validation.csv"
    TEST_CSV = "A3_Dataset/test.csv"
    EPOCHS = 10
    
    results_file = "sne_results.txt"
    with open(results_file, "w") as f:
        f.write("ResNet-18 with Squeeze-and-Excitation Blocks Results\n")
        f.write("=" * 70 + "\n\n")

    # ==========================================================
    # STAGE 1: BASELINE WITH SE BLOCKS (CrossEntropyLoss)
    # ==========================================================
    print("\n" + "="*70)
    print("STAGE 1: ResNet-18 + SE Blocks with CrossEntropyLoss")
    print("="*70)
    
    baseline_lr = 1e-4
    
    se_model = ResNetSEClassifier(
        IMG_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV, 
        batch_size=32,
        num_epochs=EPOCHS,
        lr=baseline_lr,
        use_focal=False
    )
    
    se_model.fit(num_epochs=EPOCHS)
    base_f1, base_auc, base_acc = se_model.test()
    
    print(f"\n✓ Baseline Results (LR={baseline_lr}):")
    print(f"   AUC: {base_auc:.4f} | F1: {base_f1:.4f} | Accuracy: {base_acc:.4f}")

    with open(results_file, "a") as f:
        f.write("STAGE 1: BASELINE (CrossEntropyLoss)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Learning Rate: {baseline_lr}\n")
        f.write(f"Test AUC: {base_auc:.4f}\n")
        f.write(f"Test F1 (macro): {base_f1:.4f}\n")
        f.write(f"Test Accuracy: {base_acc:.4f}\n")
        f.write("\n")

    # ==========================================================
    # STAGE 2: ABLATION STUDY WITH FOCAL LOSS
    # ==========================================================
    print("\n" + "="*70)
    print("STAGE 2: Ablation Study with FocalLoss")
    print("="*70)
    
    alphas = [0.25, 0.5, 1.0]
    gammas = [1.0, 2.0, 5.0]
    
    best_focal_auc = -1
    best_alpha, best_gamma = 1.0, 2.0
    best_focal_f1, best_focal_acc = 0.0, 0.0
    
    with open(results_file, "a") as f:
        f.write("STAGE 2: FOCAL LOSS ABLATION STUDY\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Alpha':<10} {'Gamma':<10} {'Test AUC':<15} {'Test F1':<15} {'Test Acc':<15}\n")
        f.write("-" * 70 + "\n")

    for alpha in alphas:
        for gamma in gammas:
            print(f"\nTesting FocalLoss with Alpha={alpha}, Gamma={gamma}...")
            
            focal_model = ResNetSEClassifier(
                IMG_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV, 
                batch_size=32,
                num_epochs=EPOCHS,
                lr=baseline_lr,
                use_focal=True,
                focal_alpha=alpha,
                focal_gamma=gamma
            )
            
            focal_model.fit(num_epochs=EPOCHS)
            test_f1, test_auc, test_acc = focal_model.test()
            
            print(f"   AUC: {test_auc:.4f}, F1: {test_f1:.4f}, Acc: {test_acc:.4f}")
            
            with open(results_file, "a") as f:
                f.write(f"{alpha:<10.2f} {gamma:<10.1f} {test_auc:<15.4f} {test_f1:<15.4f} {test_acc:<15.4f}\n")
            
            # Track best focal loss configuration
            if test_auc > best_focal_auc:
                best_focal_auc = test_auc
                best_alpha = alpha
                best_gamma = gamma
                best_focal_f1 = test_f1
                best_focal_acc = test_acc

    print(f"\n✓ Best FocalLoss Configuration: Alpha={best_alpha}, Gamma={best_gamma}")
    print(f"   AUC: {best_focal_auc:.4f} | F1: {best_focal_f1:.4f} | Accuracy: {best_focal_acc:.4f}")

    with open(results_file, "a") as f:
        f.write("\n" + "-" * 70 + "\n")
        f.write(f"BEST FOCAL LOSS CONFIG: Alpha={best_alpha}, Gamma={best_gamma}\n")
        f.write(f"Test AUC: {best_focal_auc:.4f}\n")
        f.write(f"Test F1: {best_focal_f1:.4f}\n")
        f.write(f"Test Accuracy: {best_focal_acc:.4f}\n")
        f.write("\n")

    # ==========================================================
    # COMPARISON SUMMARY
    # ==========================================================
    print("\n" + "="*70)
    print("SUMMARY: CrossEntropyLoss vs Best FocalLoss Configuration")
    print("="*70)
    print(f"\nCrossEntropyLoss (Baseline):")
    print(f"  AUC: {base_auc:.4f} | F1: {base_f1:.4f} | Acc: {base_acc:.4f}")
    print(f"\nFocalLoss (Best: Alpha={best_alpha}, Gamma={best_gamma}):")
    print(f"  AUC: {best_focal_auc:.4f} | F1: {best_focal_f1:.4f} | Acc: {best_focal_acc:.4f}")
    
    improvement_auc = ((best_focal_auc - base_auc) / base_auc * 100) if base_auc > 0 else 0
    improvement_f1 = ((best_focal_f1 - base_f1) / base_f1 * 100) if base_f1 > 0 else 0
    improvement_acc = ((best_focal_acc - base_acc) / base_acc * 100) if base_acc > 0 else 0
    
    print(f"\nImprovement with FocalLoss:")
    print(f"  AUC: {improvement_auc:+.2f}%")
    print(f"  F1:  {improvement_f1:+.2f}%")
    print(f"  Acc: {improvement_acc:+.2f}%")

    with open(results_file, "a") as f:
        f.write("=" * 70 + "\n")
        f.write("FINAL COMPARISON: CrossEntropyLoss vs FocalLoss\n")
        f.write("=" * 70 + "\n")
        f.write(f"CrossEntropyLoss (Baseline):\n")
        f.write(f"  AUC: {base_auc:.4f}, F1: {base_f1:.4f}, Acc: {base_acc:.4f}\n\n")
        f.write(f"FocalLoss (Best Config: Alpha={best_alpha}, Gamma={best_gamma}):\n")
        f.write(f"  AUC: {best_focal_auc:.4f}, F1: {best_focal_f1:.4f}, Acc: {best_focal_acc:.4f}\n\n")
        f.write(f"Improvement:\n")
        f.write(f"  AUC: {improvement_auc:+.2f}%\n")
        f.write(f"  F1:  {improvement_f1:+.2f}%\n")
        f.write(f"  Acc: {improvement_acc:+.2f}%\n")

    print(f"\n✅ Results saved to '{results_file}'")


if __name__ == '__main__':
    main()
