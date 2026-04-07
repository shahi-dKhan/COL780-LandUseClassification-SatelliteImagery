import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from load_data import CropData
import multiprocessing as mp

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
    
class DieTClassifier(nn.Module):
    def __init__(self, img_dir, train_csv, val_csv, test_csv, batch_size=32, num_classes=10, 
                 lr=1e-4, use_focal=False, focal_alpha=1.0, focal_gamma=2.0, run_name="deit3"):
        super().__init__()
        self.run_name = run_name
        self.device = self.get_device()
        self.train_loader, self.val_loader, self.test_loader = self.get_dataloaders(img_dir, train_csv, val_csv, test_csv, batch_size)
        self.model = self.build_model(num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss() if not use_focal else FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.best_auc = -1
        self.best_model_path = f"best_{self.run_name}.pth"
        
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
    
    def build_model(self, num_classes=10):
        print("Building DeiT-3 Small...")
        # timm.create_model handles prepending the [CLS] token and passes its 
        # representation to the final replaced linear classification head automatically 
        # when we specify num_classes.
        model = timm.create_model('deit3_small_patch16_224', pretrained=True, num_classes=num_classes)
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
            
        return running_loss / len(self.train_loader)
    
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
        train_losses = []
        val_aucs = []
        
        for epoch in range(num_epochs):
            epoch_loss = self.train_one_epoch()
            train_losses.append(epoch_loss)
            
            val_f1, val_auc, val_acc = self.evaluate(self.val_loader)
            val_aucs.append(val_auc)
            
            print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}")

            # Early stopping via best validation AUC
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                print("Saving a new Model")
                torch.save(self.model.state_dict(), self.best_model_path)
                
        return train_losses, val_aucs
    
    def test(self):
        print(f"\nLoading best model from {self.best_model_path} for testing...")
        self.model.load_state_dict(torch.load(self.best_model_path))

        test_f1, test_auc, test_acc = self.evaluate(self.test_loader)
        print(f"Test Metrics -> AUC: {test_auc:.4f} | F1: {test_f1:.4f} | Acc: {test_acc:.4f}")
        return test_f1, test_auc, test_acc
    
    

def main():
    IMG_DIR = "path/to/images"
    TRAIN_CSV = "path/to/train.csv"
    VAL_CSV = "path/to/val.csv"
    TEST_CSV = "path/to/test.csv"
    
    EPOCHS = 10
    LR = 1e-4
    BATCH_SIZE = 32
    
    results_file = "deit3_results.txt"
    with open(results_file, "w") as f:
        f.write("DeiT-3 Small Fine-Tuning Experiments\n")
        f.write("="*60 + "\n")

    # ==========================================================
    # EXPERIMENT 1: DeiT-3 Baseline (CrossEntropy)
    # ==========================================================
    print("\n--- EXPERIMENT 1: DeiT-3 with CrossEntropy ---")
    deit_baseline = DieTClassifier(
        IMG_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV, 
        batch_size=BATCH_SIZE, lr=LR, 
        use_focal=False, run_name="deit3_baseline_ce"
    )
    
    deit_baseline.fit(num_epochs=EPOCHS)
    base_f1, base_auc, base_acc = deit_baseline.test()
    
    with open(results_file, "a") as f:
        f.write("EXPERIMENT 1: DeiT-3 Small (CrossEntropy)\n")
        f.write(f"Learning Rate: {LR} | Batch Size: {BATCH_SIZE}\n")
        f.write(f"Test AUC: {base_auc:.4f} | Test F1: {base_f1:.4f} | Test Acc: {base_acc:.4f}\n\n")

    # ==========================================================
    # EXPERIMENT 2: Ablation Study - DeiT-3 with Focal Loss
    # ==========================================================
    print("\n--- EXPERIMENT 2: DeiT-3 Ablation Study (Focal Loss) ---")
    
    # Example hyperparameters for Focal Loss Ablation
    focal_alpha, focal_gamma = 1.0, 2.0
    print(f"Testing Focal Loss -> Alpha={focal_alpha}, Gamma={focal_gamma}")
    
    deit_focal = DieTClassifier(
        IMG_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV, 
        batch_size=BATCH_SIZE, lr=LR, 
        use_focal=True, focal_alpha=focal_alpha, focal_gamma=focal_gamma, 
        run_name=f"deit3_focal_a{focal_alpha}_g{focal_gamma}"
    )
    
    deit_focal.fit(num_epochs=EPOCHS)
    focal_f1, focal_auc, focal_acc = deit_focal.test()
    
    with open(results_file, "a") as f:
        f.write("EXPERIMENT 2: DeiT-3 Small (Focal Loss Ablation)\n")
        f.write(f"Alpha: {focal_alpha} | Gamma: {focal_gamma}\n")
        f.write(f"Test AUC: {focal_auc:.4f} | Test F1: {focal_f1:.4f} | Test Acc: {focal_acc:.4f}\n\n")

    print("\n✅ Training Complete! Check 'deit3_results.txt' for final logs.")

if __name__ == '__main__':
    # Context set for Dataloaders to prevent mp deadlocks
    mp.set_start_method('spawn', force=True)
    main()
    
    
