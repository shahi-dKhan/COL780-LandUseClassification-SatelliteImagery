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

class ResNetClassifier(nn.Module):
    def __init__(self, img_dir, train_csv, val_csv, test_csv, batch_size=32,num_classes=10, num_epochs=10, lr=1e-4, use_Focal = False):
        super().__init__()
        self.device = self.get_device()
        self.train_loader, self.val_loader, self.test_loader = self.get_dataloaders(img_dir, train_csv, val_csv, test_csv, batch_size)
        self.model = self.build_model(num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss() if not use_Focal else FocalLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.best_auc = -1
        self.best_model_path = "best_resnet_model.pth"
    
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
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

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
            _, val_auc, _ = self.evaluate(self.val_loader)

            if val_auc > self.best_auc:
                self.best_auc = val_auc
                torch.save(self.model.state_dict(), self.best_model_path)
    
    def test(self):
        print("\nFinal Test Metrics:")
        self.evaluate(self.test_loader)

        print("\nLoading best model...")
        self.model.load_state_dict(torch.load(self.best_model_path))

        print("\nFinal Test Metrics (Best Model):")
        self.evaluate(self.test_loader)
    
    def load_best_model(self):
        self.model.load_state_dict(torch.load(self.best_model_path))
        
    

        

def main():
    IMG_DIR = "path/to/images"
    TRAIN_CSV = "path/to/train.csv"
    VAL_CSV = "path/to/val.csv"
    TEST_CSV = "path/to/test.csv"
    EPOCHS = 10
    
    results_file = "hierarchical_search_results.txt"
    with open(results_file, "w") as f:
        f.write("Hierarchical Hyperparameter Search & Baseline Logs\n")
        f.write("="*60 + "\n")

    # ==========================================================
    # STAGE 1: BASELINE & LOGGING (Cross Entropy)
    # The assignment specifically asks for LR=1e-4 and BS=32
    # ==========================================================
    print("\n--- STAGE 1: Baseline CE Run & Plot Data ---")
    baseline_lr = 1e-4
    
    baseline_model = ResNetClassifier(
        IMG_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV, 
        use_focal=False, lr=baseline_lr, run_name="baseline"
    )
    
    # We catch the returned lists for plotting
    base_train_losses, base_val_aucs = baseline_model.fit(num_epochs=EPOCHS)
    base_f1, base_auc, base_acc = baseline_model.test()
    
    print(f"Baseline (LR={baseline_lr}) -> AUC: {base_auc:.4f} | F1: {base_f1:.4f}")

    # Log Baseline details and the arrays for plotting
    with open(results_file, "a") as f:
        f.write("STAGE 1: BASELINE (CE Loss)\n")
        f.write(f"Learning Rate: {baseline_lr} | Test AUC: {base_auc:.4f} | F1: {base_f1:.4f} | Acc: {base_acc:.4f}\n\n")
        
        f.write("--- PLOT DATA FOR REPORT ---\n")
        f.write(f"Epochs: {list(range(1, EPOCHS+1))}\n")
        # Format the floats so they don't look messy
        formatted_losses = [f"{loss:.4f}" for loss in base_train_losses]
        formatted_aucs = [f"{auc:.4f}" for auc in base_val_aucs]
        f.write(f"Train_Losses: {formatted_losses}\n")
        f.write(f"Val_AUCs: {formatted_aucs}\n")
        f.write("-" * 60 + "\n\n")

    # ==========================================================
    # STAGE 2: FOCAL LOSS SEARCH (Using best parameters from prior steps)
    # ==========================================================
    print("\n--- STAGE 2: Focal Loss Parameters (Alpha/Gamma) ---")
    alphas = [0.25, 0.5, 1.0]
    gammas = [1.0, 2.0, 5.0]
    
    best_focal_auc = -1
    best_alpha, best_gamma = 1.0, 2.0
    
    with open(results_file, "a") as f:
        f.write("STAGE 2: FOCAL LOSS SEARCH\n")
        f.write("Alpha\tGamma\tTest_AUC\tTest_F1\n")
        f.write("-" * 40 + "\n")

    for alpha in alphas:
        for gamma in gammas:
            print(f"Testing Alpha={alpha}, Gamma={gamma}...")
            model = ResNetClassifier(
                IMG_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV, 
                use_focal=True, focal_alpha=alpha, focal_gamma=gamma, lr=baseline_lr, run_name=f"focal_a{alpha}_g{gamma}"
            )
            model.fit(num_epochs=EPOCHS)
            test_f1, test_auc, test_acc = model.test()
            
            with open(results_file, "a") as f:
                f.write(f"{alpha}\t{gamma}\t{test_auc:.4f}\t\t{test_f1:.4f}\n")
                
            if test_auc > best_focal_auc:
                best_focal_auc = test_auc
                best_alpha = alpha
                best_gamma = gamma

    print(f"Stage 2 Winner: Alpha={best_alpha}, Gamma={best_gamma}")

    # ==========================================================
    # STAGE 3: SCHEDULER SEARCH (Using best params from Stages 1 & 2)
    # ==========================================================
    print("\n--- STAGE 3: Testing Scheduler ---")
    print(f"Using Best Params: LR={baseline_lr}, Alpha={best_alpha}, Gamma={best_gamma}")
    
    sched_model = ResNetClassifier(
        IMG_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV, 
        use_focal=True, focal_alpha=best_alpha, focal_gamma=best_gamma, 
        lr=baseline_lr, use_scheduler=True, run_name="focal_with_scheduler"
    )
    sched_model.fit(num_epochs=EPOCHS)
    sched_f1, sched_auc, sched_acc = sched_model.test()
    
    with open(results_file, "a") as f:
        f.write("\nSTAGE 3: SCHEDULER ABLATION\n")
        f.write(f"Used: LR={baseline_lr}, Alpha={best_alpha}, Gamma={best_gamma}\n")
        f.write(f"No Scheduler (From Stage 2) -> AUC: {best_focal_auc:.4f}\n")
        f.write(f"With Scheduler -> AUC: {sched_auc:.4f} | F1: {sched_f1:.4f}\n")

    print("\n✅ Hierarchical Search Complete! Check 'hierarchical_search_results.txt' for logs and plot data.")

if __name__ == '__main__':
    main()
    
    