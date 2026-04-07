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
    
    
class DynT(nn.Module):
    def __init__(self, normalized_shape):
        super(DynT, self).__init__()
        self.normalized_shape = normalized_shape
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.alpha = nn.Parameter(torch.ones(normalized_shape))
        
    def forward(self, x):
        return torch.tanh(self.alpha * x)

def replacelayernorm_with_dynt(module):
    for name, child in module.named_children():
        # what are these module? they are the layers of the model, and we want to replace the LayerNorm layers with out custom layers.
        # So, module is the model, and child is the layer of the model. We want to check if the layer is a LayerNorm layer, and if it is, we want to replace it with our custom layer.
        # And then, the child layer has another child layer? Yes, it can have another child layer, and we want to check that as well. So we need to do this recursively.
        # So module is basically the top layer? Is module even a layer, or just a pointer? If a module has multiple child layers, and each child layer can have multiple child layer.. what is the architecture of Deit? 
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, DynT(child.normalized_shape))
        else:
            replacelayernorm_with_dynt(child) 
        
class DieTClassifier(nn.Module):
    def __init__(self, img_dir, train_csv, val_csv, test_csv, batch_size=32, num_classes=10, 
                 lr=1e-4, use_focal=False, focal_alpha=1.0, focal_gamma=2.0, run_name="deit3", use_dyt=True):
        super().__init__()
        self.run_name = run_name
        self.device = self.get_device()
        self.train_loader, self.val_loader, self.test_loader = self.get_dataloaders(img_dir, train_csv, val_csv, test_csv, batch_size)
        self.model = self.build_model(num_classes, use_dyt).to(self.device)
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
    
    def build_model(self, num_classes=10, use_dyt=True):
        print("Building DeiT-3 Small...")
        # timm.create_model handles prepending the [CLS] token and passes its 
        # representation to the final replaced linear classification head automatically 
        # when we specify num_classes.
        model = timm.create_model('deit3_small_patch16_224', pretrained=True, num_classes=num_classes)
        if use_dyt:
            replacelayernorm_with_dynt(model)
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
    
    