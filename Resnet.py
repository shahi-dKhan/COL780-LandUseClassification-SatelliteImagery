import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from load_data import CropData
import multiprocessing as mp


best_auc = -1
best_model_path = "best_resnet_model.pth"
def get_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")


def get_dataloaders(img_dir, train_csv_path, validation_csv_path, test_csv_path, batch_size, device):
    train_dataset = CropData(img_dir_path=img_dir, csv_file_path=train_csv_path)
    validation_dataset = CropData(img_dir_path=img_dir, csv_file_path=validation_csv_path)
    test_dataset = CropData(img_dir_path=img_dir, csv_file_path=test_csv_path)

    # On macOS/MPS, worker spawning can be fragile during script iteration.
    num_workers = 0 if device.type == "mps" else 4
    pin_memory = device.type == "cuda"

    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    validation_data = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_data, validation_data, test_data


def build_model(num_classes, device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    return model.to(device)

# ------------------ Evaluation ------------------
def evaluate_metrics(model, data, device):
    model.eval()

    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for images, labels in data:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_predictions, average="macro")

    try:
        macro_auc = roc_auc_score(
            all_labels,
            all_probs,
            multi_class="ovr",
            average="macro"
        )
    except:
        macro_auc = None

    acc = accuracy_score(all_labels, all_predictions)

    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Macro ROC AUC Score: {macro_auc}")
    print(f"Accuracy: {acc:.4f}")

    return macro_f1, macro_auc, acc


def main():
    # ------------------ Paths ------------------
    img_dir = "./A3_Dataset"
    train_csv_path = "./A3_Dataset/train.csv"
    validation_csv_path = "./A3_Dataset/validation.csv"
    test_csv_path = "./A3_Dataset/test.csv"

    batch_size = 64
    num_epochs = 10
    num_classes = 10

    device = get_device()
    train_data, validation_data, test_data = get_dataloaders(
        img_dir,
        train_csv_path,
        validation_csv_path,
        test_csv_path,
        batch_size,
        device,
    )

    model = build_model(num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_data:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_data)
        print(f"\nEpoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        print("Validation Metrics:")
        _, val_auc, _ = evaluate_metrics(model, validation_data, device)

        # Save the best model based on validation AUC
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)

    print("\nFinal Test Metrics:")
    evaluate_metrics(model, test_data, device)
    
    print("\nLoading best model...")
    model.load_state_dict(torch.load(best_model_path))

    print("\nFinal Test Metrics (Best Model):")
    evaluate_metrics(model, test_data)


if __name__ == "__main__":
    mp.freeze_support()
    main()