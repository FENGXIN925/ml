import os
import cv2
import torch
import numpy as np
import random
import sys
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation, ToTensor, Normalize, Resize
from torchvision.models import ResNet18_Weights
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms import functional as F
import json
from PIL import Image

# 1. 设置随机种子和确定性行为
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 2. 记录环境信息
def log_environment():
    print("Python version:", sys.version)
    print("PyTorch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("NumPy version:", np.__version__)
    print("Random seed:", 42)

log_environment()

# 3. 自定义数据集类
class PneumoniaDataset(Dataset):
    def __init__(self, base_dir, labels, img_size=224, augment=False):
        self.labels = labels
        self.img_size = img_size
        self.files = []
        self.augment = augment

        self.transform_augment = Compose([
            Resize((img_size, img_size)),
            RandomHorizontalFlip(),
            RandomRotation(10),
            ToTensor(),
            Normalize(mean=[0.5], std=[0.5])
        ])

        self.transform_normal = Compose([
            Resize((img_size, img_size)),
            ToTensor(),
            Normalize(mean=[0.5], std=[0.5])
        ])

        for label in labels:
            path = os.path.join(base_dir, label)
            if os.path.exists(path):
                class_num = labels.index(label)
                for img in os.listdir(path):
                    if img.lower().endswith((".jpeg", ".png", ".jpg")):
                        self.files.append((os.path.join(path, img), class_num))
            else:
                print(f"Warning: The directory {path} does not exist. Skipping this label.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, label = self.files[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = Image.fromarray(img)

        if self.augment:
            img_tensor = self.transform_augment(img)
        else:
            img_tensor = self.transform_normal(img)

        img_tensor = img_tensor.repeat(3, 1, 1)
        return img_tensor, label

# 4. 数据集分割函数
def split_dataset(dataset, train_split=0.7, val_split=0.15, test_split=0.15, shuffle=True):
    assert train_split + val_split + test_split == 1.0, "Splits must sum to 1.0"
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    if shuffle:
        np.random.shuffle(indices)

    train_end = int(train_split * dataset_size)
    val_end = train_end + int(val_split * dataset_size)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    return train_set, val_set, test_set

# 5. 定义迁移学习模型
class TransferLearningModel(nn.Module):
    def __init__(self, num_classes=2):
        super(TransferLearningModel, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

# 6. AUC-ROC 计算
def compute_auc_roc(y_true, y_scores, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'{model_name} AUC (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc='lower right')
    plt.show()
    return roc_auc

# 7. 混淆矩阵
def plot_confusion_matrix(y_true, y_pred, labels, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

# 8. Grad-CAM 可视化
def visualize_grad_cam(model, input_tensor, target_class):
    cam_extractor = SmoothGradCAMpp(model)
    activation_map = cam_extractor(input_tensor.unsqueeze(0), class_idx=target_class)
    heatmap = F.to_pil_image(activation_map.squeeze(0))
    heatmap.show()

# 9. 模型训练和评估函数
def evaluate_model(model, data_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_scores = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_scores.extend(outputs.softmax(dim=1)[:, 1].cpu().numpy())

    avg_loss = running_loss / len(data_loader)
    accuracy = 100 * sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    return avg_loss, accuracy, all_labels, all_preds, all_scores

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        val_loss, val_acc, _, _, _ = evaluate_model(model, val_loader, criterion)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

    return train_losses, train_accuracies, val_losses, val_accuracies

# 10. 主程序
if __name__ == "__main__":
    base_dir = r'C:\Users\11033\ML\date\raw\date'  # 修改路径到数据集根目录
    labels = ['NORMAL', 'PNEUMONIA']
    dataset = PneumoniaDataset(base_dir, labels, img_size=224)

    # 检查数据集是否为空
    if len(dataset) == 0:
        raise ValueError("The dataset is empty. Please check your data path and ensure the directories contain images.")

    train_set, val_set, test_set = split_dataset(dataset)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransferLearningModel(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)

    print("----- Training Transfer Learning Model -----")
    train_losses, train_accuracies, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20
    )

    test_loss, test_acc, y_true, y_pred, y_scores = evaluate_model(model, test_loader, criterion)
    auc_score = compute_auc_roc(y_true, y_scores, "Transfer Learning")
    plot_confusion_matrix(y_true, y_pred, labels, "Transfer Learning")

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%, AUC: {auc_score:.2f}")

    # 保存结果到 JSON 文件
    results = {
        "model": "Transfer Learning",
        "train_loss": train_losses,
        "train_accuracy": train_accuracies,
        "val_loss": val_losses,
        "val_accuracy": val_accuracies,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "AUC": auc_score
    }
    with open("transfer_results.json", "w") as f:
        json.dump(results, f)
    print("Results saved to transfer_results.json")
