import os
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import json
from torchvision import transforms

# ------------------------------
# 1. 设置随机种子和可复现性
# ------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ------------------------------
# 2. 数据集定义
# ------------------------------
class PneumoniaDatasetGraph(Dataset):
    def __init__(self, base_dir, labels, img_size=224, patch_size=15, augment=False):
        self.labels = labels
        self.img_size = img_size
        self.patch_size = patch_size
        self.files = []
        self.augment = augment

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        for label in labels:
            path = os.path.join(base_dir, label)
            class_num = labels.index(label)
            for img in os.listdir(path):
                if img.lower().endswith((".jpeg", ".png", ".jpg")):
                    self.files.append((os.path.join(path, img), class_num))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, label = self.files[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (self.img_size, self.img_size)) / 255.0

        if self.augment:
            img_resized = self.transform(torch.tensor(img_resized).unsqueeze(0)).squeeze(0).numpy()
        else:
            img_resized = (img_resized - 0.5) / 0.5

        patches = np.array(self._extract_patches(img_resized), dtype=np.float32)  # 转为单个 numpy 数组
        x = torch.tensor(patches).unsqueeze(1)

        features = x.view(x.size(0), -1).mean(dim=1).unsqueeze(1)
        edge_index = self._create_grid_edge_index()

        data = Data(x=features, edge_index=edge_index, y=torch.tensor(label, dtype=torch.long))
        return data

    def _extract_patches(self, img):
        patches = []
        for i in range(0, self.img_size, self.patch_size):
            for j in range(0, self.img_size, self.patch_size):
                patch = img[i:i + self.patch_size, j:j + self.patch_size]
                if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
                    patch = np.pad(patch, ((0, self.patch_size - patch.shape[0]), (0, self.patch_size - patch.shape[1])), 'constant')
                patches.append(patch)
        return patches

    def _create_grid_edge_index(self):
        num_patches = (self.img_size // self.patch_size) ** 2
        edge_index = []
        grid_size = self.img_size // self.patch_size
        for idx in range(num_patches):
            row = idx // grid_size
            col = idx % grid_size
            neighbors = []
            if row > 0:
                neighbors.append(idx - grid_size)
            if row < grid_size - 1:
                neighbors.append(idx + grid_size)
            if col > 0:
                neighbors.append(idx - 1)
            if col < grid_size - 1:
                neighbors.append(idx + 1)
            for neighbor in neighbors:
                edge_index.append([idx, neighbor])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index

# ------------------------------
# 3. 数据集分割
# ------------------------------
def split_dataset(dataset, train_split=0.7, val_split=0.15, test_split=0.15, shuffle=True, seed=42):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    if shuffle:
        np.random.seed(seed)
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

# ------------------------------
# 4. 定义模型
# ------------------------------
class GCNModel(nn.Module):
    def __init__(self, num_node_features=1, hidden_channels=16, num_classes=2):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = global_mean_pool(x, data.batch)
        x = self.fc(x)
        return x

# ------------------------------
# 5. 训练与评估
# ------------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=20):
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data.y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        scheduler.step(val_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    return train_losses, val_losses, val_accuracies

def evaluate_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            outputs = model(data)
            loss = criterion(outputs, data.y)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == data.y).sum().item()
            total += data.y.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def test_model(model, loader, criterion, device):
    test_loss, test_accuracy = evaluate_model(model, loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    return test_loss, test_accuracy

# ------------------------------
# 6. 可视化与 AUC 计算
# ------------------------------
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

def plot_confusion_matrix(y_true, y_pred, labels, model_name):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ------------------------------
# 7. 主程序
# ------------------------------
if __name__ == "__main__":
    base_dir = r'C:\Users\11033\ML\date\raw\train'  # 更新为实际数据路径
    labels = ['NORMAL', 'PNEUMONIA']

    dataset = PneumoniaDatasetGraph(base_dir, labels)
    train_set, val_set, test_set = split_dataset(dataset)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCNModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)

    train_losses, val_losses, val_accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=20)

    test_loss, test_accuracy = test_model(model, test_loader, criterion, device)

    # 保存结果
    y_true = []
    y_pred = []
    y_scores = []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_scores.extend(outputs[:, 1].cpu().numpy())

    auc_score = compute_auc_roc(y_true, y_scores, "GCN")
    plot_confusion_matrix(y_true, y_pred, labels, "GCN")

    results = {
        "model": "GCN",
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_accuracy": val_accuracies,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "auc": auc_score
    }

    with open("gcn_results.json", "w") as f:
        json.dump(results, f)
    print("Results saved to gcn_results.json")
