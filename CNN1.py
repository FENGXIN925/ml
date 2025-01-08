import os
import sys
import cv2
import torch
import json
import numpy as np
import random
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

# 设置随机种子和确定性行为
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 自定义数据集类
class PneumoniaDataset(Dataset):
    def __init__(self, base_dir, labels, img_size=224, augment=False):
        self.labels = labels
        self.img_size = img_size
        self.files = []  # 存储 (文件路径, 标签)
        self.augment = augment

        # 数据增强
        self.transform_augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.transform_normal = transforms.Compose([
            transforms.ToPILImage(),
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
        return len(self.files)  # 返回文件列表的长度

    def __getitem__(self, idx):
        img_path, label = self.files[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (self.img_size, self.img_size))

        if self.augment:
            img_resized = self.transform_augment(img_resized)  # 数据增强
        else:
            img_resized = self.transform_normal(img_resized)

        # 扩展为三通道
        img_resized = img_resized.repeat(3, 1, 1)

        return img_resized, label

# 数据集分割函数
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

# 定义CNN模型
class OptimizedCNN(nn.Module):
    def __init__(self):
        super(OptimizedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # 输入通道改为3
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.3)
        self.fc1_in_features = None
        self.fc1 = None
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        if self.fc1 is None:
            self.fc1_in_features = x.size(1)
            self.fc1 = nn.Linear(self.fc1_in_features, 128).to(x.device)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 模型初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = OptimizedCNN().to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)

# 早停类
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

early_stopping = EarlyStopping(patience=5)

# 评估函数
def evaluate_model(model, data_loader, criterion):
    model.eval()  # 切换到评估模式
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # 使用传入的 criterion 计算损失

            # 累计损失
            running_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # 保存概率和真实标签
            all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算平均损失和准确率
    avg_loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy, np.array(all_labels), np.array(all_probs)

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20):
    # 初始化记录列表
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(epochs):
        # 切换到训练模式
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 训练一个epoch
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 累积损失
            running_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # 计算训练损失和准确率
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # 验证模型
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # 更新学习率调度器
        scheduler.step(val_loss)

        # 打印每个 epoch 的结果
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

    # 返回记录的损失和准确率
    return train_losses, train_accuracies, val_losses, val_accuracies

# 测试函数
def test_model(model, test_loader, criterion):
    test_loss, test_acc, y_true, y_scores = evaluate_model(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    # 计算 AUC-ROC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    print(f"Test AUC-ROC: {roc_auc:.4f}")

    # 绘制 AUC-ROC 曲线
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

    return test_loss, test_acc, roc_auc, y_true, y_scores

# 混淆矩阵绘制
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# 加载数据集
base_dir = r'C:\Users\11033\ML\date\raw\train'  # 更新为您的数据集路径
labels = ['NORMAL', 'PNEUMONIA']
dataset = PneumoniaDataset(base_dir, labels, img_size=224)

# 分割数据集
train_set, val_set, test_set = split_dataset(dataset)

# 创建DataLoader，确保可复现性
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0,
                          worker_init_fn=lambda worker: set_seed(42))
val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)

# 主程序
if __name__ == "__main__":
    # 创建训练集增强版本
    train_set_augmented = PneumoniaDataset(base_dir, labels, img_size=224, augment=True)
    train_loader = DataLoader(train_set_augmented, batch_size=32, shuffle=True, num_workers=0,
                              worker_init_fn=lambda worker: set_seed(42))

    # 训练模型
    train_losses, train_accuracies, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20
    )

    # 测试模型
    test_loss, test_acc, roc_auc, y_true, y_scores = test_model(model, test_loader, criterion)

    # 绘制混淆矩阵
    y_pred = np.argmax(y_scores, axis=1)
    plot_confusion_matrix(y_true, y_pred, labels)

    # 保存结果到JSON文件
    results = {
        "model": "CNN",
        "train_loss": train_losses,
        "train_accuracy": train_accuracies,
        "val_loss": val_losses,
        "val_accuracy": val_accuracies,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "auc_roc": roc_auc
    }
    with open("cnn_results.json", "w") as f:
        json.dump(results, f)
    print("Results saved to cnn_results.json")
