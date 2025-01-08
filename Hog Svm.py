import os
import cv2
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

# ------------------------------
# 1. 设置随机种子和可复现性
# ------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

# ------------------------------
# 2. 数据集加载和预处理
# ------------------------------
def load_data(base_dir, labels, img_size=224, augment=False):
    data = []
    target = []

    # 定义数据增强方法，仅在训练集使用
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    for label in labels:
        path = os.path.join(base_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            if img.lower().endswith(('.jpeg', '.png', '.jpg')):
                img_path = os.path.join(path, img)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                image_resized = cv2.resize(image, (img_size, img_size))

                # 归一化到 [0, 1]
                image_resized = image_resized / 255.0

                if augment and class_num == labels.index('NORMAL'):
                    # 数据增强后转换为灰度图格式
                    augmented_image = transform(image_resized).squeeze(0).numpy()
                    data.append(augmented_image)
                else:
                    data.append(image_resized)

                target.append(class_num)

    return np.array(data), np.array(target)

# ------------------------------
# 3. HOG特征提取
# ------------------------------
def extract_hog_features(images):
    hog = cv2.HOGDescriptor(
        _winSize=(224, 224),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    features = []

    for img in images:
        img_uint8 = (img * 255).astype(np.uint8)
        hog_feature = hog.compute(img_uint8).flatten()
        features.append(hog_feature)

    return np.array(features)

# ------------------------------
# 4. 数据集分割
# ------------------------------
def split_dataset(data, labels, test_size=0.3, seed=42):
    return train_test_split(data, labels, test_size=test_size, random_state=seed, stratify=labels)

# ------------------------------
# 5. SVM分类器训练和评估
# ------------------------------
def train_and_evaluate_svm(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    y_scores = svm.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['NORMAL', 'PNEUMONIA'], output_dict=True)

    fpr, tpr, _ = roc_curve(y_test, y_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    return accuracy, report, roc_auc, y_test, y_pred

# ------------------------------
# 6. 绘制曲线和混淆矩阵
# ------------------------------
def plot_training_curves(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# ------------------------------
# 主函数
# ------------------------------
def main():
    # 数据路径和标签
    base_dir = r'C:\Users\11033\ML\data\raw\train'  # 更新为您的数据集路径
    labels = ['NORMAL', 'PNEUMONIA']

    # 加载训练数据（使用数据增强）
    print("Loading training data with augmentation...")
    train_data, train_target = load_data(base_dir, labels, img_size=224, augment=True)

    # 加载测试数据（不使用数据增强）
    print("Loading test data without augmentation...")
    test_data, test_target = load_data(base_dir, labels, img_size=224, augment=False)

    # 提取HOG特征
    print("Extracting HOG features...")
    train_features = extract_hog_features(train_data)
    test_features = extract_hog_features(test_data)

    # 数据集分割
    print("Splitting dataset...")
    X_train, X_val, y_train, y_val = split_dataset(train_features, train_target, test_size=0.2, seed=42)

    # 训练和评估SVM分类器
    print("Training and evaluating SVM...")
    train_loss, val_loss = [0.3, 0.2], [0.35, 0.25]  # 模拟损失
    accuracy, report, roc_auc, y_test, y_pred = train_and_evaluate_svm(X_train, test_features, y_train, test_target)

    # 绘制训练曲线
    plot_training_curves(train_loss, val_loss)

    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(test_target, y_pred)
    plot_roc_curve(fpr, tpr, roc_auc)

    # 绘制混淆矩阵
    plot_confusion_matrix(test_target, y_pred, labels)

    # 保存结果到JSON文件
    results = {
        "model": "HOG + SVM",
        "test_accuracy": accuracy,
        "classification_report": report,
        "roc_auc": roc_auc
    }
    with open("hog_svm_results.json", "w") as f:
        json.dump(results, f)
    print("Results saved to hog_svm_results.json")

if __name__ == "__main__":
    main()
