import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from model import UltrasoundModel
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────────────
BENIGN_PATH    = "./Dataset/benign"
MALIGNANT_PATH = "./Dataset/malignant"
NORMAL_PATH    = "./Dataset/normal"
TEST_PATH      = "./Dataset/test"
BATCH_SIZE     = 16
EPOCHS         = 10
LR             = 0.001
MODEL_SAVE     = "best_model.pth"
MIN_SIZE       = 224

# ── TRANSFORMS ────────────────────────────────────────────────────────────────
def make_transform(augment=False):
    ops = [
        transforms.Resize(MIN_SIZE, antialias=True),
        transforms.CenterCrop(MIN_SIZE),
        transforms.Grayscale(num_output_channels=3),
    ]
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(contrast=0.3, brightness=0.2),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]
    return transforms.Compose(ops)

# ── LOAD EACH CLASS FOLDER SEPARATELY ────────────────────────────────────────
# This avoids the "test folder treated as class" problem
print("📂 Loading dataset...")

from torchvision.datasets import ImageFolder
from torch.utils.data import Subset

# Load each folder as its own dataset with correct label
class LabeledFolder(torch.utils.data.Dataset):
    def __init__(self, folder_path, label, transform):
        self.transform = transform
        self.label     = label
        self.files     = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img), self.label

transform_train = make_transform(augment=True)
transform_val   = make_transform(augment=False)

benign_data    = LabeledFolder(BENIGN_PATH,    0, transform_train)
malignant_data = LabeledFolder(MALIGNANT_PATH, 1, transform_train)
normal_data    = LabeledFolder(NORMAL_PATH,    2, transform_train)

full_dataset = ConcatDataset([benign_data, malignant_data, normal_data])
total        = len(full_dataset)

print(f"Benign   : {len(benign_data)} images")
print(f"Malignant: {len(malignant_data)} images")
print(f"Normal   : {len(normal_data)} images")
print(f"Total    : {total} images")

# Split 80/20
train_size = int(0.8 * total)
val_size   = total - train_size
train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])
print(f"Train: {train_size} | Val: {val_size}")

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── MODEL ─────────────────────────────────────────────────────────────────────
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️  Device: {device}")

model     = UltrasoundModel(num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# ── TRAINING ──────────────────────────────────────────────────────────────────
best_val_acc    = 0.0
train_loss_hist = []
val_acc_hist    = []

print("\n🚀 Training started...\n")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_loss_hist.append(avg_loss)

    model.eval()
    correct, total_v = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            _, preds = torch.max(model(imgs), 1)
            correct  += (preds == labels).sum().item()
            total_v  += labels.size(0)

    val_acc = correct / total_v
    val_acc_hist.append(val_acc)
    scheduler.step()

    status = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE)
        status = " ✅ Saved!"

    print(f"Epoch [{epoch+1:2d}/{EPOCHS}] | Loss: {avg_loss:.4f} | Val Acc: {val_acc*100:.2f}%{status}")

print(f"\n🏆 Best Val Accuracy: {best_val_acc*100:.2f}%")
print(f"📦 Model saved → {MODEL_SAVE}")

# ── TEST SET ──────────────────────────────────────────────────────────────────
print("\n📊 Evaluating test set...")
try:
    test_data   = LabeledFolder(TEST_PATH, 0, make_transform(augment=False))
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model.load_state_dict(torch.load(MODEL_SAVE, map_location=device))
    model.eval()
    correct, total_t = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            _, preds = torch.max(model(imgs), 1)
            correct  += (preds == labels).sum().item()
            total_t  += labels.size(0)
    print(f"✅ Test Accuracy: {correct/total_t*100:.2f}%")
except Exception as e:
    print(f"⚠️  Test eval skipped: {e}")

# ── PLOT ──────────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss_hist, marker='o', color='#1FA4A9')
plt.title("Training Loss"); plt.xlabel("Epoch")

plt.subplot(1, 2, 2)
plt.plot([a*100 for a in val_acc_hist], marker='o', color='#EC4899')
plt.title("Validation Accuracy (%)"); plt.xlabel("Epoch")

plt.tight_layout()
plt.savefig("training_results.png")
print("📈 Graph → training_results.png")