import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from dataset import DrumSampleDataset, CLASS_NAMES
from model import DrumCNN

def train_model(data_root, out_path, epochs=20, batch_size=32, lr=1e-3, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    train_ds = DrumSampleDataset(os.path.join(data_root, "labeled"), split="train")
    val_ds   = DrumSampleDataset(os.path.join(data_root, "labeled"), split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = DrumCNN(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for mel, labels in train_loader:
            mel = mel.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(mel)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * mel.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total if total > 0 else 0.0
        train_acc = correct / total if total > 0 else 0.0

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for mel, labels in val_loader:
                mel = mel.to(device)
                labels = labels.to(device)
                outputs = model(mel)
                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0

        print(f"Epoch {epoch+1}/{epochs} | train loss {train_loss:.4f} | train acc {train_acc:.3f} | val acc {val_acc:.3f}")

    torch.save(model.state_dict(), out_path)
    print("Saved model to", out_path)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(base_dir, "data")
    out_path = os.path.join(base_dir, "drum_cnn.pth")
    train_model(data_root=data_root, out_path=out_path)

