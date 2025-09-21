import argparse, os, numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from src.models import SimpleCNN

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", default=["100","101"], help="ECG records (placeholder)")
    ap.add_argument("--win", type=int, default=10, help="window length (s)")
    ap.add_argument("--step", type=int, default=5, help="window step (s)")
    ap.add_argument("--epochs", type=int, default=5, help="training epochs")
    ap.add_argument("--out", default="out/cnn.pth", help="output model path")
    args = ap.parse_args()

    # --- Fake ECG windows for demo (replace with real preprocessing) ---
    np.random.seed(0)
    X = np.random.randn(200, 3000).astype(np.float32)  # 200 windows of length 3000
    y = np.random.randint(0, 2, size=200)

    maxlen = X.shape[1]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    Xtr, ytr = torch.tensor(Xtr).unsqueeze(1), torch.tensor(ytr, dtype=torch.long)
    Xte, yte = torch.tensor(Xte).unsqueeze(1), torch.tensor(yte, dtype=torch.long)

    # --- Model ---
    model = SimpleCNN(input_len=maxlen, n_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --- Training loop ---
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(Xtr)
        loss = criterion(outputs, ytr)
        loss.backward()
        optimizer.step()
        acc = (outputs.argmax(1) == ytr).float().mean().item()
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss.item():.4f} - Train Acc: {acc:.3f}")

    # Save model
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # ✅ Save only the weights (simpler for loading in app)
    torch.save({
    "input_len": maxlen,   # ✅ use actual input length
    "state_dict": model.state_dict()
}, args.out)

    print("✅ Model saved to out/cnn.pth")


if __name__ == "__main__":
    main()
