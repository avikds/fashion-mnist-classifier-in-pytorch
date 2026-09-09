"""
Fashion-MNIST Classifier in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_fashion_mnist
import os
import tempfile
import gzip
import urllib.request
import numpy as np
import torch

def load_fashion_mnist(n_train=10000, n_test=2000):
    base_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    # Download each file once into the system temporary directory.
    file_paths = {}
    temp_dir = tempfile.gettempdir()

    for filename in files:
        path = os.path.join(temp_dir, filename)
        file_paths[filename] = path

        if not os.path.exists(path):
            urllib.request.urlretrieve(base_url + filename, path)

    # Parse image IDX files.
    def load_images(path):
        with gzip.open(path, "rb") as f:
            data = f.read()

        images = np.frombuffer(
            data,
            dtype=np.uint8,
            offset=16
        ).reshape(-1, 28, 28)

        return images

    # Parse label IDX files.
    def load_labels(path):
        with gzip.open(path, "rb") as f:
            data = f.read()

        labels = np.frombuffer(
            data,
            dtype=np.uint8,
            offset=8
        )

        return labels

    train_images = load_images(file_paths["train-images-idx3-ubyte.gz"])
    train_labels = load_labels(file_paths["train-labels-idx1-ubyte.gz"])

    test_images = load_images(file_paths["t10k-images-idx3-ubyte.gz"])
    test_labels = load_labels(file_paths["t10k-labels-idx1-ubyte.gz"])

    # Select the requested number of samples.
    train_images = train_images[:n_train]
    train_labels = train_labels[:n_train]

    test_images = test_images[:n_test]
    test_labels = test_labels[:n_test]

    # Convert images to float32 tensors and scale pixels to [0, 1].
    X_train = torch.from_numpy(train_images.copy()).to(torch.float32) / 255.0
    X_test = torch.from_numpy(test_images.copy()).to(torch.float32) / 255.0

    # Convert labels to int64 tensors.
    y_train = torch.from_numpy(train_labels.copy()).to(torch.int64)
    y_test = torch.from_numpy(test_labels.copy()).to(torch.int64)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }

# Step 2 - FashionDataset
class FashionDataset(Dataset):
    def __init__(self, X, y, mean=0.2860, std=0.3530):
        self.X = X
        self.y = y
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        image = (self.X[i] - self.mean) / self.std
        label = self.y[i].to(torch.int64)

        return image, label

# Step 3 - make_loaders
def make_loaders(data, batch_size=64, val_size=2000, seed=42):
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    # The last val_size training samples form the validation set.
    X_tr = X_train[:-val_size]
    y_tr = y_train[:-val_size]

    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]

    train_dataset = FashionDataset(X_tr, y_tr)
    val_dataset = FashionDataset(X_val, y_val)
    test_dataset = FashionDataset(X_test, y_test)

    # Seeded generator for reproducible training shuffling.
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "sizes": (
            len(train_dataset),
            len(val_dataset),
            len(test_dataset)
        ),
    }

# Step 4 - MLP
class MLP(nn.Module):
    def __init__(self, hidden1=300, hidden2=100, n_classes=10):
        super().__init__()

        self.fc1 = nn.Linear(784, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, n_classes)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.out(x)

        return x

def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

# Step 5 - train_one_epoch
def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()

    total_loss = 0.0
    num_batches = 0

    for X_batch, y_batch in loader:
        optimizer.zero_grad()

        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return float(total_loss / num_batches)

# Step 6 - evaluate
def evaluate(model, loader, loss_fn):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)

            loss = loss_fn(logits, y_batch)

            batch_size = y_batch.size(0)

            # Convert the batch mean loss to summed loss.
            total_loss += loss.item() * batch_size

            predictions = logits.argmax(dim=1)
            total_correct += (predictions == y_batch).sum().item()
            total_examples += batch_size

    mean_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return float(mean_loss), float(accuracy)

# Step 7 - fit (not yet solved)
# TODO: implement

# Step 8 - lr_range_test (not yet solved)
# TODO: implement

# Step 9 - random_search (not yet solved)
# TODO: implement

# Step 10 - test_accuracy (not yet solved)
# TODO: implement

# Step 11 - save_model (not yet solved)
# TODO: implement

# Step 12 - predict_classes (not yet solved)
# TODO: implement

