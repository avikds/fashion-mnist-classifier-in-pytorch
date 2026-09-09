"""
Fashion-MNIST Classifier in PyTorch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Fashion-MNIST classifier in PyTorch (Hands-On ML, chapter 10).

Story: the real Fashion-MNIST files, a Dataset with normalization and seeded
DataLoaders, an MLP nn.Module, one-epoch and evaluation loops, a fit function
that restores the best validation epoch, a learning-rate range test, a small
random search, one test-set score, and a saved model serving raw images.
"""
import os
import tempfile
import numpy as np
import torch
import torch.nn as nn


def main() -> None:
    data = load_fashion_mnist(n_train=10000, n_test=2000)
    loaders = make_loaders(data, batch_size=64, val_size=2000)
    print(f"Fashion-MNIST slices: train/val/test = {loaders['sizes']}, {len(loaders['train'])} training batches per epoch")

    # ---- 1. Where should the learning rate be? ----
    curve = lr_range_test(MLP, loaders["train"], [0.001, 0.01, 0.05, 0.1, 0.5, 2.0], n_batches=20)
    print("LR range test (mean loss over 20 batches): " + "  ".join(f"{lr}:{v:.2f}" for lr, v in curve.items()))

    # ---- 2. Train with validation and best-epoch restore ----
    torch.manual_seed(0)
    model = MLP()
    print(f"\nMLP 784-300-100-10 with {count_parameters(model):,} parameters")
    hist = fit(model, loaders, epochs=5, lr=0.05)
    for e, (tl, vl, va) in enumerate(zip(hist["train_loss"], hist["val_loss"], hist["val_acc"])):
        print(f"  epoch {e + 1}: train loss {tl:.3f}  val loss {vl:.3f}  val acc {va:.3f}")
    print(f"restored weights from epoch {hist['best_epoch'] + 1}")

    # ---- 3. A small random search ----
    search = random_search(loaders, n_trials=3, epochs=2)
    for t in search["trials"]:
        print(f"  trial hidden={t['hidden1']}/{t['hidden2']} lr={t['lr']}: val acc {t['val_acc']:.3f}")
    print(f"best config: {search['best']}")

    # ---- 4. Test once, ship ----
    print(f"\nTEST accuracy {test_accuracy(model, loaders):.3f} (best validation was {max(hist['val_acc']):.3f})")
    path = os.path.join(tempfile.gettempdir(), "fashion_mlp.pt")
    save_model(model, path)
    served = load_model(path)
    raw = (data["X_test"][:6] * 255).round().to(torch.uint8).numpy()
    preds = predict_classes(served, raw)
    truth = [CLASS_NAMES[i] for i in data["y_test"][:6].tolist()]
    for p, t in zip(preds, truth):
        print(f"  predicted {p:<12} truth {t}")


if __name__ == "__main__":
    main()

