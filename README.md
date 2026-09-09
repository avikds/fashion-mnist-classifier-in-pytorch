# Fashion-MNIST Classifier in PyTorch

Chapter 10 of Hands-On Machine Learning as a PyTorch practitioner writes it: download the real Fashion-MNIST files, wrap them in a Dataset and DataLoaders with normalization, build an MLP as an nn.Module, write the training and evaluation loops with a validation pass every epoch and best-checkpoint restore, run a learning-rate range test and a small hyperparameter search, then save the weights, reload them into a fresh model and serve predictions on raw images as class names.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** load_fashion_mnist
- [x] **2.** FashionDataset
- [x] **3.** make_loaders
- [x] **4.** MLP
- [x] **5.** train_one_epoch
- [x] **6.** evaluate
- [x] **7.** fit
- [x] **8.** lr_range_test
- [x] **9.** random_search
- [ ] **10.** test_accuracy
- [ ] **11.** save_model
- [ ] **12.** predict_classes

---

Built on Deep-ML.
