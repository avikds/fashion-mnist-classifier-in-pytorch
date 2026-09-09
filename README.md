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
- [x] **10.** test_accuracy
- [x] **11.** save_model
- [x] **12.** predict_classes

## Results

```
Fashion-MNIST slices: train/val/test = (8000, 2000, 2000), 125 training batches per epoch
LR range test (mean loss over 20 batches): 0.001:2.32  0.01:2.28  0.05:2.06  0.1:1.73  0.5:1.71  2.0:nan

MLP 784-300-100-10 with 266,610 parameters
  epoch 1: train loss 1.086  val loss 0.704  val acc 0.737
  epoch 2: train loss 0.600  val loss 0.587  val acc 0.776
  epoch 3: train loss 0.519  val loss 0.503  val acc 0.814
  epoch 4: train loss 0.466  val loss 0.500  val acc 0.815
  epoch 5: train loss 0.432  val loss 0.462  val acc 0.834
restored weights from epoch 5
  trial hidden=100/100 lr=0.05: val acc 0.793
  trial hidden=200/50 lr=0.1: val acc 0.817
  trial hidden=100/100 lr=0.01: val acc 0.697
best config: {'hidden1': 200, 'hidden2': 50, 'lr': 0.1, 'val_acc': 0.8165}

TEST accuracy 0.842 (best validation was 0.834)
  predicted Sneaker      truth Ankle boot
  predicted Pullover     truth Pullover
  predicted Trouser      truth Trouser
  predicted Trouser      truth Trouser
  predicted Shirt        truth Shirt
  predicted Trouser      truth Trouser
```
