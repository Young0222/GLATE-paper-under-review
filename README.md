# GLATE

## This is the PyTorch implementation code for our paper: Incorporating Dynamic Temperature Estimation into Contrastive Learning on Graphs


## Environment Requirements

The code has been tested under Python 3.7.13. The required packages are as follows:

* Pytorch == 1.12.1
* Pytorch Geometric == 2.3.0


## Example : Cora dataset

```python
python train_glate.py --dataset Cora
```

The running result is:

(T) | Epoch=001, loss=8.5669, this epoch 0.9237, total 0.9237

(T) | Epoch=002, loss=8.5673, this epoch 0.0124, total 0.9361

(T) | Epoch=003, loss=8.5540, this epoch 0.0125, total 0.9485

(T) | Epoch=004, loss=8.5513, this epoch 0.0112, total 0.9598

(T) | Epoch=005, loss=8.5377, this epoch 0.0113, total 0.9711

...

Total training time: 3.2937s

Downstream task accuracy result: 0.8504+-0.0029
