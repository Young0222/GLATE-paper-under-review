# GLATE

## This is our PyTorch implementation code for our paper: Incorporating Dynamic Temperature Estimation into Contrastive Learning on Graphs


## Environment Requirements

The code has been tested under Python 3.7.13. The required packages are as follows:

* Pytorch == 1.12.1
* Pytorch Geometric == 2.3.0


## Example : Cora dataset

```python
python train_glate.py --dataset Cora
```

The running result is:
...
(T) | Epoch=195, loss=7.7958, this epoch 0.0120, total 3.1680
(T) | Epoch=196, loss=7.7987, this epoch 0.0112, total 3.1792
(T) | Epoch=197, loss=7.8143, this epoch 0.0108, total 3.1900
(T) | Epoch=198, loss=7.7945, this epoch 0.0108, total 3.2008
(T) | Epoch=199, loss=7.7998, this epoch 0.0108, total 3.2116
(T) | Epoch=200, loss=7.8072, this epoch 0.0121, total 3.2237
(E) | evaluate: micro_f1=0.8511+-0.0025, macro_f1=0.0000+-0.0000, ACC=0.8511+-0.0025
