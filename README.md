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

## Example : CiteSeer dataset

```python
python train_glate.py --dataset CiteSeer
```

## Example : PubMed dataset

```python
python train_glate.py --dataset PubMed
```

## Example : PPI dataset

```python
python GLATE-PPI.py
```
