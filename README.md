# symbolic-bias
Code for the paper: Symbolic inductive bias for visually grounded learning of spoken language. https://arxiv.org/abs/1812.09244, published at ACL 2019.

## Install

Clone repo and set up and activate a virtual environment with python3

```
cd symbolic-bias
virtualenv -p python3 .
```
Install Python code (in development mode if you will be modifying something).

```
python setup.py develop
```

Download trained models and unpack them:

```
wget http://grzegorz.chrupala.me/data/symbolic-bias/experiments.tgz
tar zxvf experiments.tgz
```
Download data and unpack them:

```
wget http://grzegorz.chrupala.me/data/symbolic-bias/data.tgz
tar zxvf data.tgz
```

## Usage

Execute function ``main`` in file [analysis/analyze.py](analysis/analyze.py). 

```
cd analysis
python -c 'import analyze; analyze.main()'
```

Inspect the definition of this function 
to see how to compute the results from each table in the paper.
