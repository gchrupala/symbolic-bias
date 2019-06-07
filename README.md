# symbolic-bias
Code for the paper: Symbolic inductive bias for visually grounded learning of spoken language. https://arxiv.org/abs/1812.09244

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

Download trained models and results and unpack them:

```
wget http://grzegorz.chrupala.me/data/symbolic-bias/experiments.tgz
tar zxvf experiments.tgz
```

## Usage
