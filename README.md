## CONFIGURATION:
1. conda create -n yourenvname python=3.7
2. conda activate yourenvname
3. conda install pip
4. pip install -r requirements.txt

## RUNNING:
### AttNet (https://arxiv.org/abs/1802.04712):
- python main.py --model=attnet --dataset=elephant
- python main.py --model=attnet --dataset=fox
- python main.py --model=attnet --dataset=tiger
- python main_mask1.py --model=attnet

### mi-net (https://arxiv.org/abs/1610.02501):
- python main.py --model=minet --dataset=elephant
- python main.py --model=minet --dataset=fox
- python main.py --model=minet --dataset=tiger
- python main_mask1.py --model=minet

#### Default values: 
- Elephant, fox and tiger: epochs=10, folds=10, run=5, no_cuda=True
- mask1: epochs=100, folds=10, run=5, no_cuda=True
