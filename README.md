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
- python main_mask1_messidor.py --model=attnet --dataset=musk1
- python main_mask1_messidor.py --model=attnet --dataset=messidor

### mi-net (https://arxiv.org/abs/1610.02501):
- python main.py --model=minet --dataset=elephant
- python main.py --model=minet --dataset=fox
- python main.py --model=minet --dataset=tiger
- python main_mask1_messidor.py --model=minet --dataset=musk1
- python main_mask1_messidor.py --model=minet --dataset=messidor

### MI-net (https://arxiv.org/abs/1610.02501):
- python main.py --model=MInet --dataset=elephant
- python main.py --model=MInet --dataset=fox
- python main.py --model=MInet --dataset=tiger

### BSN in pytorch (https://www.sciencedirect.com/science/article/abs/pii/S0020025519306826):
- python BSN/main_bsn.py --dataset=elephant
- python BSN/main_bsn.py --dataset=fox
- python BSN/main_bsn.py --dataset=tiger

#### Default values: 
- Elephant, fox and tiger: epochs=10, folds=10, run=5, no_cuda=True
- mask1, messidor: epochs=50, folds=10, run=5, no_cuda=True
