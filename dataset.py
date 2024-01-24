import numpy as np
from torch.utils.data import Dataset
import scipy.io

class MyDataset(Dataset):
    def __init__(self, bags, y):
        self.bags = bags
        self.y = y

    def __getitem__(self, index):
        return self.bags[index], self.y[index]
        
    def __len__(self):
        return len(self.bags)

def create_bags_mat(path):
    mat=scipy.io.loadmat(path)
    ids=mat['bag_ids'][0]
    f=scipy.sparse.csr_matrix.todense(mat['features'])
    l=np.array(scipy.sparse.csr_matrix.todense(mat['labels']))[0]
    bags=[]
    labels=[]
    for i in set(ids):
        bags.append(np.array(f[ids==i]))
        labels.append(l[ids==i][0])
    bags=np.array(bags)
    labels=np.array(labels)
    return bags, labels