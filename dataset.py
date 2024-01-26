import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io
from sklearn.model_selection import KFold

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
        labels.append(0 if l[ids==i][0] == -1 else 1)
    bags=np.array(bags, dtype=object)
    labels=np.array(labels)
    return bags, labels

def load_dataset(dataset_nm, n_folds):
    """Load data from file, do pre-processing, split it into train/test set.
    Parameters
    -----------------
    dataset_nm : string
        Name of dataset.
    n_folds : int
        Number of cross-validation folds.
    Returns
    -----------------
    datasets : list
        List contains split datasets for k-Fold cross-validation.
    """
    # load data from file
    data = scipy.io.loadmat('./data/'+dataset_nm+'.mat')
    ins_fea = data['x']['data'][0,0]
    if dataset_nm.startswith('musk'):
        bags_nm = data['x']['ident'][0,0]['milbag'][0,0]
    else:
        bags_nm = data['x']['ident'][0,0]['milbag'][0,0][:,0]
    bags_label = data['x']['nlab'][0,0][:,0] - 1

    # L2 norm for musk1 and musk2
    if dataset_nm.startswith('newsgroups') is False:
        mean_fea = np.mean(ins_fea, axis=0, keepdims=True)+1e-6
        std_fea = np.std(ins_fea, axis=0, keepdims=True)+1e-6
        ins_fea = np.divide(ins_fea-mean_fea, std_fea)

    # store data in bag level
    ins_idx_of_input = {}            # store instance index of input
    for id, bag_nm in enumerate(bags_nm):
        if bag_nm in ins_idx_of_input: ins_idx_of_input[bag_nm].append(id)
        else:                                ins_idx_of_input[bag_nm] = [id]
    bags_fea = []
    for bag_nm, ins_idxs in ins_idx_of_input.items():
        bag_fea = ([], [])
        for ins_idx in ins_idxs:
            bag_fea[0].append(ins_fea[ins_idx])
            bag_fea[1].append(bags_label[ins_idx])
        bags_fea.append(bag_fea)

    # random select 90% bags as train, others as test
    num_bag = len(bags_fea)
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=None)
    datasets = []
    for train_idx, test_idx in kf.split(bags_fea):
        dataset = {}
        dataset['train'] = [bags_fea[ibag] for ibag in train_idx]
        dataset['test'] = [bags_fea[ibag] for ibag in test_idx]
        datasets.append(dataset)
    return datasets

def convertToBatch(bags):
    """Convert to batch format.
    Parameters
    -----------------
    bags : list
        A list contains instance features of bags and bag labels.
    Return
    -----------------
    data_set : list
        Convert dataset to batch format(instance features, bag label).
    """
    batch_num = len(bags)
    data_set = []
    for ibag, bag in enumerate(bags):
        batch_data = torch.tensor(np.array(bag[0]), dtype=torch.float32)
        batch_label = torch.tensor(np.array(bag[1]), dtype=torch.float32)
        data_set.append((batch_data, batch_label))
    return data_set

def split_data_label(ds):
    train_data = []
    train_label = []
    for d in ds:
        train_data.append(d[0])
        train_label.append(d[1][0])
    return train_data, train_label
