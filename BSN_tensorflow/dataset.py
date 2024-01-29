import numpy as np
import scipy
from sklearn.model_selection import StratifiedKFold


class MIL_Dataset():
    def __init__(self, seed, path, folds):
        self.seed = seed
        self.path = path
        self.folds = folds
        self.datasets = self.load_dataset()
        # self.num_bags = 
        # datasets.datasets[fold]['test']

    def get_next_batch(self, step, fold, is_train):
        if is_train:
            step = step%len(self.datasets[fold]['train'])
            batch_fea, batch_label = self.datasets[fold]['train'][step]
        else:
            step = step%len(self.datasets[fold]['test'])
            batch_fea, batch_label = self.datasets[fold]['test'][step]
        return batch_fea, batch_label

    def create_bags_mat(self, path):
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

    def load_dataset(self):
        bags, labels=self.create_bags_mat(path=self.path)
        skf = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.seed)
        # model, optimizer = get_model()
        # dataset = [[],[]],[[],[]]
        dataset = [{}]*self.folds
        for idx, (tr, ts) in enumerate(skf.split(bags, labels)):
            dataset[idx]['train'] = [(bags[tr][i], labels[tr][i]) for i in range(len(bags[tr]))]
            dataset[idx]['test'] = [(bags[ts][i], labels[ts][i]) for i in range(len(bags[ts]))]
            # dataset[0][0].append(bags[tr])
            # dataset[0][1].append(labels[tr])
            # dataset[1][0].append(bags_ts=bags[ts])
            # dataset[1][1].append(labels[ts])
        return dataset

            # loader_training = DataLoader(MyDataset(bags_tr, y_tr), batch_size=1)
            # loader_test = DataLoader(MyDataset(bags_ts, y_ts), batch_size=1)