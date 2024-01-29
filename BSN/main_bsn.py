import argparse
from torch.autograd import Variable
from sklearn.model_selection import StratifiedKFold
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from MI_net import MINet
from bsn_net import BSN
from dataset import MyDataset, create_bags_mat

# Training settings
parser = argparse.ArgumentParser(description='FLV')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--folds', type=int, default=10, metavar='N',
                    help='number of folds for cross validation (default: 10)')
parser.add_argument('--run', type=int, default=5, metavar='N',
                    help='number of run (default: 5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
# parser.add_argument('--model', type=str, default='minet', help='Choose b/w attnet and minet')
parser.add_argument('--model_save_dir', type=str, default='models', help='Choose path to save models')
parser.add_argument('--dataset', type=str, default='elephant', help='Choose b/w elephant, fox and tiger')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

if args.dataset == 'elephant':
    path = '..\\data\\elephant_100x100_matlab.mat'
elif args.dataset == 'fox':
    path = '..\\data\\fox_100x100_matlab.mat'
elif args.dataset == 'tiger':
    path = '..\\data\\tiger_100x100_matlab.mat'
else:
    print("ERRORE: nome dataset errato!")
    exit(1)

def train(epoch, train_loader,model, optimizer):
    model.train()
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label) in enumerate(train_loader):
        bag_label = label[0]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data = data.float()
        data, bag_label = Variable(data), Variable(bag_label)

        optimizer.zero_grad()
        loss = model.calculate_objective(data, bag_label)
        train_loss += loss.data
        error, _, _, _ = model.calculate_classification_error(data, bag_label)
        train_error += error

        loss.backward()
        optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)
    print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss.cpu().numpy(), train_error))


def test(test_loader, model):
    model.eval()
    test_loss = 0.
    test_error = 0.
    predictions = []
    
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            bag_label = label[0]
            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
            data = data.float()
            data, bag_label = Variable(data), Variable(bag_label)
            loss = model.calculate_objective(data, bag_label)
            test_loss += loss.data
            error, predicted_label, prob_label, _ = model.calculate_classification_error(data, bag_label)
            predictions.append(prob_label)
            test_error += error

    test_error /= len(test_loader)
    test_loss /= len(test_loader)
    print('Test Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss.cpu().numpy(), test_error))
    return predictions, 1-test_error

def get_training_features(train_loader, base_model):
    # get features of training instances
    base_model.eval()

    tr_bags = []
    tr_labels = []
    tr_mask = []
    
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(train_loader):
            bag_label = label[0]
            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
            data = data.float()
            data, bag_label = Variable(data), Variable(bag_label)
            _, _, _, fea = base_model.calculate_classification_error(data, bag_label)
            tr_bags += fea
            # tr_bags.append(fea)
            tr_labels.append(bag_label)
            tr_mask += [batch_idx for i in range(data.shape[1])]
    return (tr_bags, tr_labels, tr_mask)

def get_model(model_name, num_references=None):
    if model_name == 'MInet':
        model = MINet(230, 1, pooling_mode='max')
        optimizer = optim.SGD(model.parameters(), lr=5e-4, weight_decay=0.005, momentum=0.9, nesterov=True)
    elif model_name == 'bsn':
        model = BSN(230, 1, pooling_mode='max', num_references=num_references)
        optimizer = optim.SGD(model.parameters(), lr=5e-4, weight_decay=0.01, momentum=0.9, nesterov=True)
    else:
        print("ERRORE: nome modello errato!")
        exit(1)
    return model, optimizer


def train_bsn(epoch, train_loader, model, optimizer, feas):
    tr_bags, tr_labels, tr_mask = feas
    model.train()
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label) in enumerate(train_loader):
        bag_label = label[0]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data = data.float()
        data, bag_label = Variable(data), Variable(bag_label)
        optimizer.zero_grad()
        loss, error, _ = model.calculate_objective(data, bag_label, tr_bags, tr_mask)
        train_loss += loss.data
        # error, _, _, _ = model.calculate_classification_error(data, bag_label, tr_bags, tr_mask)
        train_error += error

        loss.backward()
        optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)
    print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss.cpu().numpy(), train_error))


def test_bsn(test_loader, model, feas):
    tr_bags, tr_labels, tr_mask = feas
    model.eval()
    test_loss = 0.
    test_error = 0.
    predictions = []
    
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            bag_label = label[0]
            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
            data = data.float()
            data, bag_label = Variable(data), Variable(bag_label)
            loss, error, _ = model.calculate_objective(data, bag_label, tr_bags, tr_mask)
            test_loss += loss.data
            # error, predicted_label, prob_label, _ = model.calculate_classification_error(data, bag_label, tr_bags, tr_mask)
            # predictions.append(prob_label)
            test_error += error

    test_error /= len(test_loader)
    test_loss /= len(test_loader)
    print('Test Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss.cpu().numpy(), test_error))
    return predictions, 1-test_error



if __name__ == "__main__":   
    base_model_name = 'MInet'
    accs_base_model = np.zeros((args.run, args.folds), dtype=float)
    accs = np.zeros((args.run, args.folds), dtype=float)
    seeds = [args.seed+i*5 for i in range(args.run)]
    for irun in range(args.run):
        accs_v=[]
        accs_v_base_model=[]
        bags, labels=create_bags_mat(path=path)
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seeds[irun])
        for idx, (tr, ts) in enumerate(skf.split(bags, labels)):
    
            print('\n\nrun=', irun, '  fold=', idx)
            model, optimizer = get_model(base_model_name)

            bags_tr=bags[tr]
            y_tr=labels[tr]
            bags_ts=bags[ts]
            y_ts=labels[ts]

            loader_training = DataLoader(MyDataset(bags_tr, y_tr), batch_size=1)
            loader_test = DataLoader(MyDataset(bags_ts, y_ts), batch_size=1)

            for e in range(args.epochs):
                train(e, loader_training, model, optimizer)
            predictions, acc = test(loader_test, model)
            print (f'accuracy base model (fold {idx})=', acc)
            accs_base_model[irun][idx] = acc
            accs_v_base_model.append(acc)
            feas = get_training_features(loader_training, model)

            bsn, optimizer2 = get_model("bsn", len(feas[1]))
            for e in range(args.epochs):
                train_bsn(e, loader_training, bsn, optimizer2, feas)
            predictions, acc = test_bsn(loader_test, bsn, feas)
            print (f'accuracy bsn (fold {idx})=', acc)
            accs[irun][idx] = acc
            accs_v.append(acc)
        
        # print ("\nmean auc=", np.mean(aucs))
        print ("MI-net: mean acc=", np.mean(accs_v_base_model))
        print ("BSN:    mean acc=", np.mean(accs_v))

    print('\n\nFINAL MI-net: mean accuracy = ', np.mean(accs))
    print('FINAL MI-net: std = ', np.std(accs))
    print('FINAL BSN: mean accuracy = ', np.mean(accs))
    print('FINAL BSN: std = ', np.std(accs))
