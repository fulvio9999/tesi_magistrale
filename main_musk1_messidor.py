import argparse
import torch.optim as optim
from torch.autograd import Variable
import torch
import numpy as np
from sklearn.metrics import roc_auc_score as auc_roc
from sklearn import metrics
from AttNet import Attention
from dataset import convertToBatch, load_dataset, split_data_label
from miNet import MiNet
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='FLV')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--folds', type=int, default=10, metavar='N',
                    help='number of folds for cross validation (default: 10)')
parser.add_argument('--run', type=int, default=5, metavar='N',
                    help='number of run (default: 5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='minet', help='Choose b/w attnet and minet')
parser.add_argument('--dataset', type=str, default='musk1', help='Choose b/w musk1 or messidor')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

if args.dataset != "musk1" and args.dataset != "messidor":
    print("ERRORE: nome dataset errato!")
    exit(1)

def train(epoch, train_loader, model, optimizer):
    model.train()
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label) in enumerate(train_loader):
        bag_label = label[0]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        # data = data.float()
        data, bag_label = Variable(data), Variable(bag_label)

        optimizer.zero_grad()
        loss = model.calculate_objective(data, bag_label)
        train_loss += loss.data
        error, _, _ = model.calculate_classification_error(data, bag_label)
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
            error, predicted_label, prob_label = model.calculate_classification_error(data, bag_label)
            predictions.append(prob_label)
            test_error += error

    test_error /= len(test_loader)
    test_loss /= len(test_loader)
    print('Test Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss.cpu().numpy(), test_error))
    return predictions, 1-test_error

def get_model():
    input_dim = 166 if args.dataset == "musk1" else 687
    if args.model == 'attnet':
        model = Attention(input_dim=input_dim).cuda() if args.cuda else Attention(input_dim=input_dim)
        optimizer = optim.SGD(model.parameters(), lr=5e-4, weight_decay=0.005, momentum=0.9, nesterov=True)
    elif args.model == 'minet':
        model = MiNet(input_dim, 1, pooling_mode='max')
        optimizer = optim.SGD(model.parameters(), lr=5e-4, weight_decay=0.005, momentum=0.9, nesterov=True)
    else:
        print("ERRORE: nome modello errato!")
        exit(1)
    return model, optimizer

if __name__ == '__main__':
    # perform five times 10-fold cross-validation experiments
    run = 5
    n_folds = 10
    accs = np.zeros((run, n_folds), dtype=float)
    for irun in range(run):
        accs_v=[]
        # aucs=[]
        dataset = load_dataset(args.dataset, n_folds)
        model, optimizer = get_model()
        for ifold in range(n_folds):
            print('\n\nrun=', irun, '  fold=', ifold)

            train_bags = dataset[ifold]['train']
            test_bags = dataset[ifold]['test']

            # convert bag to batch
            train_set = convertToBatch(train_bags)
            test_set = convertToBatch(test_bags)
            dimension = train_set[0][0].shape[1]
            
            for e in range(args.epochs):
                train(e, train_set, model, optimizer)
            predictions, acc = test(test_set, model)

            # _, y_ts = split_data_label(test_set)
            # # auc=auc_roc(y_ts, predictions)
            # auc = 0
            # print (f'auc (fold {ifold})=',auc)
            # f, t, a=metrics.roc_curve(y_ts, predictions)
            # AN=sum(x==0 for x in np.array(y_ts))
            # AP=sum(x==1 for x in np.array(y_ts))
            # TN=(1.0-f)*AN
            # TP=t*AP
            # Acc2=(TP+TN)/len(y_ts)
            # acc=max(Acc2)
            print (f'accuracy (fold {ifold})=', acc)
            accs_v.append(acc)
            # aucs.append(auc)
            accs[irun][ifold] = acc

        # print ("\nmean auc=", np.mean(aucs))
        print ("mean acc=", np.mean(accs_v))

    print('\n\nFINAL: mean accuracy = ', np.mean(accs))
    print('FINAL: std = ', np.std(accs))
