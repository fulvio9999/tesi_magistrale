import torch
import torch.nn as nn
import torch.nn.functional as F

class Score_pooling(nn.Module):
    def __init__(self, output_dim, input_dim=64, pooling_mode='max'):
        super(Score_pooling, self).__init__()
        self.output_dim = output_dim
        self.pooling_mode = pooling_mode
        self.fc = nn.Linear(input_dim, output_dim)
    
    def choice_pooling(self, x):
        if self.pooling_mode == 'max':
            return torch.max(x, dim=0, keepdim=True)[0]
        if self.pooling_mode == 'lse':
            return torch.log(torch.mean(torch.exp(x), dim=0, keepdim=True))
        if self.pooling_mode == 'ave':
            return torch.mean(x, dim=0, keepdim=True)

    def forward(self, x):
        # compute instance-level score
        x = self.choice_pooling(x)
        x = self.fc(x)
        output = torch.sigmoid(x)
        # do-pooling operator
        # output = self.choice_pooling(x)
        return output

class MINet(nn.Module):
    def __init__(self, input_dim, output_dim, pooling_mode='max'):
        super(MINet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(p=0.5)
        self.score_pooling = Score_pooling(output_dim, 64, pooling_mode)

    def forward(self, x):
        x = x.squeeze(0)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        Y_prob = self.score_pooling(x).squeeze()
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, x
    
    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, _ = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        loss = nn.BCELoss()(Y_prob, Y)
        # loss = self.compute_loss(Y_prob, Y)
        return loss
    
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        Y_prob, Y_hat, fea = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()
        return error, Y_hat, Y_prob, fea
    
    def compute_loss(self, logits, y):
        cross_entropy = -y * torch.log(logits + 1e-7) - (1 - y) * torch.log(1 - logits + 1e-7)
        cross_entropy_mean = torch.mean(cross_entropy)
        
        regularization_losses = torch.sum(torch.stack([param.norm(2) for param in self.parameters()]))
        
        return cross_entropy_mean + 0.001 * regularization_losses