import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, atomsize, lensize, k1, s1, f1, k2, s2, k3, s3, f3, k4, s4, n_hid, n_out):
        super(CNN, self).__init__()
        self.atomsize = atomsize
        self.lensize = lensize
        self.n_out = n_out
        self.k1 = k1
        self.s1 = s1
        self.f1 = f1
        self.k2 = k2
        self.s2 = s2
        self.k3 = k3
        self.s3 = s3
        self.f3 = f3
        self.k4 = k4
        self.s4 = s4

        self.l1 = (self.atomsize+(self.k1//2*2)-self.k1)//self.s1+1
        self.l2 = (self.l1+(self.k2//2*2)-self.k2)//self.s2+1
        self.l3 = (self.l2+(self.k3//2*2)-self.k3)//self.s3+1
        self.l4 = (self.l3+(self.k4//2*2)-self.k4)//self.s4+1

        self.conv1 = nn.Conv2d(1, f1, kernel_size=(k1, lensize), stride=(s1, 1), padding=(k1//2, 0))
        self.bn1 = nn.BatchNorm2d(f1)
        self.conv2 = nn.Conv2d(f1, f3, kernel_size=(k3, 1), stride=(s3, 1), padding=(k3//2, 0))
        self.bn2 = nn.BatchNorm2d(f3)
        self.fc3 = nn.Linear(self.l4*f3, n_hid)
        self.bn3 = nn.BatchNorm1d(n_hid)
        self.fc4 = nn.Linear(n_hid, n_out)



    def forward(self, x, t):
        y, sr = self.predict(x)
        loss = F.binary_cross_entropy_with_logits(y, t) + sr
        accuracy = (y.round() == t).float().mean()
        self.reporter({'loss': loss, 'accuracy': accuracy})
        return loss

    def predict(self, x):
        h = F.leaky_relu(self.bn1(self.conv1(x))) # 1st conv
        h = F.avg_pool2d(h, (self.k2, 1), stride=(self.s2, 1), padding=(self.k2//2, 0)) # 1st pooling
        h = F.leaky_relu(self.bn2(self.conv2(h))) # 2nd conv
        h = F.avg_pool2d(h, (self.k4, 1), stride=(self.s4, 1), padding=(self.k4//2, 0)) # 2nd pooling
        h = F.adaptive_max_pool2d(h, (self.l4, 1)) # global max pooling, fingerprint
        h = h.view(-1, self.l4 * self.f3) # flatten for fully connected layer
        h = self.fc3(h) # fully connected
        sr = 0.00001 * torch.mean(torch.log(1 + h * h)) # sparse regularization
        h = F.leaky_relu(self.bn3(h))
        return self.fc4(h), sr

    def fingerprint(self, x):
        h = F.leaky_relu(self.bn1(self.conv1(x))) # 1st conv
        h = F.avg_pool2d(h, (self.k2, 1), stride=(self.s2, 1), padding=(self.k2//2, 0)) # 1st pooling
        h = F.leaky_relu(self.bn2(self.conv2(h))) # 2nd conv
        h = F.avg_pool2d(h, (self.k3, 1), stride=(self.s3, 1), padding=(self.k3//2, 0)) # 2nd pooling
        h = F.adaptive_max_pool2d(h, (self.l4, 1)) # global max pooling, fingerprint
        return h.view(-1, self.l4 * self.f3).detach().cpu().numpy()

    def layer1(self, x):
        h = self.bn1(self.conv1(x)) # 1st conv
        return h.detach().cpu().numpy()

    def pool1(self, x):
        h = F.leaky_relu(self.bn1(self.conv1(x))) # 1st conv
        h = F.avg_pool2d(h, (self.k2, 1), stride=(self.s2, 1), padding=(self.k2//2, 0)) # 1st pooling
        return h.detach().cpu().numpy()

    def layer2(self, x):
        h = F.leaky_relu(self.bn1(self.conv1(x))) # 1st conv
        h = F.avg_pool2d(h, (self.k2, 1), stride=(self.s2, 1), padding=(self.k2//2, 0)) # 1st pooling
        h = self.bn2(self.conv2(h)) # 2nd conv
        return h.detach().cpu().numpy()

    def pool2(self, x):
        h = F.leaky_relu(self.bn1(self.conv1(x))) # 1st conv
        h = F.avg_pool2d(h, (self.k2, 1), stride=(self.s2, 1), padding=(self.k2//2, 0)) # 1st pooling
        h = F.leaky_relu(self.bn2(self.conv2(h))) # 2nd conv
        h = F.avg_pool2d(h, (self.k3, 1), stride=(self.s3, 1), padding=(self.k3//2, 0)) # 2nd pooling
        return h.detach().cpu().numpy()
