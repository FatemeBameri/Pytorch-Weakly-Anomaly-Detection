import numpy as np
import torch
import torch.nn.functional as F
torch.set_default_tensor_type('torch.FloatTensor')
from torch.nn import L1Loss
from torch.nn import MSELoss
from torchmetrics.functional import kl_divergence

# Train based on original version + proposed (distribution)

# calculate the jenson shannon divergence
def js_divergence(p, q,log_prob):
    m = 0.5 * (p + q)
    return (0.5 * kl_divergence(p, m,log_prob) + 0.5 * kl_divergence(q, m,log_prob))

def sparsity(arr, batch_size, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2*loss


def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]

    loss = torch.sum((arr2-arr)**2)

    return lamda1*loss


def l1_penalty(var):
    return torch.mean(torch.norm(var, dim=0))


class SigmoidMAELoss(torch.nn.Module):
    def __init__(self):
        super(SigmoidMAELoss, self).__init__()
        from torch.nn import Sigmoid
        self.__sigmoid__ = Sigmoid()
        self.__l1_loss__ = MSELoss()

    def forward(self, pred, target):
        return self.__l1_loss__(pred, target)


class SoftmaxMAELoss(torch.nn.Module):
    def __init__(self):
        super(SoftmaxMAELoss, self).__init__()
        from torch.nn import Softmax
        self.__softmax__ = Softmax()
        self.__l1_loss__ = MSELoss()

    def forward(self, pred, target):
        return self.__l1_loss__(pred, target)

class SigmoidCrossEntropyLoss(torch.nn.Module):
    # Implementation Reference: http://vast.uccs.edu/~adhamija/blog/Caffe%20Custom%20Layer.html
    def __init__(self):
        super(SigmoidCrossEntropyLoss, self).__init__()

    def forward(self, x, target):
        tmp = 1 + torch.exp(- torch.abs(x))
        return torch.abs(torch.mean(- x * target + torch.clamp(x, min=0) + torch.log(tmp)))


class RTFM_loss(torch.nn.Module):
    def __init__(self, alpha, margin):
        super(RTFM_loss, self).__init__()
        self.alpha = alpha
        self.margin = margin
        self.sigmoid = torch.nn.Sigmoid()
        self.mae_criterion = SigmoidMAELoss()
        self.criterion = torch.nn.BCELoss()

    def forward(self, score_normal, score_abnormal, nlabel, alabel, feat_n, feat_a,
                feat_abnormal, feat_normal):
        label = torch.cat((nlabel, alabel), 0)
        score_abnormal = score_abnormal
        score_normal = score_normal

        score = torch.cat((score_normal, score_abnormal), 0)
        score = score.squeeze()

        #label = label.cuda()
        label = label

        gt = np.array(label.cpu())
        pred = np.round(np.array(score.cpu().detach().numpy()))
        acc = (pred == gt).sum().item() / pred.shape[0]
        error = 1 - acc
        #print('%%%%%%%%%%%%%%')
        #print(acc)
        #error_list.append(error)

        loss_cls = self.criterion(score, label)  # BCE loss in the score space

        #loss_abn = torch.abs(self.margin - torch.norm(torch.mean(feat_a, dim=1), p=2, dim=1))
        #loss_abn = torch.abs(self.margin - torch.norm(torch.mean(feat_a, dim=1), p=2, dim=0))

        #loss_nor = torch.norm(torch.mean(feat_n, dim=1), p=2, dim=1)
        #loss_nor = torch.norm(torch.mean(feat_n, dim=1), p=2, dim=0)

        #loss_rtfm = torch.mean((loss_abn + loss_nor) ** 2)
        #xp = feat_n.view(1, -1)
        #xq = feat_a.view(1, -1)
        #diff = kl_divergence(xp, xq, log_prob='True').item()
        #loss_total = loss_cls + self.alpha * loss_rtfm
        log_prob = False
        ###### Param method #######
        feat_n = feat_n.view(1, -1)
        feat_a = feat_a.view(1, -1)
        ###### Param method #######
        diff = js_divergence(feat_n, feat_a, log_prob)
        #diff2=torch.mean(torch.cdist(feat_abnormal,feat_normal, p=2))
        #Myrange=torch.tensor(400.0)
        #Myrange = torch.tensor(200.0)# ped2 ==200# chuk ==100
        #R = torch.tensor(50) #1, 10,50, 100,150
        loss_rtfm1 = torch.max(torch.tensor(0), 1-diff)
        #print(diff2)
        #loss_rtfm2 = R - (torch.div(torch.min(Myrange, diff2), Myrange)) * R
        loss_rtfm = loss_rtfm1

        loss_total = loss_cls + self.alpha * loss_rtfm

        return loss_total

def train(nloader, aloader, model, batch_size, optimizer, device,step,train_losses):
    with torch.set_grad_enabled(True):
        model.train()

        ninput, nlabel = next(nloader)
        ainput, alabel = next(aloader)

        input = torch.cat((ninput, ainput), 0).to(device)

        score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abnormal,\
            feat_normal,scores  = model(input)  # b*32  x 2048

        scores = scores.view(batch_size * 32 * 2, -1)

        scores = scores.squeeze()
        abn_scores = scores[batch_size * 32:]

        nlabel = nlabel[0:batch_size]
        alabel = alabel[0:batch_size]

        loss_criterion = RTFM_loss(0.0001, 100)
        loss_sparse = sparsity(abn_scores, batch_size, 8e-3)
        loss_smooth = smooth(abn_scores, 8e-4)
        cost = loss_criterion(
            score_normal, score_abnormal, nlabel, alabel, feat_select_normal, feat_select_abn
            , feat_abnormal,feat_normal)+loss_sparse+loss_smooth

        train_losses.append(cost.item())

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    return train_losses
