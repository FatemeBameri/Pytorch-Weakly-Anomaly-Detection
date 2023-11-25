import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
from sklearn import metrics
import os
from os import path

def test(dataloader, model, args, device,step,auc_list, acc_list):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0, device=device)

        for i, input in enumerate(dataloader):
            input = input.to(device)
            input = input.permute(0, 2, 1, 3)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abnormal, feat_normal,logits= model(inputs=input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            pred = torch.cat((pred, sig))

        if args.dataset == 'shanghai':
            gt = np.load(args.gtsh)
        else:
            gt = np.load(args.gtucf)

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16) 

        gt = list(gt)
        fpr, tpr, threshold = roc_curve(gt, pred)
        #rec_auc = auc(fpr, tpr)

        pred2 = np.round(pred)
        acc = (pred2== np.array(gt)).sum().item() / pred.shape[0]
        acc_list.append(acc)
        #precision, recall, th = precision_recall_curve(gt, pred)
        #pr_auc = auc(recall, precision)

        rec_auc = metrics.auc(fpr, tpr)

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='purple', lw=lw, label='ROC curve (area = %0.2f)' % rec_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.title('UCSD Ped2')
        plt.grid(color='blue', linestyle='-', linewidth=0.2, alpha = 0.9)
        os.makedirs('output_graphs', exist_ok=True)
        plt.savefig(path.join('output_graphs', 'roc_auc.jpg'))
        plt.close()
        auc_list.append(rec_auc)

        return auc_list, rec_auc , acc_list, gt, pred