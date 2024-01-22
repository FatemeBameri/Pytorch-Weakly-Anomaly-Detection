import torch
import torch.nn as nn

torch.set_default_tensor_type('torch.FloatTensor')
import torch.nn.init as torch_init
from Cal_pdf import diff_seg, calculating_dist, mean_diff_seg
from Cal_pdf import correct_index
from atten_method import RTFM_Attention


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class Model(nn.Module):
    def __init__(self, n_features, batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.num_segments = 32
        self.k_abn = self.num_segments // 10
        self.k_nor = self.num_segments // 10

        self.Atten = RTFM_Attention(len_feature=2048)

        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        self.drop_out = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.apply(weight_init)

    def forward(self, inputs):

        k_abn = self.k_abn
        out = inputs
        bs, ncrops, t, f = out.size()

        out = out.view(-1, t, f)

        out = self.Atten(out)

        out = self.drop_out(out)

        features = self.relu(out)

        scores = self.relu(self.fc1(features))
        scores = self.drop_out(scores)
        scores = self.relu(self.fc2(scores))
        scores = self.drop_out(scores)
        scores = self.sigmoid(self.fc3(scores))
        scores = scores.view(bs, ncrops, -1).mean(1)
        scores = scores.unsqueeze(dim=2)
        normal_features = features[0:self.batch_size * 10]
        normal_scores = scores[0:self.batch_size]
        abnormal_features = features[self.batch_size * 10:]
        abnormal_scores = scores[self.batch_size:]

        ###########################################################

        if bs == 1:
            abnormal_features = normal_features
            abnormal_scores = normal_scores
            n_size = bs
        else:
            n_size = int(bs / 2)

        ############## Estimation Distribution for videos###############

        abnormal_features2 = abnormal_features.view(n_size, ncrops, t, f).mean(1)
        normal_features2 = normal_features.view(n_size, ncrops, t, f).mean(1)
        type_dist = ['param', 'non-param']
        test_type = type_dist[0]

        # (0,2)-(2,0)-(1,4)-(4,1)-(5,6)-(6,5)-(8,2)-(2,8)-(7,4)-(4,7)-kde
        index_dist = 6
        index_distn = 5
        abnormal_dist_features = calculating_dist(test_type, abnormal_features, index_dist)
        normal_dist_features = calculating_dist(test_type, normal_features, index_distn)

        # --------------------------------------------------------------------------------------
        topk = 4
        topk_ab = torch.topk(abnormal_dist_features, topk)
        topk_n = torch.topk(normal_dist_features, topk)

        ######################### Comparing Abnormal Distributions #############

        k_near = 3  # 1,2,3,4,5,6,7,8,9
        total_diff_abnormal, total_diff_index_q = diff_seg(n_size, t, abnormal_features2, k_near)
        ############################################################################################
        total_diff_abnormal = torch.tensor(total_diff_abnormal)
        select_idx = torch.ones_like(total_diff_abnormal)
        select_idx = self.drop_out(select_idx)

        #######  process abnormal videos -> select top3 feature magnitude  #####
        diff_drop = total_diff_abnormal * select_idx
        idx_abn = torch.topk(diff_drop, k_abn, dim=1)[1]
        diff_index_q = torch.tensor(total_diff_index_q)

        total_new_idx_abn = correct_index(n_size, idx_abn, diff_index_q)

        idx_abn_feat = total_new_idx_abn.unsqueeze(2).expand([-1, -1, f])

        abnormal_features = abnormal_dist_features.view(n_size, ncrops, t, f)
        #abnormal_features = abnormal_features.view(n_size, ncrops, t, f)
        abnormal_features = abnormal_features.permute(1, 0, 2, 3)
        # idx_abn_feat = idx_abn_feat.to('cuda')
        idx_abn_feat = idx_abn_feat.to('cpu')
        total_select_abn_feature = torch.zeros(0, device=inputs.device)
        for abnormal_feature in abnormal_features:
            feat_select_abn = torch.gather(abnormal_feature, 1,
                                           idx_abn_feat)  # top 3 features in abnormal bag
            total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))

        idx_abn_score = total_new_idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])
        # abnormal_scores = abnormal_scores.to('cuda')
        # idx_abn_score = idx_abn_score.to('cuda')
        abnormal_scores = abnormal_scores.to('cpu')
        idx_abn_score = idx_abn_score.to('cpu')
        score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score),
                                    dim=1)  # top 3 scores in abnormal bag

        ################################# Comaring Distributions ###################################

        # Estimation Distribution for normal videos
        total_diff_normal, total_diff_index_nq = diff_seg(n_size, t, abnormal_features2, k_near)
        ############################################################################################
        total_diff_normal = torch.tensor(total_diff_normal)
        select_idx = torch.ones_like(total_diff_normal)
        select_idx = self.drop_out(select_idx)

        #######  process abnormal videos -> select top3 features  ######
        nfea_magnitudes_drop = total_diff_normal * select_idx
        idx_n = torch.topk(nfea_magnitudes_drop, k_abn, dim=1)[1]
        diff_index_nq = torch.tensor(total_diff_index_nq)

        total_new_idx_n = correct_index(n_size, idx_n, diff_index_nq)

        idx_n_feat = total_new_idx_n.unsqueeze(2).expand([-1, -1, f])

        normal_features = normal_dist_features.view(n_size, ncrops, t, f)
        #normal_features = normal_features.view(n_size, ncrops, t, f)
        normal_features = normal_features.permute(1, 0, 2, 3)
        # idx_n_feat = idx_n_feat.to('cuda')
        idx_n_feat = idx_n_feat.to('cpu')
        total_select_n_feature = torch.zeros(0, device=inputs.device)
        for normal_feature in normal_features:
            feat_select_n = torch.gather(normal_feature, 1,
                                         idx_n_feat)  # top 3 features in normal bag
            total_select_n_feature = torch.cat((total_select_n_feature, feat_select_n))

        idx_n_score = total_new_idx_n.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
        # normal_scores = normal_scores.to('cuda')
        # idx_n_score = idx_n_score.to('cuda')
        normal_scores = normal_scores.to('cpu')
        idx_n_score = idx_n_score.to('cpu')
        score_normal = torch.mean(torch.gather(normal_scores, 1, idx_n_score),
                                  dim=1)  # top 3 scores in normal bag

        feat_abnormal = total_select_abn_feature
        feat_normal = total_select_n_feature
        feat_dist_abnormal = topk_ab.values
        feat_dist_normal = topk_n.values

        scores = torch.cat((normal_scores, abnormal_scores), 0)

        return score_abnormal, score_normal, feat_dist_abnormal, feat_dist_normal, \
            feat_abnormal, feat_normal, scores
