import torch.nn
from torch.distributions.half_cauchy import HalfCauchy
from torch.distributions.exponential import Exponential
from torch.distributions.geometric import Geometric
from torch.distributions.gumbel import Gumbel
from torch.distributions.cauchy import Cauchy
from torch.distributions.gamma import Gamma
from torch.distributions.half_normal import HalfNormal
from torch.distributions.normal import Normal
from torch.distributions.studentT import StudentT
from torch.distributions.beta import Beta
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture


class Claculating_param_pdf(torch.nn.Module):
    def __init__(self, type_dtb=None):
        super(Claculating_param_pdf, self).__init__()
        self.type_dtb = type_dtb
        self.cauchy = Cauchy(torch.tensor([2.0]), torch.tensor([1.5])) # (2,1.5)
        self.half_cauchy = HalfCauchy(torch.tensor([0.5]))  # halfcauchy distribution
        self.exponential = Exponential(torch.tensor([1.5]))
        self.geometric = Geometric(torch.tensor([0.3]))
        self.gumble = Gumbel(torch.tensor([1.0]), torch.tensor([2.0]))
        self.gamma = Gamma(torch.tensor([1.0]), torch.tensor([2.0]))
        self.half_normal = HalfNormal(torch.tensor([1.0]))
        self.normal = Normal(torch.tensor([0.0]), torch.tensor([2.0]))
        self.student = StudentT(torch.tensor([2.0]))
        self.beta = Beta(torch.tensor([1.0]), torch.tensor([3.0]))


    def forward(self, input):
        if self.type_dtb == 'Cauchy':
            return 10**self.cauchy.log_prob(input.cpu())
        if self.type_dtb == 'HalfCauchy':
            return 10 ** self.half_cauchy.log_prob(input)
        if self.type_dtb == 'Exponential':
            return 10 ** self.exponential.log_prob(input)
        if self.type_dtb == 'Geometric':
            return 10 ** self.geometric.log_prob(input)
        if self.type_dtb == 'Gumbel':
            return 10 ** self.gumble.log_prob(input)
        if self.type_dtb == 'Gamma':
            return 10 ** self.gamma.log_prob(input)
        if self.type_dtb == 'HalfNormal':
            return 10 ** self.half_normal.log_prob(input)
        if self.type_dtb == 'Normal':
            return 10 ** self.normal.log_prob(input)
        if self.type_dtb == 'StudentT':
            return 10 ** self.student.log_prob(input)
        if self.type_dtb == 'Beta':
            return 10 ** self.beta.log_prob(input)


class Claculating_non_param_pdf(torch.nn.Module):
    def __init__(self, type_dtb=None):
        super(Claculating_non_param_pdf, self).__init__()
        self.type_dtb = type_dtb
        # kernel =‘gaussian’, ‘tophat’, ‘epanechnikov’, ‘exponential’, ‘linear’, ‘cosine’
        self.kdea = KernelDensity(kernel='gaussian', bandwidth=1.5)
        self.kden = KernelDensity(kernel='gaussian', bandwidth=1.5)
        self.gmm = GaussianMixture(n_components=3, covariance_type="full")

    def forward(self, input):

        dist_in = torch.zeros(input.shape[0], input.shape[1])

        if self.type_dtb == 'kdea':

            for numvideo in range(input.shape[0]):
                kde_in = self.kdea.fit(input[numvideo, :, :].cpu().detach().numpy())
                probabilities = kde_in.score_samples(input[numvideo, :, :].cpu().detach().numpy())
                dist_in[numvideo, :] = torch.tensor(probabilities)
        if self.type_dtb == 'kden':

            for numvideo in range(input.shape[0]):
                kde_in = self.kden.fit(input[numvideo, :, :].cpu().detach().numpy())
                probabilities = kde_in.score_samples(input[numvideo, :, :].cpu().detach().numpy())
                dist_in[numvideo, :] = torch.tensor(probabilities)

        if self.type_dtb == 'gmm':
            for numvideo in range(input.shape[0]):
                clf = self.gmm.fit(input[numvideo, :, :].cpu().detach().numpy())
                probabilities = -clf.score_samples(input[numvideo, :, :].cpu().detach().numpy())
                dist_in[numvideo, :] = torch.tensor(probabilities)

        return dist_in


def diff_seg(num_videos,num_seg,features, k_near):

    total_diff_abnormal = []
    total_diff_index_q = []

    for num_video in range(num_videos):
        diff_abnormal = []
        diff_index_q = []
        for p in range(num_seg - 1):
            for q in range(p + 1, num_seg):
                if (abs(p - q) < k_near):
                    xp = features[num_video, p].view(1, -1)
                    xq = features[num_video, q].view(1, -1)
                    #diff = torch.cdist(xp,xq, p=2)
                    #diff = torch.exp(torch.cdist(xp,xq, p=2))
                    diff = torch.exp(torch.mean(torch.abs(xp-xq)))
                    diff_abnormal.append(diff)
                    diff_index_q.append(q)
        total_diff_abnormal.append(diff_abnormal)
        total_diff_index_q.append(diff_index_q)

    return total_diff_abnormal, total_diff_index_q


def calculating_dist(type_dist, features, index_dist):

    if type_dist == 'param':

        param_dists = ['Cauchy', 'HalfCauchy', 'Exponential', 'Geometric', 'Gumbel'
            ,'Gamma', 'HalfNormal', 'Normal','StudentT', 'Beta']

        type_dtb = param_dists[index_dist]
        pdf_features = Claculating_param_pdf(type_dtb)
        dist_features = pdf_features(features)

    else:
        non_param_dists = ['kdea','kden', 'gmm']
        type_non_dbt = non_param_dists[index_dist]
        pdf_non_param = Claculating_non_param_pdf(type_non_dbt)
        dist_features = pdf_non_param(features)

    return dist_features


def correct_index(n_size, idx_abn, diff_index_q):

    total_new_idx_abn = torch.zeros([n_size, 3], dtype=torch.int64)

    for tr in range(n_size):
        temporal0 = idx_abn[tr, 0]
        total_new_idx_abn[tr, 0] = diff_index_q[tr, temporal0]

        temporal1 = idx_abn[tr, 1]
        total_new_idx_abn[tr, 1] = diff_index_q[tr, temporal1]

        temporal2 = idx_abn[tr, 2]
        total_new_idx_abn[tr, 2] = diff_index_q[tr, temporal2]

    return total_new_idx_abn
