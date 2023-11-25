import torch.utils.data as data
import numpy as np
from utils import process_feat
import torch
from torch.utils.data import DataLoader
torch.set_default_tensor_type('torch.FloatTensor')


class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.modality = args.modality
        self.is_normal = is_normal
        self.dataset = args.dataset
        if self.dataset == 'shanghai':
            if test_mode:
                self.rgb_list_file = args.TestListSh
            else:
                self.rgb_list_file = args.TrainListSh
        else:
            if test_mode:
                self.rgb_list_file = args.TestListUcf
            else:
                self.rgb_list_file = args.TrainListUcf

        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None


    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            if self.dataset == 'shanghai':
                if self.is_normal:
                    #self.list = self.list[63:]
                    #self.list = self.list[25:]
                    self.list = self.list[6:]
                    #self.list = self.list[16:]
                    print('normal list for shanghai tech')
                    print(self.list)
                else:
                    #self.list = self.list[:63]// sh
                    #self.list = self.list[:25]// ped1
                    self.list = self.list[:6] # ped2
                    #self.list = self.list[:16]// chuk
                    print('abnormal list for shanghai tech')
                    print(self.list)

            elif self.dataset == 'ucf':
                if self.is_normal:
                    self.list = self.list[810:]
                    print('normal list for ucf')
                    print(self.list)
                else:
                    self.list = self.list[:810]
                    print('abnormal list for ucf')
                    print(self.list)

    def __getitem__(self, index):

        label = self.get_label()  # get video level label 0/1
        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        features = np.array(features, dtype=np.float32)

        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            return features
        else:
            # process 10-cropped snippet feature
            features = features.transpose(1, 0, 2)  # [10, B, T, F]
            divided_features = []
            for feature in features:
                feature = process_feat(feature, 32)  # divide a video into 32 segments
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)

            return divided_features, label

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame
