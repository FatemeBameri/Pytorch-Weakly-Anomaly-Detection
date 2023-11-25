import argparse

parser = argparse.ArgumentParser()
# feature-size--> swin: 1024; ResNet: 512; S3D:1024; I3D: 2048;//
# swin+ResNet: 1024+512; swin+S3D: 1024+1024; swin+I3D: 1024+2048
parser.add_argument('--feature-size', type=int, default=1024, help='size of feature (default: 2048)')
parser.add_argument('--modality', default='RGB', help='the type of the input, RGB,AUDIO, or MIX')
parser.add_argument('--TrainListSh', default='run_setting_info/chuk-s3-train.list', help='list of rgb features ')
parser.add_argument('--TestListSh', default='run_setting_info/chuk-s3-test.list', help='list of test rgb features ')
parser.add_argument('--gtsh', default='run_setting_info/gt_s3_chuk.npy', help='file of ground truth ')
parser.add_argument('--gpus', default=1, type=int, choices=[0], help='gpus')
parser.add_argument('--lr', type=str, default='[0.001]*50', help='learning rates for steps(list form)')
parser.add_argument('--batch-size', type=int, default=4, help='number of instances in a batch of data (default: 16)')
parser.add_argument('--workers', default=4, help='number of workers in dataloader')
parser.add_argument('--num-classes', type=int, default=1, help='number of class')
parser.add_argument('--dataset', default='shanghai', help='dataset to train on (default: )')
parser.add_argument('--plot-freq', type=int, default=10, help='frequency of plotting (default: 10)')
parser.add_argument('--max-epoch', type=int, default=50, help='maximum iteration to train (default: 100)')
