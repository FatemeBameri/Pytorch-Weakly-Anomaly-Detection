from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils import save_best_record
from model import Model
from dataset import Dataset
from train_phase import train
from test_phase import test
import default_setting
from tqdm import tqdm
from config import *
import matplotlib.pyplot as plt
from os import path
from matplotlib.pyplot import close

if __name__ == '__main__':
    args = default_setting.parser.parse_args()
    config = Config(args)

    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=False)

    model = Model(args.feature_size, args.batch_size)
    feature_dim = args.feature_size

    for name, value in model.named_parameters():
        print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    optimizer = optim.Adam(model.parameters(),
                           lr=config.lr[0], weight_decay=0.0005)

    test_info = {"epoch": [], "test_AUC": []}
    best_AUC = -1
    output_path = ''  # put your own path here

    train_losses = []
    auc_list = []
    acc_list = []

    folder_save = 'ped2_out'
    dataset = 'UCSD Ped2'

    for step in tqdm(
            range(1, args.max_epoch + 1),
            total=args.max_epoch,
            dynamic_ncols=True
    ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]

        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)

        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)

        train_losses = train(loadern_iter, loadera_iter, model, args.batch_size, optimizer, device, step, train_losses)

        fig = plt.figure(1)
        if step == args.max_epoch:
            plt.plot(train_losses, '-d', label='Training loss', color="Magenta")
            plt.legend(loc="upper right", shadow=True)
        else:
            plt.plot(train_losses, '-d', color="Magenta")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(dataset)
        plt.grid(color='blue', linestyle='-', linewidth=0.2, alpha = 0.9)
        os.makedirs(folder_save, exist_ok=True)
        plt.savefig(path.join(folder_save, 'train_loss.jpg'))
        close(fig)

        auc_list, auc, acc_list, gt, pred= test(test_loader, model, args, device, step, auc_list, acc_list)
        test_info["epoch"].append(step)
        test_info["test_AUC"].append(auc)
        if step > 4:

            if test_info["test_AUC"][-1] > best_AUC:
                best_AUC = test_info["test_AUC"][-1]
                # torch.save(model.state_dict(), './ckpt/' + args.model_name + '{}-i3d.pkl'.format(step))
                save_best_record(test_info, os.path.join(output_path, '{}-step-AUC.txt'.format(step)))
                np.save('mypred.npy', pred)
                np.save('mygt.npy', gt)

        fig = plt.figure(2)
        plt.plot(auc_list, 'cs-')
        plt.xlabel('Epoch')
        plt.ylabel('Auc')
        plt.title(dataset)
        plt.grid(color='blue', linestyle='-', linewidth=0.2, alpha = 0.9)
        os.makedirs(folder_save, exist_ok=True)
        plt.savefig(path.join(folder_save, 'auc.jpg'))

        fig = plt.figure(3)
        plt.plot(acc_list, 'g^-')
        plt.xlabel('Epoch')
        plt.ylabel('Test Accuracy')
        plt.title(dataset)
        plt.grid(color='blue', linestyle='-', linewidth=0.2, alpha = 0.9)
        os.makedirs(folder_save, exist_ok=True)
        plt.savefig(path.join(folder_save, 'acc.jpg'))


    #torch.save(model.state_dict(), './logs/' + args.model_name + 'model.pkl')
