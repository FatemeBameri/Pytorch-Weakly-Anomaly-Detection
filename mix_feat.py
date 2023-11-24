from pathlib import Path
import numpy as np
import torch

rootdir1 = Path('D:/First_Paper_code/SHRunColab1/chuk_swin/')
rootdir2 = Path('D:/First_Paper_code/SHRunColab1/chuk_I3d/')

videos1 = [str(f) for f in rootdir1.glob('*.npy')]
videos2 = [str(f) for f in rootdir2.glob('*.npy')]
outdir = 'D:/First_Paper_code/SHRunColab1/chuk_swin_I3d/'

for video in videos1:
    data1 = np.load(video)
    #name = video[39:]
    #name = video[42:]
    name = video[42:]
    data2 = np.load(str(rootdir2)+'/'+name)
    out_data = np.concatenate((data1, data2), axis = 2)
    np.save(outdir+name,out_data)


