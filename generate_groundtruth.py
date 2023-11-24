from pathlib import Path
import numpy as np
import math

# Reference code: https://github.com/tianyu0207/RTFM/blob/main/list/make_gt_sh.py

rootdir = Path('D:/ground_truth_demo/Mask2/')
test_dir = "D:/Avenue_Dataset/chuk_swin"
videos = [str(f) for f in rootdir.glob('*.npy')]
print(videos)

abnormal_videos = []
for video in videos:
    #s = video[53:]// sh
    #s = video[31:]// ped2
    #s = video[27:] #// ped1
    s = video[60:]
    abnormal_videos.append(s)

rf = open("chuk-s3-test.txt", "r")
Lines = rf.readlines()
test_video = []
# Strips the newline character
for line in Lines:
    test_video.append(line[0:len(line)-1])
rf.close()

n_test=[]
ab_test = []
general_gt = []

for v in test_video:
    if v in abnormal_videos:
        m = np.load(str(rootdir) + "/" + v)
        index = math.floor(len(m) / 16)*16
        general_gt.extend(m[:index])
    else:
        mn= np.load(test_dir + "/" + v)
        indexn = mn.shape[0]*16
        for i in range(indexn):
            general_gt.append(0)
np.save('gt_s3_chuk.npy',np.array(general_gt))
print()

