import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import close


pred = np.load('mypred.npy')
gt = np.load('mygt.npy')

datap = pred[0:433]
datag = gt[0:433]

fig = plt.figure(1)

fig, axs = plt.subplots(2,sharex=True)
axs[0].plot(datap,'tab:red')
axs[0].set_title('Prediction')
axs[1].plot(datag,'tab:green')
axs[1].set_title('Ground truth')

for ax in axs.flat:
    ax.grid(color='b', linestyle='-', linewidth=0.5, alpha=0.5)

for ax in axs.flat:
    ax.set(xlabel='Frames', ylabel='Anomaly Score')

for ax in axs.flat:
    ax.label_outer()

fig.savefig(('gp.jpg'))


plt.plot(datap, '-', color="red")
plt.xlabel('Frames')
plt.ylabel('Anomaly Score')
plt.grid(color='red', linestyle='-', linewidth=0.2, alpha = 0.9)
plt.savefig('p.jpg')
close(fig)

fig = plt.figure(2)
plt.plot(datag, '-', color="black")
plt.xlabel('Frames')
plt.ylabel('Anomaly Score')
plt.grid(color='black', linestyle='-', linewidth=0.2, alpha = 0.9)
plt.savefig('g.jpg')
close(fig)




