import numpy as np
import sys
import os
from termcolor import colored, cprint
import matplotlib.pyplot as plt

target_dir = sys.argv[1]
dataDict = {}

for dirpath, _, filenames in os.walk(target_dir):
    print(dirpath)
    acc = np.zeros((100, 1))
    loss = np.zeros((100, 1))
    flag = False
    for name in filenames:
        if name == 'AccData.npy':
            flag = True
            acc = np.load(os.path.join(dirpath, name))
        if name == 'LossData.npy':
            flag = True
            loss = np.load(os.path.join(dirpath, name))
    if flag == True:
        dataDict[dirpath.split('/')[1]] = {'acc': acc, 'loss': loss}

plt.figure(figsize=(8, 8))
plt.title("Loss in %s" % target_dir)
plt.axis((0, 100, 0, 1))

for attr, value in dataDict.items():
    plt.plot(value['loss'], label = attr)

plt.legend()

plt.show()

plt.figure(figsize=(8, 8))
plt.title("Acc in %s" % target_dir)
plt.axis((0, 100, 0, 1))

for attr, value in dataDict.items():
    plt.plot(value['acc'], label = attr)

plt.legend()

plt.show()
