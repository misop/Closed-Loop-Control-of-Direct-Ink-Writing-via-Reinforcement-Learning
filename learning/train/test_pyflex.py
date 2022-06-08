import sys
import os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'train'))

from environments.pyflex import Pyflex
import datetime
import numpy as np

numFlexes = 5
numSteps = 10
pyflexes = []

for i in range(numFlexes):
    flex = Pyflex()
    pyflexes.append(flex)

for i in range(numFlexes):
    pyflexes[i].init(512, 512)

for i in range(numFlexes):
    pyflexes[i].reset(3, 1.0, 1024*1024)

from matplotlib import pyplot as plt

fig, plts = plt.subplots(1, numFlexes)
fig.set_size_inches(24,6)

frames = []
for i in range(numFlexes):
    frames.append(np.zeros((512,512)))

for s in range(numSteps):
    for i in range(numFlexes):
        px = i*(16 / numFlexes) - 8
        frames[i] = pyflexes[i].step(px, 2, 0, 5)
        plts[i].clear()
        plts[i].matshow(frames[i], cmap='gray', vmin=0, vmax=1)
    plt.show(block=False)
    plt.pause(0.001)

for k in range(3):

    print('reseting scenes...')

    for i in range(numFlexes):
        pyflexes[i].reset(3, 1.0, 1024*1024)

    for s in range(numSteps):
        for i in range(numFlexes):
            px = i*(16 / numFlexes) - 8
            frames[i] = pyflexes[i].step(px, 2, 0, 5)
            plts[i].clear()
            plts[i].matshow(frames[i], cmap='gray', vmin=0, vmax=1)
        plt.show(block=False)
        plt.pause(0.001)

