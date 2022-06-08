import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def parse(data_str):
    data_items = data_str.replace(' ', '').split(',')
    data = dict()
    for item in data_items:
        key, value = item.split('=')
        data[key] = float(value)
        if key != 'Epoch':
            data[key] = data[key]
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-path', type = str, required = True)
    parser.add_argument('--log-path-comparison', type = str, default = None)
    
    args = parser.parse_args()
    log_path = args.log_path

    fp = open(log_path, 'r')
    datalines = fp.readlines()
    fp.close()

    print('total {} lines'.format(len(datalines)))

    start_epoch = 0

    timesteps, length, reward = [], [], []
    value_loss, action_loss = [], []
    for i, dataline in enumerate(datalines):
        data = parse(dataline)
        if i >= start_epoch:
            timesteps.append(data['iterations'])
            length.append(data['mean(len)'])
            reward.append(data['mean(reward)'])
            value_loss.append(data['value_loss'])
            action_loss.append(data['action_loss'])

    print('max', np.argmax(reward))

    yhat = savgol_filter(reward, 61, 3)

    plt.plot(reward)
    plt.plot(yhat)
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.show()
