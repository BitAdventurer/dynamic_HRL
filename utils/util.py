import torch
import config
import numpy as np
import os
args = config.parser.parse_args()

def get_device():
    """Returns the appropriate device (CPU or CUDA)"""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_parameter():
    file_path = args.path + '/feature_extract/weight_and_bias.pth'

    params = torch.load(file_path)

    return params['weight'], params['bias']

def load_signal_data():
    data_dir = os.getcwd() + '/data/Signal_split'
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    data = {}

    for file in data_files:
        file_path = os.path.join(data_dir, file)
        data[file] = np.load(file_path)
    #dic={name:value}
    return data

def load_dfc_data():
    data_dir = os.getcwd() + '/data/Signal_dFC_split'
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    data = {}

    for file in data_files:
        file_path = os.path.join(data_dir, file)
        data[file] = np.load(file_path)
    #dic={name:value}
    return data


def get_shape(lst):
    shape = []
    while isinstance(lst, list):
        shape.append(len(lst))
        lst = lst[0] if lst else []
    return tuple(shape)




def soft_update(target, source, tau):
    """
    Perform a soft update (EMA) of the target network parameters.
    target: target network (nn.Module)
    source: source network (nn.Module)
    tau: float, update coefficient (0 < tau < 1)
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

if __name__=='__main__':
    pass