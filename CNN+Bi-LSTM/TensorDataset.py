import os
import sys
import pickle as pk
import numpy as np
import random
import torch
import cv2
import pickle as pk
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as transforms
sys.path.insert(1, '../../YOLO/Utils')
from utils import extract_yolo_features
from utils import nn_ball
import torch.nn as nn

'''
    TensorDataset class prepares the data for the DataLoader.
    It creates the sequence of tensors extracted by the frame_extractor.py script
'''
class TensorDataset(Dataset):
    def __init__(self, match_code, seq_dim, reduce_data_el=None, yolo=False):
        self.match_code = match_code
        self.path_match = '../../Data/Input/frames_tensor/' + match_code
        self.seq_dim = seq_dim
        path_labels = '../../Data/Outputs/Binary Outputs/' + match_code + "_binary.pkl"
        labels = pk.load(open(path_labels, "rb"))
        self.labels_tensor = torch.from_numpy(np.array(labels))
        self.reduce_data_el = reduce_data_el
        self.yolo = yolo
        if yolo:
            self.yolo_features = extract_yolo_features("../../Data/Input/bbox/" + self.match_code + ".txt", 1280, 720, len(labels))

    def __len__(self):
        if self.reduce_data_el == None:
            n_frames = len(os.listdir(self.path_match))
            return n_frames - self.seq_dim + 1
        else:
            return self.reduce_data_el - self.seq_dim + 1

    def __getitem__(self, start_idx):
        frame_name = self.match_code.split("_")[0][0] + "_" + self.match_code.split("_")[1][0] + "_" + self.match_code.split("_")[2]
        inputs_seq = pk.load(open(self.path_match + "/" + frame_name + "_" + str(start_idx+1) + ".pickle", "rb"))
        if self.yolo:
            yolo_seq = torch.from_numpy(nn_ball(self.yolo_features[start_idx + 1], n_players=5)).unsqueeze(0)
        finish_idx = start_idx + self.seq_dim
        for idx in range(start_idx + 1, finish_idx):
            input_tensor = pk.load(open(self.path_match + "/" + frame_name + "_" + str(idx) + ".pickle", "rb"))
            inputs_seq = torch.cat((inputs_seq, input_tensor))
            if self.yolo:
                yolo_f = torch.from_numpy(nn_ball(self.yolo_features[idx], n_players=5))
                yolo_seq = torch.cat((yolo_seq, yolo_f.unsqueeze(0)))
        labels_seq = self.labels_tensor[start_idx:finish_idx]
        if self.yolo:
            return ( inputs_seq, labels_seq, yolo_seq )
        else:
            return ( inputs_seq, labels_seq )

if __name__ == "__main__":
    from tqdm import tqdm
    from Cnn_Lstm import Cnn_Lstm
    dataset = TensorDataset("roma_lazio_1", 1, None, yolo=True)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    model = Cnn_Lstm(input_dim=512, yolo_dimension=0,  hidden_dim=128, layer_dim=1,
                output_dim=1, readout_dim=2, drop_out=0.5, device=1, cnn_model="resnet18")
    pbar = tqdm(total=len(dataset))
    device = torch.device("cpu")
    n_frames = 0
    n_ball = 0
    for i, (input, labels, yolo_features) in enumerate(data_loader):
        yolo = yolo_features[0].type('torch.FloatTensor')
        n_frames += 1
        if yolo[0][0].item() == 0 and yolo[0][1].item() == 1:
            n_ball += 1
        pbar.update(1)
    pbar.close()
    print(n_ball/n_frames)
