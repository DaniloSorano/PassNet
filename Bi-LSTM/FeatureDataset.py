import os
import pickle as pk
import numpy as np
import random
import torch
import cv2
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
'''
    FeatureDataset class prepares the data for the DataLoader.
    It creates the sequence of features extracted by the feature_extraction.py script
'''
class FeatureDataset(Dataset):
    def __init__(self, match_code, model_type, seq_dim, reduce_data_el=None):
        self.reduce_data_el = reduce_data_el
        self.seq_dim = seq_dim
        filename = match_code + ".txt"
        features = []
        if reduce_data_el == None:
            with open("../../Data/Input/" + model_type + "/" + filename, "r") as file:
                line_len = len(file.readlines())
        else:
            line_len=reduce_data_el
        with open("../../Data/Input/" + model_type + "/" + filename, "r") as file:
            pbar_features = tqdm(total=line_len, desc="Read Features")
            i=0
            for line in file:
               frame_feature_string = line.split(", ")
               frame_feature = [float(el) for el in frame_feature_string]
               features.append(frame_feature)
               pbar_features.update(1)
               i+=1
               if reduce_data_el != None and reduce_data_el == i:
                   break
            pbar_features.close()

        self.features_tensor = torch.from_numpy(np.array(features))
        path_labels = '../../Data/Outputs/Binary Outputs/' + match_code + "_binary.pkl"
        labels = pk.load(open(path_labels, "rb"))
        self.labels_tensor = torch.from_numpy(np.array(labels))

    def __len__(self):
        if self.reduce_data_el == None:
            n_features = len(self.features_tensor)
            return n_features - self.seq_dim + 1
        else:
            return self.reduce_data_el - self.seq_dim + 1

    def __getitem__(self, start_idx):
        input_tensor = self.features_tensor[start_idx+1]
        inputs_seq = input_tensor.unsqueeze(0)
        finish_idx = start_idx + self.seq_dim
        for idx in range(start_idx + 1, finish_idx):
            input_tensor = self.features_tensor[start_idx+1]
            inputs_seq = torch.cat((inputs_seq, input_tensor.unsqueeze(0)))
        labels_seq = self.labels_tensor[start_idx:finish_idx]
        return ( inputs_seq, labels_seq )

if __name__ == "__main__":
    from tqdm import tqdm
    dataset = FeatureDataset("roma_juve_1", "vgg16",10)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    i=0
    device = torch.device(1)
    pbar = tqdm(total=len(dataset))
    for input, label in data_loader:
        input.to(device)
        print(input.shape, label.shape)
        i+=1
        pbar.update(1)
    pbar.close()
    print(i)
