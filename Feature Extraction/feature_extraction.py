# Code for feature extraction
from PIL import Image
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from tqdm import tqdm
import torchvision.models as models
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import image
import cv2
import os, sys
import pickle as pk

'''
This script allows to extract feature from the match video.
Whent the script start from command line is necessary to define two arguments:
the pretrained model and filename of the video
'''

# These variables define some default parameters as the resolution(240p) and the normalization
scaler = transforms.Resize((352, 240)) #240p
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def copy_data(m, i, o):
    my_embedding.copy_(o.data)

filename = sys.argv[1]
print ("Feature extraction from: ", filename)
vidcap = cv2.VideoCapture("../Video/" + filename)

#Different pretrained models can be used to extract the features
if sys.argv[2] == "vgg16" :
    cnn = models.vgg16(pretrained=True)
    cnn.classifier = nn.Sequential(*list(cnn.classifier.children())[:-3])
    input_dim = 4096
elif sys.argv[2] == "resnet34" :
    cnn = models.resnet34(pretrained=True)
    layer = cnn.avgpool
    my_embedding = torch.zeros(1, 512)
elif sys.argv[2] == "resnet18" :
    cnn = models.resnet18(pretrained=True)
    cnn = nn.Sequential(*(list(cnn.children())[:-1]))
    input_dim = 512
elif sys.argv[2] == "resnet50" :
    cnn = models.resnet50(pretrained=True)
    layer = cnn.avgpool
    my_embedding = torch.zeros(2, 512)
else:
    cnn = models.vgg19(pretrained=True)
    layer = cnn.classifier[3]
    my_embedding = torch.zeros(2, 4096)

if not os.path.exists( "../Data/Input/" + str(sys.argv[2])):
    os.makedirs( "../Data/Input/" + str(sys.argv[2]),mode=0o0775)

cnn.eval()
total_frame = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
pbar = tqdm(total=int(total_frame))
count = 1
frame_number = 1
frames_feature = []

success,image = vidcap.read()

while success:

    if count in [1,6,11,16,21]:
      image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
      image = Image.fromarray(image)
      image = normalize(to_tensor(scaler(image))).unsqueeze(0)

      out = cnn(image)
      out = out.view(input_dim)

      frames_feature.append(str(out.tolist()).strip("[]"))

      if len(frames_feature) == 10:
        for frame in frames_feature:
          file = open("../Data/Input/" + sys.argv[2] + "/" + str(filename.split('.')[0]) + ".txt", "a+")
          file.write(frame + "\n")
          file.close()
        frames_feature = []

    if count == 25:
        count = 1
    else:
        count += 1

    frame_number += 1

    success,image = vidcap.read()

    pbar.update(1)

pbar.close()
