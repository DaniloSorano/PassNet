import torch
import torch.nn as nn
import torchvision.models as models
'''
Class that implements the CNN pre-trained model with fine-tuning.
This class implements also the model with Object Positions extracted from YOLOv3
'''
class Cnn_Lstm(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, readout_dim, drop_out, device, cnn_model='vgg19', yolo_dimension=0):
        super(Cnn_Lstm, self).__init__()

        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        self.yolo_dimension = yolo_dimension
        self.input_dim = input_dim
        self.layer_dim = layer_dim

        #This class allows to uses different pre-trained models
        if cnn_model == "vgg16" :
            self.cnn = models.vgg16(pretrained=True)
            self.cnn.classifier = nn.Sequential(*list(self.cnn.classifier.children())[:-3])
        elif cnn_model == "resnet34" :
            self.cnn = models.resnet34(pretrained=True)
            self.cnn = nn.Sequential(*(list(self.cnn.children())[:-1]))
        elif cnn_model == "resnet18" :
            self.cnn = models.resnet18(pretrained=True)
            self.cnn = nn.Sequential(*(list(self.cnn.children())[:-1]))
        elif cnn_model == "resnet50" :
            self.cnn = models.resnet50(pretrained=True)
            self.cnn = nn.Sequential(*(list(self.cnn.children())[:-1]))
        else:
            self.cnn = models.vgg19(pretrained=True)
            self.cnn.classifier = nn.Sequential(*list(self.cnn.classifier.children())[:-3])
        #Here we create the istance for the Bidirectional LSTM
        if yolo_dimension != 0:
            self.lstm = nn.LSTM(input_dim+yolo_dimension, hidden_dim // 2, layer_dim, bidirectional=True)
        else:
            self.lstm = nn.LSTM(input_dim, hidden_dim // 2, layer_dim, bidirectional=True)

        # Readout layer
        classifier_layers = []
        for _ in range(readout_dim):
            classifier_layers.append(nn.Linear(hidden_dim, hidden_dim))
            classifier_layers.append(nn.ReLU())
        classifier_layers.append(nn.Dropout(drop_out))
        classifier_layers.append(nn.Linear(hidden_dim, output_dim))
        self.classifier = nn.Sequential(*classifier_layers)

    #Implementation of forward function
    def forward(self, x, yolo_f=torch.tensor([])):
        f = self.cnn(x)
        f = f.view(-1, 1, self.input_dim)
        # The objects positions are concatenated with the feature extracted from the pretrained model
        if len(yolo_f) != 0:
            yolo_f = yolo_f.unsqueeze(1)
            f = torch.cat((f, yolo_f), 2)
        out, (hn, cn) = self.lstm(f)
        out = self.classifier(out[:, -1, :])
        return out.squeeze(1)
