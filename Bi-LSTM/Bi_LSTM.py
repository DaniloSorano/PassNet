import torch
import torch.nn as nn
'''
Class that implements the CNN pre-trained model without fine-tuning.
'''
class Bi_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, readout_dim, drop_out, graphics_card=0):
        super(Bi_LSTM, self).__init__()
        self.device = torch.device(graphics_card if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.layer_dim = layer_dim

        # The class instantiates the LSTM class by setting the bidirectional paramters equal TRUE
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2 , layer_dim, bidirectional = True)

        # Readout layer
        classifier_layers = []
        for _ in range(readout_dim):
            classifier_layers.append(nn.Linear(hidden_dim, hidden_dim))
            classifier_layers.append(nn.ReLU())
        classifier_layers.append(nn.Dropout(drop_out))
        classifier_layers.append(nn.Linear(hidden_dim, output_dim))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        x = x.view(-1, 1, self.input_dim)
        out, (hn, cn) = self.lstm(x)
        out = self.classifier(out[:, -1, :])
        return out.squeeze(1)
