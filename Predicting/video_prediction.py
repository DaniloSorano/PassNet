import sys
import json
import torch
import pickle as pk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
sys.path.insert(1, '../../CNN+Bi-LSTM')
from TensorDataset import TensorDataset
from Cnn_Lstm import Cnn_Lstm
sys.path.insert(1, '../../Model Utils')
from ModelUtils import ModelUtils
sys.path.insert(1, '../../Dummy Classifiers')
from DummyClassifier import DummyClassifier

fig= plt.figure(figsize=(12,9))
plt.rcParams.update({'font.size': 16})

with open("Model Ini/video_lab_ini.json") as ini_file:
    ini = json.load(ini_file)

# We create the TensorDataset for the matches inside the list ini["matches_code_val"]
dataset = TensorDataset(ini["matches_predictions"], ini["seq_dim"], reduce_data_el=None, yolo=ini["yolo"])
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=ini["batch_size"], shuffle=ini["shuffle"])
cnn_lstm_model = Cnn_Lstm(input_dim=ini['input_dim'], hidden_dim=ini['hidden_dim'], layer_dim=ini['layer_dim'],
            output_dim=ini["output_dim"], drop_out=0.5, readout_dim=ini["readout_dim"], device=ini["device"], cnn_model=ini["cnn_bi_lstm_model"], yolo_dimension=ini["yolo_dimension"])

cnn_lstm_model.load_state_dict(torch.load("Model State/model_" + str(ini["model_to_load"]) + ".pth"))
cnn_lstm_model.eval()
model_utils = ModelUtils(model = cnn_lstm_model, seq_dim=ini["seq_dim"], input_dim=ini["input_dim"], model_type="CNN+LSTM", learning_rate=ini["learning_rate"], momentum=ini["momentum"], device_name=ini["device"], optimizer=ini["optimizer"], criterion="BCEWithLogitsLoss")
out_sig, labels, predictions = model_utils.get_model_prediction(data_loader, pos=0, desc="Model Prediction", threshold=ini["threshold"], yolo=ini["yolo"])

pk.dump(predictions, open("Predictions/model_" + str(ini["model_to_load"]) + "_prediction.pkl","wb")) #Threshold value equal to 0.5
pk.dump(out_sig, open("Predictions/model_" + str(ini["model_to_load"]) + "sigpred.pkl","wb"))

pred_threshold = []
for value in out_sig:
    if value >= ini["threshold"]:
        pred_threshold.append(1)
    else:
        pred_threshold.append(0)

conf_matrix_values = confusion_matrix(labels, pred_threshold, labels=[0,1]).ravel()
tn = conf_matrix_values[0]
fp = conf_matrix_values[1]
fn = conf_matrix_values[2]
tp = conf_matrix_values[3]

print("--------------- Model Metrics -------------------")
print("accuracy:", (tp + tn) * 100 / (tp + tn + fn + fp))
print("precision_pass:", tp * 100/ (tp + fp))
print("recall_pass:",  tp * 100 / (tp + fn))
print("precision_no_pass:", tn * 100 / (tn + fn))
print("recall_no_pass:",  tn * 100 / (tn + fp))
print("f1_score:", 2 * tp * 100 / (2 * tp + fp + fn))
