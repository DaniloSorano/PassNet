import json
import sys
import os
import torch
sys.path.insert(1, '../../CNN+Bi-LSTM')
from TensorDataset import TensorDataset
from Cnn_Lstm import Cnn_Lstm
sys.path.insert(1, '../../Bi-LSTM')
from Bi_LSTM import Bi_LSTM
from FeatureDataset import FeatureDataset
sys.path.insert(1, '../../Model Utils')
from ModelUtils import ModelUtils

'''
    In this script the models (with or without fine-tuning) is trained.
'''

# The script reads the initialization parameters from "train_ini.json" file.
with open("Model Parameters/train_ini.json") as ini_file:
    ini = json.load(ini_file)

if ini["model_type"] == "CNN+LSTM":
    '''
    The script inside this 'if' statement create the training and validation
    datasets by using the class "TensorDataset" and "DataLoader" for the model
    with fine-tuning
    '''
    for i, match_code in enumerate(ini["matches_code_train"]):
        dataset = TensorDataset(match_code=match_code, seq_dim=ini["seq_dim"], reduce_data_el=None, yolo=ini["yolo"])
        if i == 0:
            train_dataset = dataset
        else:
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, dataset])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=ini["batch_size"], shuffle=True)
    validations_loaders = []
    for match_code in ini["matches_code_val"]:
        val_dataset = TensorDataset(match_code=match_code, seq_dim=ini["seq_dim"], reduce_data_el=None,  yolo=ini["yolo"])
        validations_loaders.append(torch.utils.data.DataLoader(dataset=val_dataset, batch_size=ini["batch_size"], shuffle=False))

    yolo_dimension = 0
    if ini["yolo"]:
        yolo_dimension = ini["yolo_dimension"]
    # Here the script instantiate the class "Cnn_LSTM" for the model with fine-tuning
    model = Cnn_Lstm(input_dim=ini['input_dim'], yolo_dimension=yolo_dimension, hidden_dim=ini['hidden_dim'], layer_dim=ini['layer_dim'],
                output_dim=ini["output_dim"], readout_dim=ini["readout_dim"], drop_out=ini["drop_out"], device=ini["device"], cnn_model=ini["cnn_model"])

elif ini["model_type"] == "LSTM":

    '''
    The script inside this 'elif' statement create the training and validation
    datasets by using the class "FeatureDataset" and "DataLoader" for the model
    without fine-tuning
    '''
    for i, match_code in enumerate(ini["matches_code_train"]):
        dataset = FeatureDataset(match_code=match_code, model_type="vgg16", seq_dim=ini["seq_dim"])
        if i == 0:
            train_dataset = dataset
        else:
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, dataset])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset ,batch_size=ini["batch_size"], shuffle=ini["shuffle"])
    validations_loaders = []
    for match_code in ini["matches_code_val"]:
        val_dataset = FeatureDataset(match_code=match_code, model_type="vgg16", seq_dim=ini["seq_dim"])
        validations_loaders.append(torch.utils.data.DataLoader(dataset=val_dataset, batch_size=ini["batch_size"], shuffle=False))
    # Here the script instantiate the class "Bi_LSTM" for the model without fine-tuning
    model = Bi_LSTM(ini['input_dim'], ini['hidden_dim'], ini['layer_dim'], ini["output_dim"], ini["readout_dim"], graphics_card=ini["device"])

models_steps_list = os.listdir("Model Steps")
first_epochs=0
metrics = {}

'''
Here the model load the old state if the folder model steps contains
some models saved and if the "load_last_model" is equal True
'''
if len(models_steps_list) != 0 and ini["load_last_model"]:
    model.load_state_dict(torch.load("Model Steps/" + models_steps_list[len(models_steps_list)-1]))
    first_epochs = int(models_steps_list[len(models_steps_list)-1].split("_")[1].split(".")[0]) + 1
    with open("Model Metrics/model_metrics.json") as metrics_file:
        metrics = json.load(metrics_file)

# The script use the class ModelUtils that implements the function to train the model
model_utils = ModelUtils(model = model, input_dim=ini['input_dim'], seq_dim=ini['seq_dim'], model_type=ini["model_type"], learning_rate=ini["learning_rate"], momentum=ini["momentum"], device_name=ini["device"], optimizer=ini["optimizer"], criterion="BCEWithLogitsLoss")
metrics = model_utils.train_model(train_loader=train_loader, val_loaders=validations_loaders, epochs=ini["epochs"], metrics=metrics, save_epoch_model=True, start_epoch=first_epochs, eval_epochs=ini["eval_epochs"], save_epochs_steps=1, yolo=ini["yolo"])
# When the training finish the metrics obtained on each steps are saved into a json file.
with open("Model Metrics/model_metrics.json", "w") as metrics_file:
    json.dump(metrics, metrics_file)
