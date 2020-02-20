import json
import sys
import os
import torch
from tqdm import tqdm
sys.path.insert(1, '../../Bi-LSTM')
from Bi_LSTM import Bi_LSTM
from FeatureDataset import FeatureDataset
sys.path.insert(1, '../../Model Utils')
from ModelUtils import ModelUtils

'''This script allows to performs Grid Search on model without fine-tuning'''

#The "grid_search_ini.json" file contains all the initialization parameters.
with open("grid_search_ini.json") as ini_file:
    ini = json.load(ini_file)

for i, match_code in enumerate(ini["matches_code_train"]):
    dataset = FeatureDataset(match_code=match_code, model_type=ini["cnn_model"], seq_dim=ini["seq_dim"])
    if i == 0:
        train_dataset = dataset
    else:
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, dataset])

train_loader = torch.utils.data.DataLoader(dataset=train_dataset ,batch_size=ini["batch_size"], shuffle=True)
#Create Validations Loaders from features extracted by pretrained models (ResNet18 and VGG16)
validations_loaders = []
for match_code in ini["matches_code_val"]:
    val_dataset = FeatureDataset(match_code=match_code, model_type=ini["cnn_model"], seq_dim=ini["seq_dim"])
    validations_loaders.append(torch.utils.data.DataLoader(dataset=val_dataset, batch_size=ini["batch_size"], shuffle=False))
pbar_grid = tqdm(total=8, desc="Try:")
for lr in ini["learning_rate"]:
    for drop in ini["drop_out"]:
        folder_name = "Model" + "_" + str(lr) + "_" + str(drop) + "_" + str(ini["seq_dim"])
        if not os.path.exists( "Models/" + folder_name ):
            os.makedirs( "Models/" + folder_name, mode=0o0775 )
        if not os.path.exists( "Models/" + folder_name + "/Model Steps" ):
            os.makedirs( "Models/" + folder_name + "/Model Steps", mode=0o0775 )
        #Initialize LSTM Model
        model = Bi_LSTM(ini['input_dim'], ini['hidden_dim'], ini['layer_dim'], ini["output_dim"], ini["readout_dim"], drop, graphics_card=ini["device"])
        model_utils = ModelUtils(model = model, input_dim=ini['input_dim'], seq_dim=ini["seq_dim"], model_type="LSTM", learning_rate=lr, momentum=0.9, device_name=ini["device"], optimizer=ini["optimizer"], criterion="BCEWithLogitsLoss")
        metrics = model_utils.train_model(train_loader=train_loader, val_loaders=validations_loaders, epochs=ini["epochs"], metrics={}, save_epoch_model=True, start_epoch=0, eval_epochs=2, save_epochs_steps=1, path_model_save="Models/"+folder_name +"/Model Steps/")
        with open("Models/" + folder_name + "/model_metrics.json", "w") as metrics_file:
            json.dump(metrics, metrics_file)
        pbar_grid.update(1)
pbar_grid.close()
