import os
import pdb
import torch
import json
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
import pickle as pk

'''
The "ModelUtils" class implements the function to training and test the models with and without fine-tuning
 '''
class ModelUtils():
    def __init__(self, model, seq_dim, input_dim, model_type, learning_rate, momentum, device_name, optimizer="SGD", criterion="BCEWithLogitsLoss"):
        self.model = model
        self.seq_dim = seq_dim
        self.input_dim = input_dim
        self.model_type = model_type
        # Optimizer types
        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
        elif optimizer == "SGD-Nesterov":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)
        elif optimizer == "ADAM":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        ## Loss types
        if criterion == "BCEWithLogitsLoss":
            self.criterion = nn.BCEWithLogitsLoss()
        elif optimizer == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss()
        elif optimizer == "MSELoss":
            self.criterion = nn.MSELoss()
        self.device = torch.device(device_name)
        self.model.to(self.device)

    # This function iterates the single epoch on training and validation set and computes the metrics
    def epoch_iterator(self, data_loader, type="Validation", pos=0, desc="Batch", yolo=False):
        tn = 0
        tp = 0
        fn = 0
        fp = 0
        running_loss = 0.0
        batches_metrics = {
            "confusion_matrix" : [0, 0, 0, 0],
            "loss" :  0.0,
            "mean_average_precision" : 0.0,
            "accuracy" : 0.0,
            "precision_pass" : 0.0,
            "precision_no_pass" : 0.0,
            "recall_pass" : 0.0,
            "recall_no_pass" : 0.0,
            "f1_score" : 0.0,
        }
        all_labels = []
        all_predictions = []
        batches_metrics["dataset_type"] = type
        self.pbar_batch = tqdm(total=len(data_loader), desc=desc, position=pos)
        if yolo:
            for i, (images, labels, yolo_features) in enumerate(data_loader):
                # Load images as torch tensor with gradient accumulation abilities
                if type == "Train":
                    images = images[0].type('torch.FloatTensor').to(self.device).requires_grad_()
                else:
                    images = images[0].type('torch.FloatTensor').to(self.device)
                labels = labels[0].type('torch.FloatTensor').to(self.device)
                yolo_features = yolo_features[0].type('torch.FloatTensor').to(self.device)
                if type == "Train":
                    self.optimizer.zero_grad()
                # the model get in input the tensor sequence and the object position vector
                outputs = self.model(images, yolo_features)
                _, lab = torch.max(labels, 1)
                # Computation of the Loss
                loss = self.criterion(outputs.cpu(), lab.type('torch.FloatTensor').cpu())
                running_loss += float(loss.item())
                # Updating parameters
                if type == "Train":
                    # Getting gradients w.r.t. parameters
                    loss.backward()
                    self.optimizer.step()

                # The output passes through a Sigmoid layer, now the output value are between the range 0 and 1
                sig = nn.Sigmoid()
                out_sig = sig(outputs.data)

                for j, sig_value in enumerate(out_sig):
                    all_labels.append(lab[j].item())
                    all_predictions.append(sig_value)

                pred_threshold = []
                for value in out_sig:
                    if value >= 0.5:
                        pred_threshold.append(1)
                    else:
                        pred_threshold.append(0)

                # the function "confusion_matrix" compute the Confusion Matrix that return the values: TP, TN, FP and FN
                conf_matrix_values = confusion_matrix(list(lab.cpu().numpy()), pred_threshold, labels=[0,1]).ravel()
                # The function "epoch_iterator" aggregates the confusion matrix values at each step
                tn += conf_matrix_values[0]
                fp += conf_matrix_values[1]
                fn += conf_matrix_values[2]
                tp += conf_matrix_values[3]
                self.pbar_batch.update(1)
        else:
            # Here we apply the same stuff of the "if" statemet but in this case
            # we handle only the models without objects position vector
            for i, (images, labels) in enumerate(data_loader):
                if self.model_type == "LSTM":
                    if type == "Train":
                        images = images[0].type('torch.FloatTensor').to(self.device).requires_grad_()
                    else:
                        images = images[0].type('torch.FloatTensor').to(self.device)
                    labels = labels[0].type('torch.FloatTensor').to(self.device)
                elif self.model_type == "CNN+LSTM":
                    if type == "Train":
                        images = images[0].type('torch.FloatTensor').to(self.device).requires_grad_()
                    else:
                        images = images[0].type('torch.FloatTensor').to(self.device)
                    labels = labels[0].type('torch.FloatTensor').to(self.device)
                if type == "Train":
                    self.optimizer.zero_grad()
                outputs = self.model(images)
                _, lab = torch.max(labels, 1)
                loss = self.criterion(outputs.cpu(), lab.type('torch.FloatTensor').cpu())
                running_loss += float(loss.item())
                # Updating parameters
                if type == "Train":
                    # Getting gradients w.r.t. parameters
                    loss.backward()
                    self.optimizer.step()

                sig = nn.Sigmoid()
                out_sig = sig(outputs.data)

                for j, sig_value in enumerate(out_sig):
                    all_labels.append(lab[j].item())
                    all_predictions.append(sig_value)

                pred_threshold = []
                for value in out_sig:
                    if value >= 0.5:
                        pred_threshold.append(1)
                    else:
                        pred_threshold.append(0)

                conf_matrix_values = confusion_matrix(list(lab.cpu().numpy()), pred_threshold, labels=[0,1]).ravel()
                tn += conf_matrix_values[0]
                fp += conf_matrix_values[1]
                fn += conf_matrix_values[2]
                tp += conf_matrix_values[3]
                self.pbar_batch.update(1)
        self.pbar_batch.close()

        batches_metrics["confusion_matrix"] =  [ int(tn), int(fp), int(fn), int(tp) ]
        batches_metrics["loss"] = running_loss / len(data_loader)
        # Computation of the metrics
        mean_average_precision = average_precision_score(all_labels, all_predictions)
        batches_metrics["mean_average_precision"] = mean_average_precision
        batches_metrics["accuracy"] = (tp + tn) * 100 / (tp + tn + fn + fp)
        if tp + fp == 0:
            batches_metrics["precision_pass"] = 0.0
        else:
            batches_metrics["precision_pass"] = tp *100/ (tp + fp)
        if tn + fn == 0:
            batches_metrics["precision_no_pass"] = 0.0
        else:
            batches_metrics["precision_no_pass"] = tn * 100 / (tn + fn)
        if tp + fn == 0:
            batches_metrics["recall_pass"] = 0.0
        else:
            batches_metrics["recall_pass"] = tp * 100/ (tp + fn)
        if tn + fp == 0:
            batches_metrics["recall_no_pass"] = 0.0
        else:
            batches_metrics["recall_no_pass"] = tn * 100 / (tn + fp)
        if 2 * tp + fp + fn == 0:
            batches_metrics["f1_score"] = 0
        else:
            batches_metrics["f1_score"] = (2 * tp) * 100 / ((2 * tp) + fp + fn)

        return batches_metrics



    ## This  Function returns the models prediction and corresponding labels
    def get_model_prediction(self, data_loader, pos=0, desc="Model Prediction", yolo=False, threshold=0.5):
        all_predictions = [0] * self.seq_dim
        all_labels = [0] * self.seq_dim
        start_idx = 0
        end_idx = self.seq_dim
        dict_idx = dict()
        for idx in range(0,self.seq_dim):
            dict_idx[idx] = 0

        self.pbar_batch= tqdm(total=len(data_loader), desc=desc, position=pos)
        if yolo:
            for i, (images, labels, yolo_features) in enumerate(data_loader):
                images = images[0].type('torch.FloatTensor').to(self.device)
                labels = labels[0].type('torch.FloatTensor').to(self.device)
                yolo_features = yolo_features[0].type('torch.FloatTensor').to(self.device)
                outputs = self.model(images, yolo_features)
                _, lab = torch.max(labels, 1)

                sig = nn.Sigmoid()
                out_sig = sig(outputs.data)
                for idx, index in enumerate(list(range(start_idx, end_idx))):
                    all_predictions[index] += out_sig[idx].item()
                    all_labels[index] += lab[idx].item()
                    dict_idx[index] += 1
                start_idx += 1
                end_idx += 1
                dict_idx[len(all_predictions)] = 0
                all_predictions.append(0)
                all_labels.append(0)
                self.pbar_batch.update(1)
        else:
            for i, (images, labels) in enumerate(data_loader):
                images = images[0].type('torch.FloatTensor').to(self.device)
                labels = labels[0].type('torch.FloatTensor').to(self.device)
                outputs = self.model(images)
                _, lab = torch.max(labels, 1)

                sig = nn.Sigmoid()
                out_sig = sig(outputs.data)
                for idx, index in enumerate(list(range(start_idx, end_idx))):
                    all_predictions[index] += out_sig[idx].item()
                    all_labels[index] += lab[idx].item()
                    dict_idx[index] += 1
                start_idx += 1
                end_idx += 1
                dict_idx[len(all_predictions)] = 0
                all_predictions.append(0)
                all_labels.append(0)
                self.pbar_batch.update(1)
        self.pbar_batch.close()
        all_predictions.pop(-1)
        all_labels.pop(-1)
        sig_pred = []
        for i, el in enumerate(all_predictions):
            if dict_idx[i] != 0:
                sig_pred.append(el/(dict_idx[i]))
                if (el/(dict_idx[i])) >= threshold:
                    all_predictions[i] = 1
                else:
                    all_predictions[i] = 0
                all_labels[i] = all_labels[i]/dict_idx[i]
            else:
                sig_pred.append(el)
        return sig_pred, all_labels, all_predictions

    ## The "epoch_iterator_val" execute iteraton over single epoch of validation set
    ## In this function also the metrics but differently from the function "epoch_iterator"
    def epoch_iterator_val(self, data_loader, pos=0, desc="Model Prediction", yolo=False, threshold=0.5):
        all_predictions = [0] * self.seq_dim
        all_labels = [0] * self.seq_dim
        start_idx = 0
        end_idx = self.seq_dim
        dict_idx = dict()
        for idx in range(0,self.seq_dim):
            dict_idx[idx] = 0
        running_loss = 0.0
        batches_metrics = {
            "confusion_matrix" : [0, 0, 0, 0],
            "loss" :  0.0,
            "mean_average_precision" : 0.0,
            "accuracy" : 0.0,
            "precision_pass" : 0.0,
            "precision_no_pass" : 0.0,
            "recall_pass" : 0.0,
            "recall_no_pass" : 0.0,
            "f1_score" : 0.0,
        }

        self.pbar_batch= tqdm(total=len(data_loader), desc=desc, position=pos)
        if yolo:
            for i, (images, labels, yolo_features) in enumerate(data_loader):
                images = images[0].type('torch.FloatTensor').to(self.device)
                labels = labels[0].type('torch.FloatTensor').to(self.device)
                yolo_features = yolo_features[0].type('torch.FloatTensor').to(self.device)
                outputs = self.model(images, yolo_features)
                _, lab = torch.max(labels, 1)
                loss = self.criterion(outputs.cpu(), lab.type('torch.FloatTensor').cpu())
                running_loss += float(loss.item())
                sig = nn.Sigmoid()
                out_sig = sig(outputs.data)
                for idx, index in enumerate(list(range(start_idx, end_idx))):
                    all_predictions[index] += out_sig[idx].item()
                    all_labels[index] += lab[idx].item()
                    dict_idx[index] += 1
                start_idx += 1
                end_idx += 1
                dict_idx[len(all_predictions)] = 0
                all_predictions.append(0)
                all_labels.append(0)
                self.pbar_batch.update(1)
        else:
            for i, (images, labels) in enumerate(data_loader):
                images = images[0].type('torch.FloatTensor').to(self.device)
                labels = labels[0].type('torch.FloatTensor').to(self.device)
                outputs = self.model(images)
                _, lab = torch.max(labels, 1)
                loss = self.criterion(outputs.cpu(), lab.type('torch.FloatTensor').cpu())
                running_loss += float(loss.item())
                sig = nn.Sigmoid()
                out_sig = sig(outputs.data)
                for idx, index in enumerate(list(range(start_idx, end_idx))):
                    all_predictions[index] += out_sig[idx].item()
                    all_labels[index] += lab[idx].item()
                    dict_idx[index] += 1
                start_idx += 1
                end_idx += 1
                dict_idx[len(all_predictions)] = 0
                all_predictions.append(0)
                all_labels.append(0)
                self.pbar_batch.update(1)
        self.pbar_batch.close()
        all_predictions.pop(-1)
        all_labels.pop(-1)
        sig_pred = []
        # We compute the average of single elements of the predictions and then we apply a predefined threshols
        for i, el in enumerate(all_predictions):
            if dict_idx[i] != 0:
                sig_pred.append(el/(dict_idx[i]))
                if (el/(dict_idx[i])) >= threshold:
                    all_predictions[i] = 1
                else:
                    all_predictions[i] = 0
                all_labels[i] = all_labels[i]/dict_idx[i]
            else:
                sig_pred.append(el)
        conf_matrix_values = confusion_matrix(all_labels, all_predictions, labels=[0,1]).ravel()
        tn = conf_matrix_values[0]
        fp = conf_matrix_values[1]
        fn = conf_matrix_values[2]
        tp = conf_matrix_values[3]

        batches_metrics["confusion_matrix"] =  [ int(tn), int(fp), int(fn), int(tp) ]
        batches_metrics["loss"] = running_loss / len(data_loader)
        mean_average_precision = average_precision_score(all_labels, sig_pred)
        batches_metrics["mean_average_precision"] = mean_average_precision
        batches_metrics["accuracy"] = (tp + tn) * 100 / (tp + tn + fn + fp)
        if tp + fp == 0:
            batches_metrics["precision_pass"] = 0.0
        else:
            batches_metrics["precision_pass"] = tp * 100/ (tp + fp)
        if tn + fn == 0:
            batches_metrics["precision_no_pass"] = 0.0
        else:
            batches_metrics["precision_no_pass"] = tn * 100/ (tn + fn)
        if tp + fn == 0:
            batches_metrics["recall_pass"] = 0.0
        else:
            batches_metrics["recall_pass"] = tp * 100/ (tp + fn)
        if tn + fp == 0:
            batches_metrics["recall_no_pass"] = 0.0
        else:
            batches_metrics["recall_no_pass"] = tn * 100 / (tn + fp)
        if 2 * tp + fp + fn == 0:
            batches_metrics["f1_score"] = 0.0
        else:
            batches_metrics["f1_score"] = (2 * tp) * 100  / ((2 * tp) + fp + fn)

        return sig_pred, all_labels, all_predictions, batches_metrics

    ## This Function performs the training of the models and return the metrics computed at each epoch
    def train_model(self, train_loader, val_loaders, epochs, metrics={}, save_epoch_model=False, start_epoch=1, eval_epochs=5, save_epochs_steps=1, yolo=False, path_model_save="Model Steps/"):
        if metrics == {}:
            metrics = {
                "confusion_matrix" :  [],
                "loss" :  [],
                "mean_average_precision" : [],
                "accuracy" : [],
                "precision_pass" : [],
                "precision_no_pass" : [],
                "recall_pass" : [],
                "recall_no_pass" : [],
                "f1_score" : [],
                "dataset_type" : []
            }
        epochs_list = list(range(start_epoch,epochs+1))
        eval_list_epochs = []
        current_epoch = epochs
        eval_list_epochs.append(current_epoch)
        while current_epoch != eval_epochs:
            current_epoch -= eval_epochs
            eval_list_epochs.append(current_epoch)

        save_steps_list = [start_epoch]
        current_epoch = start_epoch
        current_epoch += save_epochs_steps
        while current_epoch <= epochs:
            save_steps_list.append(current_epoch)
            current_epoch += save_epochs_steps

        pbar_epoch= tqdm(total=len(epochs_list), desc="Epoch", position=1)
        pbar_epoch.update(len(list(range(0, start_epoch))))
        '''
        The "for" loop for the epochs call the function "epoch_iterator" for the training set at each epochs.
        Moreover, at each steps defined through the function parameter "save_epochs_steps" the function "epoch_iterator_val"
        is called for the test set.
        '''
        for epoch in epochs_list:
            if epoch == 0:
                self.model.train()
                metrics_train = self.epoch_iterator(train_loader, type="Validation", pos=2, desc="Train Batch", yolo=yolo)
                metrics["confusion_matrix"].append( metrics_train["confusion_matrix"] )
                metrics["loss"].append(metrics_train["loss"])
                metrics["mean_average_precision"].append(metrics_train["mean_average_precision"])
                metrics["accuracy"].append(metrics_train["accuracy"])
                metrics["precision_pass"].append(metrics_train["precision_pass"])
                metrics["precision_no_pass"].append(metrics_train["precision_no_pass"])
                metrics["recall_pass"].append(metrics_train["recall_pass"])
                metrics["recall_no_pass"].append(metrics_train["recall_no_pass"])
                metrics["f1_score"].append(metrics_train["f1_score"])
                metrics["dataset_type"].append("Train")
                if save_epoch_model and (epoch in save_steps_list):
                    torch.save(self.model.state_dict(), path_model_save + "model_" + str(epoch) + ".pth")
                for i, val_loader in enumerate(val_loaders):
                    self.model.eval()
                    sig_pred, all_labels, all_predictions, metrics_val = self.epoch_iterator_val(val_loader, pos=2, desc="Validation " + str(i), yolo=yolo, threshold=0.5)
                    metrics["confusion_matrix"].append( metrics_val["confusion_matrix"] )
                    metrics["loss"].append(metrics_val["loss"])
                    metrics["mean_average_precision"].append(metrics_val["mean_average_precision"])
                    metrics["accuracy"].append(metrics_val["accuracy"])
                    metrics["precision_pass"].append(metrics_val["precision_pass"])
                    metrics["precision_no_pass"].append(metrics_val["precision_no_pass"])
                    metrics["recall_pass"].append(metrics_val["recall_pass"])
                    metrics["recall_no_pass"].append(metrics_val["recall_no_pass"])
                    metrics["f1_score"].append(metrics_val["f1_score"])
                    metrics["dataset_type"].append("Validation " + str(i))
                    pk.dump(all_predictions, open(path_model_save + "model_" + str(epoch) + "_val_" + str(i) + "_pred.pkl","wb"))
                    pk.dump(all_labels, open(path_model_save + "model_" + str(epoch) + "_val_" + str(i) + "_lab.pkl","wb"))
                    pk.dump(sig_pred, open(path_model_save + "model_" + str(epoch) + "_val_" + str(i) + "_sig.pkl","wb"))
            else:
                self.model.train()
                metrics_train = self.epoch_iterator(train_loader, type="Train", pos=2, desc="Train Batch", yolo=yolo)
                metrics["confusion_matrix"].append( metrics_train["confusion_matrix"] )
                metrics["loss"].append(metrics_train["loss"])
                metrics["mean_average_precision"].append(metrics_train["mean_average_precision"])
                metrics["accuracy"].append(metrics_train["accuracy"])
                metrics["precision_pass"].append(metrics_train["precision_pass"])
                metrics["precision_no_pass"].append(metrics_train["precision_no_pass"])
                metrics["recall_pass"].append(metrics_train["recall_pass"])
                metrics["recall_no_pass"].append(metrics_train["recall_no_pass"])
                metrics["f1_score"].append(metrics_train["f1_score"])
                metrics["dataset_type"].append(metrics_train["dataset_type"])
                if save_epoch_model and (epoch in save_steps_list):
                    torch.save(self.model.state_dict(), path_model_save+"model_" + str(epoch) + ".pth")
                if epoch in eval_list_epochs:
                    for i, val_loader in enumerate(val_loaders):
                        self.model.eval()
                        sig_pred, all_labels, all_predictions, metrics_val = self.epoch_iterator_val(val_loader, pos=2, desc="Validation " + str(i), yolo=yolo, threshold=0.5)
                        metrics["confusion_matrix"].append( metrics_val["confusion_matrix"] )
                        metrics["loss"].append(metrics_val["loss"])
                        metrics["mean_average_precision"].append(metrics_val["mean_average_precision"])
                        metrics["accuracy"].append(metrics_val["accuracy"])
                        metrics["precision_pass"].append(metrics_val["precision_pass"])
                        metrics["precision_no_pass"].append(metrics_val["precision_no_pass"])
                        metrics["recall_pass"].append(metrics_val["recall_pass"])
                        metrics["recall_no_pass"].append(metrics_val["recall_no_pass"])
                        metrics["f1_score"].append(metrics_val["f1_score"])
                        metrics["dataset_type"].append("Validation " + str(i))
                        pk.dump(all_predictions, open(path_model_save + "model_" + str(epoch) + "_val_" + str(i) + "_pred.pkl","wb"))
                        pk.dump(all_labels, open(path_model_save + "model_" + str(epoch) + "_val_" + str(i) + "_lab.pkl","wb"))
                        pk.dump(sig_pred, open(path_model_save + "model_" + str(epoch) + "_val_" + str(i) + "_sig.pkl","wb"))
            pbar_epoch.update(1)
        pbar_epoch.close()
        return metrics
