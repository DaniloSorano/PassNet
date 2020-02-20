import numpy as np
import pickle as pk
import random
from sklearn.metrics import confusion_matrix

#The class "DummyClassifier" implements all the baseline classifiers compared to the models
class DummyClassifier():
    def __init__(self, match_codes, data_len, path = "../Data/Outputs/Binary Outputs/"):
        self.path = path
        self.metrics = {}
        self.labels = {}
        for match_code in match_codes:
            labels_match = pk.load(open(path + match_code + "_binary.pkl","rb"))[:data_len]
            labels_match = [*map( lambda x: 0 if x[0]== 1 else 1, labels_match )]
            self.labels[match_code] = labels_match

    # This function returns the predicted labels of the baseline classifier
    def get_labels(self):
        return self.labels

    # This function returns the metrics of the baseline classifier
    def get_metrics(self):
        return self.metrics

    '''
    This function implements the Most Frequent Classifier, so predicts always the most frequent label,
    in this case the "No Pass".
    '''
    def most_frequent_classifier(self):
        for match_key in self.labels.keys():
            most_frequent_class = max(set(self.labels[match_key]), key = self.labels[match_key].count)
            pred_match = [*map(lambda x : most_frequent_class, self.labels[match_key])]
            conf_matrix_values = confusion_matrix(self.labels[match_key], pred_match).ravel()
            tn = conf_matrix_values[0]
            fp = conf_matrix_values[1]
            fn = conf_matrix_values[2]
            tp = conf_matrix_values[3]
            self.metrics[match_key] = {}
            self.metrics[match_key]["accuracy"] = (tp + tn) * 100 / (tp + tn + fn + fp)
            self.metrics[match_key]["f1_score"] = 2 * tp * 100 / (2 * tp + fp + fn)
            if tp + fp == 0:
                self.metrics[match_key]["precision_pass"] = 0
            else:
                self.metrics[match_key]["precision_pass"] = tp * 100 / (tp + fp)
            if tn + fn == 0:
                self.metrics[match_key]["precision_no_pass"] = 0
            else:
                self.metrics[match_key]["precision_no_pass"] = tn * 100 / (tn + fn)
            if tp + fn == 0:
                self.metrics[match_key]["recall_pass"] = 0
            else:
                self.metrics[match_key]["recall_pass"] = tp * 100 / (tp + fn)
            if tn + fp == 0:
                self.metrics[match_key]["recall_no_pass"] = 0
            else:
                self.metrics[match_key]["recall_no_pass"] = tn * 100 / (tn + fp)

    '''
    This function implements the Least Frequent Classifier, so predicts always the least frequent label,
    in this case the "Pass".
    '''
    def less_frequent_classifier(self):
        for match_key in self.labels.keys():
            less_frequent_class = min(set(self.labels[match_key]), key = self.labels[match_key].count)
            pred_match = [*map(lambda x : less_frequent_class, self.labels[match_key])]
            conf_matrix_values = confusion_matrix(self.labels[match_key], pred_match).ravel()
            tn = conf_matrix_values[0]
            fp = conf_matrix_values[1]
            fn = conf_matrix_values[2]
            tp = conf_matrix_values[3]
            self.metrics[match_key] = {}
            self.metrics[match_key]["accuracy"] = (tp + tn) * 100 / (tp + tn + fn + fp)
            self.metrics[match_key]["f1_score"] = 2 * tp * 100 / (2 * tp + fp + fn)
            if tp + fp == 0:
                self.metrics[match_key]["precision_pass"] = 0
            else:
                self.metrics[match_key]["precision_pass"] = tp * 100 / (tp + fp)
            if tn + fn == 0:
                self.metrics[match_key]["precision_no_pass"] = 0
            else:
                self.metrics[match_key]["precision_no_pass"] = tn * 100 / (tn + fn)
            if tp + fn == 0:
                self.metrics[match_key]["recall_pass"] = 0
            else:
                self.metrics[match_key]["recall_pass"] = tp * 100 / (tp + fn)
            if tn + fp == 0:
                self.metrics[match_key]["recall_no_pass"] = 0
            else:
                self.metrics[match_key]["recall_no_pass"] = tn * 100 / (tn + fp)

    '''
    This function implements the Random Classifier, so predicts randomly the two labels.
    '''
    def random_classifier(self):
        for match_key in self.labels.keys():
            pred_match = [*map(lambda x : random.randrange(0,2), self.labels[match_key])]
            conf_matrix_values = confusion_matrix(self.labels[match_key], pred_match).ravel()
            tn = conf_matrix_values[0]
            fp = conf_matrix_values[1]
            fn = conf_matrix_values[2]
            tp = conf_matrix_values[3]
            self.metrics[match_key] = {}
            self.metrics[match_key]["accuracy"] = (tp + tn) * 100 / (tp + tn + fn + fp)
            self.metrics[match_key]["f1_score"] = 2 * tp * 100 / (2 * tp + fp + fn)
            if tp + fp == 0:
                self.metrics[match_key]["precision_pass"] = 0
            else:
                self.metrics[match_key]["precision_pass"] = tp * 100 / (tp + fp)
            if tn + fn == 0:
                self.metrics[match_key]["precision_no_pass"] = 0
            else:
                self.metrics[match_key]["precision_no_pass"] = tn * 100 / (tn + fn)
            if tp + fn == 0:
                self.metrics[match_key]["recall_pass"] = 0
            else:
                self.metrics[match_key]["recall_pass"] = tp * 100 / (tp + fn)
            if tn + fp == 0:
                self.metrics[match_key]["recall_no_pass"] = 0
            else:
                self.metrics[match_key]["recall_no_pass"] = tn * 100 / (tn + fp)

if __name__ == "__main__":
    dummy = DummyClassifier(["roma_juve_2","chievo_juve_1"], 5000)
    print(dummy.random_classifier())
