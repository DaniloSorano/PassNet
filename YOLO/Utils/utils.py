import pickle as pk
from tqdm import tqdm
import math
import numpy as np


def extract_yolo_features(path_name, x_resolution, y_resolution, n_frames):
    #Read YOLO features from file
    with open(path_name,"r") as file:
        frames = file.readline().split("$")

    yolo_features = dict()
    pbar_frames = tqdm(total=len(frames), desc="Frames", position=0)
    for lista in frames:
        objects = lista.split("&")
        for object in objects:
            object_info = object.split(",")
            if len(object_info) >= 2:
                n_frame = int(object_info[0].split("_")[-1])
                if n_frame not in yolo_features.keys():
                    yolo_features[n_frame] = []
                object = []
                if len(object_info) >= 5:
                    if object_info[1] == "sports ball":
                        object.append(0.0)
                    else:
                        object.append(1.0)
                    object.append(1.0)
                    X_n = float(object_info[3])
                    Y_n = float(object_info[4])
                    if X_n >= 0 and Y_n >= 0:
                        if X_n < x_resolution/2:
                            X_n = -(X_n / (x_resolution/2))
                        elif X_n > x_resolution:
                            X_n = X_n /x_resolution
                        else:
                            X_n = 0.0
                        object.append(X_n)
                        if Y_n < y_resolution/2:
                            Y_n = -(Y_n / (y_resolution/2))
                        elif Y_n > y_resolution:
                            Y_n = Y_n /y_resolution
                        else:
                            Y_n = 0.0
                        object.append(Y_n)
                    else:
                        if object_info[1] == "sports ball":
                            object.append(0.0)
                            object.append(0.0)
                        else:
                            object.append(2.0)
                            object.append(2.0)
                yolo_features[n_frame].append(object)
        pbar_frames.update(1)
    pbar_frames.close()


    for frame in list(range(1,n_frames)):
        if frame not in yolo_features.keys():
            yolo_features[frame] = []
            yolo_features[frame].append([0.0, 0, 0.0, 0.0])
            yolo_features[frame].append([1.0, 0.0, 2.0, 2.0])
            yolo_features[frame].append([1.0, 0.0, 2.0, 2.0])
            yolo_features[frame].append([1.0, 0.0, 2.0, 2.0])
            yolo_features[frame].append([1.0, 0.0, 2.0, 2.0])
            yolo_features[frame].append([1.0, 0.0, 2.0, 2.0])
            yolo_features[frame].append([1.0, 0.0, 2.0, 2.0])
            yolo_features[frame].append([1.0, 0.0, 2.0, 2.0])
            yolo_features[frame].append([1.0, 0.0, 2.0, 2.0])
            yolo_features[frame].append([1.0, 0.0, 2.0, 2.0])
            yolo_features[frame].append([1.0, 0.0, 2.0, 2.0])
        else:
            sports_ball_exist = False
            for objs in yolo_features[frame]:
                if objs[0] == "sports ball":
                    sports_ball_exist = True
            if sports_ball_exist == False:
                yolo_features[frame].append([0.0, 0.0, 0.0, 0.0])
            if len(yolo_features[frame]) < 11:
                new_el_count = 11 - len(yolo_features[frame])
                for new_objs in range(0, new_el_count):
                    yolo_features[frame].append([1.0, 0.0, 2.0, 2.0])

    return yolo_features

def nn_ball(objects_detected, n_players=5):
    x_ref = 0.0
    y_ref = 0.0
    ball_idx = 0
    for i, obj in enumerate(objects_detected):
        if obj[0] == 0:
            x_ref = obj[2]
            y_ref = obj[3]
            ball_idx = i

    distance_dict = dict()
    for j, obj in enumerate(objects_detected):
        if obj[0] == 1:
            if obj[1] == 1:
                distance = math.sqrt((x_ref - obj[2])**2 + (y_ref - obj[3])**2)
                distance_dict[j] = distance
            else:
                distance_dict[j] = 1000
    sorted_players = sorted(distance_dict, key = lambda x : distance_dict[x], reverse=False)
    objs = []
    for el_ball in objects_detected[ball_idx]:
        objs.append(el_ball)
    for idx in sorted_players[:n_players]:
        for el_player in objects_detected[idx]:
            objs.append(el_player)
    return np.array(objs)

"""import torch
lista = extract_yolo_features("../../Data/Input/bbox/roma_juve_2.txt", 1280, 720, 14110)
print(nn_ball(lista[14037], n_players=5))"""
