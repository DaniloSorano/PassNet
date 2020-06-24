import cv2
import math
from tqdm import tqdm
import pickle as pk
import numpy as np
import json

class csvToEventsMatrix:
    def __init__(self, match_code="sassuolo_inter_1", pass_type="all"):
        self.pass_type = pass_type
        self.match = match_code
        self.events_pass_match = []

        if self.match == "sassuolo_inter_1":
            matchId = 2576136
            period = "1H"
        elif self.match == "sassuolo_inter_2":
            matchId = 2576136
            period = "2H"
        elif self.match == "roma_juve_1":
            matchId = 2576322
            period = "1H"
        elif self.match == "roma_juve_2":
            matchId = 2576322
            period = "2H"
        elif self.match == "chievo_juve_1":
            matchId = 2759433
            period = "1H"
        elif self.match == "chievo_juve_2":
            matchId = 2759433
            period = "2H"
        elif self.match == "roma_lazio_1":
            matchId = 2576084
            period = "1H"
        elif self.match == "roma_lazio_2":
            matchId = 2576084
            period = "2H"

        with open("Events_Data/events_Italy.json", encoding='utf-8') as file_events:
            data_events = json.load(file_events)
        with open("Events_Data/chievo-juve-2018-08-18.json", encoding='utf-8') as file_events:
            chievo_juve = json.load(file_events)
            data_events += chievo_juve['events']

        self.type_pass_dict = set()
        if self.pass_type != "all":
            for ev in data_events:
                if ev["subEventName"] == self.pass_type and ev["matchId"] == matchId  and ev["matchPeriod"] == period:
                    self.type_pass_dict.add(ev["id"])
        else:
            for ev in data_events:
                if ev["eventName"] == "Pass" and ev["matchId"] == matchId  and ev["matchPeriod"] == period:
                    self.type_pass_dict.add(ev["id"])

        events_pass_match = []
        with open("Annotations/" + match_code + "_Pass.csv") as file:
            for line in file.readlines():
                events_pass_match.append(line[:len(line)-1].split(";"))
        #Get the number of the frames in the video_match
        video = cv2.VideoCapture("Videos/" + match_code + ".mp4")
        self.total_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
        self.events_pass_match = [*filter( lambda x : int(x[0]) in self.type_pass_dict, events_pass_match[1:] )]

    def targetMatrix(self, real_fps=25, fps_to_extract=5):
        #Build the matrix frame X events
        print("Building target matrix...")
        events_matrix = np.zeros((int(self.total_frame), 2), dtype=int)

        if self.pass_type == "all":
            for row in self.events_pass_match:
                start_frame = math.ceil(float(row[4]) * fps_to_extract)
                end_frame = math.ceil(float(row[5]) * fps_to_extract)
                frame_window = np.array(range(start_frame, end_frame))
                for index, frame in enumerate(frame_window):
                    events_matrix[frame - 1][1] = int(1)

            for i in range(0, len(events_matrix)):
                if events_matrix[i][1] == 0:
                    events_matrix[i][0] = 1

            print ("The target matrix has been created!")
            self.events_matrix = events_matrix
        elif self.pass_type == "Simple pass":
            for row in self.events_pass_match:
                start_frame = math.ceil(float(row[4]) * fps_to_extract)
                end_frame = math.ceil(float(row[5]) * fps_to_extract)
                frame_window = np.array(range(start_frame, end_frame))
                for index, frame in enumerate(frame_window):
                    events_matrix[frame - 1][1] = int(1)

            for i in range(0, len(events_matrix)):
                if events_matrix[i][1] == 0:
                    events_matrix[i][0] = 1

            print ("The target matrix has been created!")
            self.events_matrix = events_matrix
        else:
            print ("The target matrix has not been created!")


    def returnTargetMatrix(self):
        return self.events_matrix

    def saveTargetMatrix(self, path="Outputs/"):
        if self.pass_type != "all":
            pass_type = self.pass_type.split(" ")[0].lower()  + "_" + self.pass_type.split(" ")[1].lower()
            filename = self.match + "_" + pass_type +"_binary.pkl"
        else:
            filename = self.match + "_binary.pkl"
        pk.dump(self.events_matrix, open(path + filename, "wb"), protocol=2)