import pymongo
import math
import json
from bson import json_util
import numpy as np
import cv2
from tqdm import tqdm
import pickle as pk
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import numpy as np

class jsonToTargetMatrix:

    def __init__(self, file_name, database, collection):

        #Initialize mongoDB connection
        print("Initialize connection to mongoDB")
        self.file_name = file_name
        self.myclient = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
        self.mydb = self.myclient[database]
        self.mycol = ""

        if (collection in self.mydb.list_collection_names()) == False:
            #Get data from json file
            print("Get Json Data from the file...")
            data = ""
            with open(file_name) as f:
                data = json.load(f)
            data = json.dumps(data["events"])
            bson = json_util.loads(data)

            # Insert the json data in mongoDB
            print("Insert the json data in mongoDB...")
            self.mycol = self.mydb[collection]
            success = self.mycol.insert_many(bson)
            print("Collection '" + collection + "' created on '" + database + "' database")
        else:
            self.mycol = self.mydb[collection]
            print("Collection '" + collection + "' already exist in '" + database + "' database")

    def accurate_pass(self, event_tags):
        accurate = False
        tag_ids = set()
        tags = event_tags
        for tag in tags:
            tag_ids.add(tag['id'])
        if 1801 in tag_ids:
            accurate = True
        return accurate

    def gaussian(self, x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def euclidian_distance(self, coordinates):
        start_x = coordinates[0]['x']
        start_y = coordinates[0]['y']
        end_x = coordinates[1]['x']
        end_y = coordinates[1]['y']
        if ((start_x == start_y == 100 or start_x == start_y == 0) or (end_x == end_y == 100 or end_x == end_y == 0)):
            distance = 1000
        else:
            distance = math.sqrt((start_x - end_x)**2 + (start_y - end_y)**2)
        return distance

    def targetMatrix(self, video_match, match="sassuolo_inter", real_fps=25, fps_to_extract=5, period="all", event_type=["Pass", "Foul", "Shot", "Duel", "Free Kick", "Offside"]):
        if match == "sassuolo_inter":
            matchId = 2576136
        elif match == "roma_juve":
            matchId = 2576322
        elif match == "chievo_juve":
            matchId = 2759433

        # Get all sorted events of a match
        match_events = []
        if period == "all":
            match_events = self.mycol.find({"matchId" : matchId,"eventName" : { '$ne' : "Touch" }},{ "eventId" : 1, "eventName" : 1, "eventSec" : 1 , "tags" : 1, "positions" : 1}).sort("eventSec")
        elif period == "first_half":
            match_events = self.mycol.find({"matchId" : matchId, "matchPeriod" : { "$eq" : "1H" },"eventName" : { '$ne' : "Touch" }},{ "eventId" : 1, "eventName" : 1, "eventSec" : 1, "tags" : 1, "positions" : 1 }).sort("eventSec")
        else:
            match_events = self.mycol.find({"matchId" : matchId, "matchPeriod" : { "$eq" : "2H" },"eventName" : { '$ne' : "Touch" }},{ "eventId" : 1, "eventName" : 1, "eventSec" : 1, "tags" : 1, "positions" : 1 }).sort("eventSec")

        match_events = list(match_events)

        #Get the number of the frames in the video_match
        video = cv2.VideoCapture(video_match)
        total_frame = ((video.get(cv2.CAP_PROP_FRAME_COUNT) / real_fps) + 2) * fps_to_extract

        #Build the matrix frame X events
        print("Building target matrix...")
        events_matrix = np.zeros((int(total_frame), len(event_type)+1))
        dist_array = []
        time_array = []

        for event in match_events:
            for j in range(0, len(event_type)):
                if event["eventName"] == event_type[j]:

                    start_frame = math.ceil((event["eventSec"] - 4) * fps_to_extract)
                    end_frame = math.ceil((event["eventSec"] + 4) * fps_to_extract)
                    start_event = math.ceil((event["eventSec"]) * fps_to_extract)

                    frame_window = np.array(range(start_frame, end_frame))
                    values = self.gaussian(frame_window, start_event, 6)

                    for index, frame in enumerate(frame_window):
                        if values[index] >= events_matrix[frame - 1][j + 1]:
                            events_matrix[frame - 1][j + 1] = values[index]
                            events_matrix[frame - 1][0] = 1 - values[index]


        for i in range(0, len(events_matrix)):
            if events_matrix[i][1] == 0:
                events_matrix[i][0] = 1

        print ("The target matrix has been created!")
        self.events_matrix = events_matrix
        self.dist_array = dist_array
        self.time_array = time_array

    def returnTargetMatrix(self):
        return self.events_matrix

    def returnTimeArray(self):
        return self.time_array

    def returnDistanceArray(self):
        return self.dist_array

    def saveTargetMatrix(self, path="../../outputs/",filename="target_matrix.pkl", drive=True):
        pk.dump(self.events_matrix, open(path + filename, "wb"), protocol=2)

        if drive:
            gauth = GoogleAuth()
            gauth.LocalWebserverAuth()

            drive = GoogleDrive(gauth)

            file1 = drive.CreateFile({"title" : filename, "parents": [{"kind": "drive#fileLink", "id": '1XK1EDCHkYLW9qSWwrGR3_dfBPYyDJ4QK'}]})
            file1.SetContentFile(path + filename)
            file1.Upload()



'''tags = event["tags"]
accurate = self.accurate_pass(tags)
distance = self.euclidian_distance(event['positions'])
if distance != 1000:
    dist_array.append(distance)
time_array.append(event["eventSec"])
if i+1 != len(match_events):
    time_diff = match_events[i+1]["eventSec"] - event["eventSec"]
else:
    time_diff = 2.0
if time_diff <= 1:
    startEvent = round((event["eventSec"]) * fps_to_extract)
    finishEvent = round((match_events[i+1]["eventSec"] - time_diff) * fps_to_extract) - 1
elif (time_diff > 1) and (time_diff <= 2):
    startEvent = round((event["eventSec"]) * fps_to_extract)
    finishEvent = round((event["eventSec"]+1) * fps_to_extract)
else:
    startEvent = round((event["eventSec"]) * fps_to_extract)
    finishEvent = round((event["eventSec"]+2) * fps_to_extract)'''
