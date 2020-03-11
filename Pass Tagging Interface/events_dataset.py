import json
import pandas as pd


'''
This class extract the events of a match.
The initialization of the class allows to read the json file of the events, players and teams.
'''
class Events_DataSet():
    def __init__(self):
        with open("Data/players.json", encoding='utf-8') as file_players:
            data_players = json.load(file_players)

        self.playersDict = dict()
        for player in data_players:
            self.playersDict[player['wyId']] = player['shortName']
        self.playersDict[25548] = "Non Disponibile"

        with open("Data/teams.json", encoding='utf-8') as file_teams:
            data_teams = json.load(file_teams)

        self.teamsDict = dict()
        for team in data_teams:
            self.teamsDict[team['wyId']] = team['name']

        with open("Data/events_Italy.json", encoding='utf-8') as file_events:
            self.data_events = json.load(file_events)

        with open("Data/chievo-juve-2018-08-18.json", encoding='utf-8') as file_events:
            chievo_juve = json.load(file_events)
            self.data_events += chievo_juve['events']

'''
This function extracts the specific events of a specific match
'''
    def createDataMatch(self, match="sassuolo_inter_1", eventName = "Pass"):
        matchId = 0
        period = 0
        print(match)
        if match == "sassuolo_inter_1":
            matchId = 2576136
            period = "1H"
        elif match == "sassuolo_inter_2":
            matchId = 2576136
            period = "2H"
        elif match == "roma_juve_1":
            matchId = 2576322
            period = "1H"
        elif match == "roma_juve_2":
            matchId = 2576322
            period = "2H"
        elif match == "chievo_juve_1":
            matchId = 2759433
            period = "1H"
        elif match == "chievo_juve_2":
            matchId = 2759433
            period = "2H"
        elif match == "roma_lazio_1":
            matchId = 2576084
            period = "1H"
        elif match == "roma_lazio_2":
            matchId = 2576084
            period = "2H"
        elif match == "roma_milan_1":
            matchId = 2576214
            period = "1H"
        elif match == "roma_milan_2":
            matchId = 2576214
            period = "2H"

        events_filter = list(filter(lambda x : x["matchId"] == matchId and x["eventName"] == eventName and x["matchPeriod"] == period, self.data_events))
        row_format = lambda x : str(x["id"]) + ";" + self.playersDict[x["playerId"]] + ";" + self.teamsDict[x["teamId"]] + ";" + str(round(x["eventSec"], 2))+ ";0;0"
        header = ["Id;Player;Team;Timestamp;" + eventName + " Start;" + eventName + " End"]
        events_match_csv = header + [*map(row_format, events_filter)]

        with open("Data/" + match + "_" + eventName + ".csv", "w") as csv_file:
            for ev in events_match_csv:
                csv_file.writelines(ev + "\n")
