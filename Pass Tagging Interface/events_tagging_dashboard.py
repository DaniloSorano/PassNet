import os
import json
import pandas as pd
from flask import Flask
from flask import request
from flask import redirect
from flask import url_for
from flask import render_template
from events_dataset import Events_DataSet
from last_file_update import last_modified_fileinfo

app = Flask(__name__)

drop_value = [
["Sassuolo vs. Internazionale 1° Tempo", "sassuolo_inter_1_Pass"],
["Sassuolo vs. Internazionale 2° Tempo", "sassuolo_inter_2_Pass"],
["Roma vs. Juventus 1° Tempo", "roma_juve_1_Pass"],
["Roma vs. Juventus 2° Tempo", "roma_juve_2_Pass"],
["Chievo vs. Juventus 1° Tempo", "chievo_juve_1_Pass"],
["Chievo vs. Juventus 2° Tempo", "chievo_juve_2_Pass"],
["Roma vs. Lazio 1° Tempo", "roma_lazio_1_Pass"],
["Roma vs. Lazio 2° Tempo", "roma_lazio_2_Pass"],
["Roma vs. Milan 1° Tempo", "roma_milan_1_Pass"],
["Roma vs. Milan 2° Tempo", "roma_milan_2_Pass"]]

matches_value = ["sassuolo_inter_1","sassuolo_inter_2","roma_juve_1","roma_juve_2","chievo_juve_1","chievo_juve_2", "roma_lazio_1", "roma_lazio_2", "roma_milan_1", "roma_milan_2"]

'''
When this script runs the command line show an address "127.0.0.1:5000/".
When the user copy and paste this link in their browser the init function
extracts the events data from the json file and save them on csv file.
After that the init function load all the data relative
to the first half of match "Sassuolo vs. Internazionale"
'''
@app.route('/')
def init():
    dataset = Events_DataSet()
    for match in matches_value:
        csv_file_pass = 'Data/' + match + '_Pass.csv'
        csv_file_shot = 'Data/' + match + '_Shot.csv'
        if os.path.isfile(csv_file_pass) == False:
            print(match[1] + " Pass CSV creation!")
            dataset.createDataMatch(match=match)
        if os.path.isfile(csv_file_shot) == False:
            print(match[1] + " Shot CSV creation!")
            dataset.createDataMatch(match=match, eventName="Shot")
    dataset = Events_DataSet()
    return redirect("/sassuolo_inter_1_Pass", code=302)

'''
The function matchView allows to change the events and the video to show.
'''
@app.route('/<match>')
def matchView(match):
    match_data = []
    csv_file = 'Data/' + match + '.csv'
    match_code = match.split("_")
    match_code = match_code[0] + "_" + match_code[1] + "_" + match_code[2]

    date_last_update = last_modified_fileinfo(csv_file)
    with open(csv_file, "r") as file:
        for line in file.readlines():
            match_data.append(line[:len(line)-1].split(";"))

    dict_name = {
        "sassuolo_inter_1_Pass" : "Sassuolo vs. Internazionale 1° Tempo (Pass)",
        "sassuolo_inter_2_Pass" : "Sassuolo vs. Internazionale 2° Tempo (Pass)",
        "roma_juve_1_Pass" : "Roma vs. Juventus 1° Tempo (Pass)",
        "roma_juve_2_Pass" : "Roma vs. Juventus 2° Tempo (Pass)",
        "chievo_juve_1_Pass" : "Chievo vs. Juventus 1° Tempo (Pass)",
        "chievo_juve_2_Pass" : "Chievo vs. Juventus 2° Tempo (Pass)",
        "roma_lazio_1_Pass" : "Roma vs. Lazio 1° Tempo (Pass)",
        "roma_lazio_2_Pass" : "Roma vs. Lazio 2° Tempo (Pass)",
        "roma_milan_1_Pass" : "Roma vs. Milan 1° Tempo (Pass)",
        "roma_milan_2_Pass" : "Roma vs. Milan 2° Tempo (Pass)",
        "sassuolo_inter_1_Shot" : "Sassuolo vs. Internazionale 1° Tempo (Shot)",
        "sassuolo_inter_2_Shot" : "Sassuolo vs. Internazionale 2° Tempo (Shot)",
        "roma_juve_1_Shot" : "Roma vs. Juventus 1° Tempo (Shot)",
        "roma_juve_2_Shot" : "Roma vs. Juventus 2° Tempo (Shot)",
        "chievo_juve_1_Shot" : "Chievo vs. Juventus 1° Tempo (Shot)",
        "chievo_juve_2_Shot" : "Chievo vs. Juventus 2° Tempo (Shot)",
        "roma_lazio_1_Pass" : "Roma vs. Lazio 1° Tempo (Shot)",
        "roma_lazio_2_Pass" : "Roma vs. Lazio 2° Tempo (Shot)",
        "roma_milan_1_Pass" : "Roma vs. Milan 1° Tempo (Shot)",
        "roma_milan_2_Pass" : "Roma vs. Milan 2° Tempo (Shot)"
    }

    return render_template('events_tagging.html', match_code=match_code, match= match, date_last_update=date_last_update, match_name = dict_name[match] ,data=match_data, dropdown = drop_value, video_link="/static/" + match_code + ".mp4")

'''
This method allows to update all the tagged events
'''
@app.route('/<match>/update', methods=['POST'])
def test(match):
    data = request.get_json()
    csv_file = 'Data/' + match + '.csv'
    eventName = match.split("_")
    eventName = eventName[len(eventName) - 1]
    match_data = data['table']
    with open(csv_file, "w") as file:
        file.writelines("Id;Player;Team;Timestamp;" + eventName + " Start;" + eventName + " End\n")
        for line in match_data[1:]:
            file.writelines(line + "\n")
    #return redirect("/", code=302)

if __name__ == '__main__':
   app.run()
