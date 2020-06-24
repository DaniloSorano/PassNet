from csvToEventsMatrix import csvToEventsMatrix
import sys

# Creation of target matrix and trasfer it to Google Drive
filename = sys.argv[1]
filename = filename.split(".")[0]
if filename == "roma_juve_1":
    matchId = 2576322
    period = "first_half"
elif filename == "roma_juve_2":
    matchId = 2576322
    period = "second_half"
elif filename == "sassuolo_inter_1":
    matchId = 2576136
    period = "first_half"
elif filename == "sassuolo_inter_2":
    matchId = 2576136
    period = "second_half"
elif filename == "chievo_juve_1":
    matchId = 2759433
    period = "first_half"
elif filename == "chievo_juve_2":
    matchId = 2759433
    period = "second_half"
    
targetMatrixBuilder = csvToEventsMatrix(match_code = filename, pass_type="all")
targetMatrixBuilder.targetMatrix()
targetMatrixBuilder.saveTargetMatrix()
target_matrix = targetMatrixBuilder.returnTargetMatrix()
print("Target Matrix: ")
print(target_matrix)
