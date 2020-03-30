import os
import argparse
from frame_extractor import extract_frames
from frame_extractor import extract_tensor

#Initialize parser arguments
parser = argparse.ArgumentParser(description='Frame Extraction')
parser.add_argument("path_input", type=str, help="The path of the input")
parser.add_argument("video_name", type=str, help="Name of the video")
parser.add_argument("fps", type=int, help="Number of fps (frames per seconds) of the video")
parser.add_argument("type", type=str, help="Extract Tensor of Frames")
args = parser.parse_args()

#Check if the frames dir of the video exists
if args.type == "tensor":
    path_output= "../Data/Input/frames_tensor/" + args.video_name.split(".")[0]
else:
    path_output = "../Data/Input/frames_image/" + args.video_name.split(".")[0]

if not os.path.exists( path_output ):
    os.makedirs( path_output, mode=0o0775 )

#Extract frames
path_input_video = args.path_input
if args.type  == "tensor":
    extract_tensor(path_input_video + args.video_name, path_output)
else:
    extract_frames(path_input_video + args.video_name, path_output)
