import cv2
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import pickle as pk

'''
This script implements two functions to extract from video of the matches the frames or the tensor
'''

# This function extract 5 FPS from a video passed in input.
# We save the JPGs in a specific folder associated with the match
def extract_frames(path_video_input, path_frames_output, fps=25):
  frame_code = path_frames_output.split("/")[len(path_frames_output.split("/"))-1].split("_")
  frame_code = list(frame_code[0])[0] + "_" + list(frame_code[1])[0] + "_" + frame_code[2]
  vidcap = cv2.VideoCapture(path_video_input)
  total_frame = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
  count = 1
  tq = tqdm(total=total_frame, unit='B', unit_scale=True)
  success,image = vidcap.read()
  if fps == 25:
    frame_n = 1
    while success:
        if count in [1,6,11,16,21]:
          cv2.imwrite(path_frames_output + "/" + frame_code + "_" + str(frame_n) + ".jpg", image)
          frame_n += 1
        if count == 25:
            count=1
        else:
            count += 1
        success,image = vidcap.read()
        tq.update(1)
    tq.close()
  else:
    frame_n = 1
    while success:
        if count in [1,6,11,16,21,26]:
          cv2.imwrite(path_frames_output + "/" + frame_code + "_" + str(frame_n) + ".jpg", image)
          frame_n += 1
        if count == 25:
            count=1
        else:
            count += 1
        success,image = vidcap.read()
        tq.update(1)
    tq.close()

# This function extract 5 FPS from a video passed in input and transform them into tensor.
# We save the tensor in a specific folder associated with the match
def extract_tensor(path_video_input, path_frames_output, fps=25):
  preprocess = transforms.Compose([
      transforms.Resize([352, 240]), #240p
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  frame_code = path_frames_output.split("/")[len(path_frames_output.split("/"))-1].split("_")
  frame_code = list(frame_code[0])[0] + "_" + list(frame_code[1])[0] + "_" + frame_code[2]
  vidcap = cv2.VideoCapture(path_video_input)
  total_frame = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
  count = 1
  tq = tqdm(total=total_frame, unit='B', unit_scale=True)
  success,image = vidcap.read()
  if fps == 25:
    frame_n = 1
    while success:
        if count in [1,6,11,16,21]:
          image = Image.fromarray(image)
          input_tensor = preprocess(image).unsqueeze(0)
          pk.dump(input_tensor, open(path_frames_output + "/" + frame_code + "_" + str(frame_n) + '.pickle', 'wb'))
          frame_n += 1
        if count == 25:
            count=1
        else:
            count += 1
        success,image = vidcap.read()
        tq.update(1)
    tq.close()
  else:
    frame_n = 1
    while success:
        if count in [1,6,11,16,21,26]:
          input_tensor = preprocess(input_image).unsqueeze(0)
          pk.dump(input_tensor, open(path_frames_output + "/" + frame_code + "_" + str(frame_n) + '.pickle', 'wb'))
          frame_n += 1
        if count == 25:
            count=1
        else:
            count += 1
        success,image = vidcap.read()
        tq.update(1)
    tq.close()
