# Soccer-Pass-Detection
The code in this repository implements the **PassNet** model, the **ResNet18 + Bi-LSTM** model and the **Bi-LSTM** model that uses data extracted from a pre-trained model as input.
Also, there is the **Pass Tagging Interface** code that allows a user to define the temporal window of the Pass event annotated by Wyscout.

## PassNet, ResNet18 + Bi-LSTM and Bi-LSTM
The three models PassNet, ResNet18 + Bi-LSTM and Bi-LSTM were implemented through the Python programming language using [PyTorch](https://pytorch.org/), an open-source machine learning framework. These models use videos of football matches as input and provide a binary sequence of values, the *Pass Vector*, as output. The figure at the bottom shows the general structure that allows the models, starting from the raw data (video), to make predictions (Pass Vector).
We can see how the structure is divided into three microstructures:**Data Extraction**, **Annotation Process** and **Training/Prediction**.
![Architecture](/Scheme/Training_Process.png)
### Data Extraction
In the data extraction phase, the data is extracted from the match videos which will then be used as input for the models. Each model takes different types of data as input, for this reason, we split the typologies of data extraction: 
* Tensors Extraction.
* Features Extraction.
* Objects Position Extraction.
#### Tensors Extraction
This type of extraction allows to extract the tensors in the size 3x352x240 from each single frame. To extract the tensors from the frames you have to launch the `main.py` script that is in the `Data Extraction` folder and add four parameters: the path of the video, the name of the video, the number of fps contained in the video and the type of extraction (frame or tensor). The script saves the tensors in .pkl format in the path `Data/Input/frames_tensor/<name_of_the_video>`. An example that shows how to run the script:  
`main.py "../Data/Video/" chievo_juve_1.mp4 25 tensor`
#### Features Extraction
You can perform the features extraction by launching the `features.py` script defining three parameters: video path, video name and pre-trained template used for the extraction.  The pre-trained models that can be used are: VGG16, VGG19, ResNet18, ResNet34 and ResNet50. The extracted features are saved inside theExtract features are saved within the `Data/Input/<name_of_the_model>` path. An example that shows how to run the script:  
`feature_extraction.py "../Data/Video/" chievo_juve_1.mp4 resnet18`.
#### Objects Position Extraction
The object position extraction (ball and players) uses a PyTorch implementation of the real-time object detection model, Yolov3, on [GitHub](https://github.com/eriklindernoren/PyTorch-YOLOv3). In order to extract the positions we had to modify the `detect.py` file inside the repository. The extraction of the positions requires first of all the download of the original repository from which we have to modify the `detect.py` file, present in our repository in the YOLO folder. After that you can start the `detect.py` script by defining the parameters from row 27 to 38 of `detect.py`:  
```python
parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")  
parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")  
parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")  
parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")  
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")  
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")  
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")  
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")  
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")  
parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")  
parser.add_argument("--save_image", type=str, help="path to checkpoint model")  
parser.add_argument("--bbox_coordinates", type=str, help="create a txt file where save bbox coordinates")
```
### Training/Predictions
We use the data extracted during the training phase as input for our three models that will return the predictions as output. The models can be trained and then used to make predictions.
#### Training
The code that allows you to train models is inside the Training folder, which contains: ```the TrainModel.py``` script and the ```Model Metrics```, ```Model Parameters``` and ```Model Steps``` folders. The training of one of the models is done first by defining the initialization parameters in the ```train_ini.json``` file in the Model Parameters folder. These parameters are:
* ```input_dim```: dimension of the input feature vector of the Bidirectional LSTM (512 for ResNets and 4096 for VGGs).
* ```hidden_dim```: dimension of the dense layers.
* ```layer_dim```: number of LSTM units.
* ```output_dim```: dimension of the output (default value is 1).
* ```readout_dim```: number of dense layers.
* ```drop_out```: the value of dropout.
* ```yolo```: if 'True' the model use the objects position as input.
* ```yolo_dimension```: This value defines the dimension of the Object position vector (default value is 24).
* ```cnn_model```: pre-trained model used to extract features (default value is resnet18).
* ```optimizer```: the type of optimizer (ADAM, SGD, SGD-Nesterov).
* ```learning_rate```: the learning rate value.
* ```momentum```: the momentum value (for ADAM optimizer this value is not considered).
* ```matches_code_train```: a list of the match (name of the video without extension) used to train the model.
* ```matches_code_val```: a list of the match (name of the video without extension) used to test the model.
* ```model_type```: the typology of the model to train.
* ```batch_size```: the batch dimension.
* ```seq_dim```: the number of tensors/features inside an input sequence.
* ```eval_epochs```: the number of epochs each time you need to evaluate the performances of the model on the matches defined in the ```matches_code_val``` parameter.
* ```epochs```: the number of the epochs.
* ```device```: the type of device to use ("cpu" or "gpu").
* ```shuffle```: if 'True' the sequence are shuffled at each epoch.
* ```save_epoch_model```: if 'True' the model are saved at each epoch.
* ```load_last_model```: this option allows to load the last saved model.
The file ```train_ini.json``` contains some default parameters.
To choose what kind of model to train you have to define some parameters, specifically:
##### PassNet
```json
{
  "yolo" : true,
  "yolo_dimension" : 24,
  "model_type" : 'CNN+LSTM',
}
```
##### ResNet18 + Bi-LSTM
```json
{
  "yolo" : false,
  "yolo_dimension" : 0,
  "model_type" : 'CNN+LSTM',
}
```
##### Bi-LSTM
```json
{
  "yolo" : false,
  "yolo_dimension" : 0,
  "model_type" : 'LSTM',
}
```
