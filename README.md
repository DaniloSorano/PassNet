# Soccer-Pass-Detection
The code in this repository implements the **PassNet** model, the **ResNet18 + Bi-LSTM** model and the **Bi-LSTM** model that uses data extracted from a pre-trained model as input.
Also, there is the **Pass Tagging Interface** code that allows a user to define the temporal window of the Pass event annotated by Wyscout.

## PassNet and ResBi
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
  "model_type" : "CNN+LSTM",
}
```
##### ResNet18 + Bi-LSTM
```json
{
  "yolo" : false,
  "yolo_dimension" : 0,
  "model_type" : "CNN+LSTM",
}
```
##### Bi-LSTM
```json
{
  "yolo" : false,
  "yolo_dimension" : 0,
  "model_type" : "LSTM",
}
```
After defining the parameters you can start training the model by launching the ```TrainModel.py``` script. This script in addition to training the model calculates the metrics at each time on the train set and the test set at each time defined in the ```eval_epochs``` parameter. Also, if the ```save_epoch_model``` parameter is set to True then the models at each step and the test set predictions associated with that step are saved in the ```Model Steps``` folder.  
We save the metrics for the test and train set inside the folder ```Model Metrics``` in a file called ```model_metrics.json```. The metrics saved into the file are: Confusion Matrix value, Loss, Average Precision, Accuracy, Precision "Pass", Precision "No Pass", Recall "Pass", Recall "No Pass" and F1 Score.
##### Example of ```model_metrics.json``` structure
```json
{
  "confusion_matrix": [[188668, 63762, 70067, 21903], [11236, 448, 3491, 195], [241551, 10879, 16842, 75128], [250776, 1654, 1879, 90091], [10565, 1119, 2462, 1224], [251127, 1303, 1452, 90518], [251622, 808, 1038, 90932], [8848, 2836, 1375, 2311], [251803, 627, 885, 91085], [251755, 675, 757, 91213], [8167, 3517, 1046, 2640], [251972, 458, 673, 91297], [251984, 446, 538, 91432], [9721, 1963, 1849, 1837], [252029, 401, 469, 91501], [252095, 335, 362, 91608], [10319, 1365, 2740, 946]], 
  "loss": [0.6884880840224533, 0.6860015377000306, 0.18270250378266384, 0.03171568362711493, 1.3286416243142554, 0.024866783258999957, 0.016759728769198748, 1.2782480960681497, 0.013865658273496829, 0.012441815979153668, 1.7202938279400213, 0.01026973170815495, 0.008810981937370678, 1.1938869221873516, 0.007898680339175061, 0.006479332306761459, 2.1567604498422766], 
  "mean_average_precision": [0.2594187828449709, 0.2764541486466187, 0.9367015227364686, 0.9972836297838847, 0.49238745473890033, 0.9982803090309817, 0.9990554740591759, 0.4781191395160796, 0.9993965579854873, 0.9995103367509003, 0.4690694077557938, 0.9996765066596487, 0.9996957309741084, 0.48987398813707, 0.9997646650372644, 0.9998190724419809, 0.377285270058428], 
  "accuracy": [61.14140534262486, 74.37215354586857, 91.95092915214866, 98.97415795586527, 76.70136629798309, 99.2000580720093, 99.46399535423926, 72.6024723487313, 99.5609756097561, 99.5842044134727, 70.31229668184776, 99.67160278745645, 99.71428571428571, 75.19843851659076, 99.74738675958189, 99.79761904761905, 73.29212752114509], 
  "precision_pass": [25.568201715986692, 30.326594090202178, 87.3510295673608, 98.19717695787236, 52.24071702944942, 98.58093464458021, 99.11925005450185, 44.89994171361958, 99.31633810188416, 99.26541006442626, 42.878025012181254, 99.50084464061904, 99.5145736737848, 48.3421052631579, 99.56366564383801, 99.63564382280326, 40.934660320207705], 
  "precision_no_pass": [72.91939629350493, 76.29524003530929, 93.4820215717918, 99.25629811402901, 81.1007906655408, 99.42513035525519, 99.58917121823795, 86.54993641788124, 99.64976571898943, 99.70021226713978, 88.6464778031043, 99.7336183181935, 99.78694925590642, 84.019014693172, 99.81425595450261, 99.85660924434656, 79.0183015544835], 
  "recall_pass": [23.815374578666958, 5.290287574606619, 81.68750679569425, 97.95694248124389, 33.2067281606077, 98.42122431227574, 98.8713710992715, 62.696690179055885, 99.03772969446558, 99.17690551266718, 71.6223548562127, 99.26823964336197, 99.41502663912145, 49.83722192078133, 99.49005110362074, 99.60639338914864, 25.66467715680955], 
  "recall_no_pass": [74.74072019965931, 96.16569667921945, 95.6902903775304, 99.34476884680902, 90.42280041081821, 99.48381729588401, 99.67991126252822, 75.72749058541595, 99.75161430891733, 99.73259913639424, 69.89900718931872, 99.81856356217565, 99.82331735530641, 83.19924683327628, 99.84114407954681, 99.86728994176603, 88.31735706949675], 
  "f1_score": [24.660680609114195, 9.00900900900901, 84.42439191580935, 98.07691260920447, 40.603748548681374, 98.50101473956832, 98.99515540797996, 52.326502886901395, 99.17683823129103, 99.22113805219246, 53.64218226150564, 99.38440604163831, 99.46477524911884, 49.078279454982635, 99.52684476157327, 99.62101645886914, 31.54910788727697], 
  "dataset_type": ["Train", "Validation 0", "Train", "Train", "Validation 0", "Train", "Train", "Validation 0", "Train", "Train", "Validation 0", "Train", "Train", "Validation 0", "Train", "Train", "Validation 0"]
}
```
#### Predicting
The models saved in the training phase in ```.pth``` format can be used to annotate new videos and obtain the model's performance in terms of *Accuracy*, *Precision 'Pass'*, *Precision 'No Pass'*, *Recall 'Pass'*, *Recall 'No Pass'* and *F1 Score 'Pass'*. The folder ```Predicting``` allows to do this, in particular the script ```video_prediction.py``` returns the predictions without threshold, the predictions with the threshold equal to 0.5 and the metrics of the video passed as input.
Before running the script you need to perform two basic operations, the first one requires you to insert the model to load in ```.pth``` format in the ```Model State``` folder. The second operation requires you to set the file ```video_lab_ini.json``` inside the ```Model Ini``` folder with the model parameters and the name of the video you want to annotate. The file parameters are the same as those in ```train_ini.json``` except for some, such as: :
* ```model_to_load```: this parameter is numeric and refers to the model number to load. In practice, the models saved during training are marked with a number that identifies the epoch.
* ```threshold```: the treshold to apply to predictions.
* ```matches_predicitions```: the name of the video to annotate. The folder with the tensors of the frames must be saved inside the path ```Data/Input/frame_tensor/<name_of_the_video>```.

## Pass Tagging Interface
The pass tagging interface allows to define the time window of the pass event. The image below shows how the mainly UI of the interface.  

![PassTaggingInterface](/Scheme/manual_annotation_application-1.png)  

a) The dropdown element allows to select wich match you want to tag,  
b) The table shows all the pass event that occurs during the match. The rows are clickable and set the video at the start time of the pass event. In addition when you set the time window you can save the data inside the specific csv by clicking the button 'Update CSV'.  
c) The player shows the video of the match. The buttons under the video allows to play/pause the video, go to previous frame, go to next frame, define the start time and the end time of the pass event.

### Run the interface


