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
This type of extraction allows to extract the tensors in the size 3x352x240 from each single frame.
