# classification-CNN-medical-Kaggle

## Classifying medical images with a convolutional neural network

### I. The dataset

I apply a convolutional neural network (CNN) to 5000 histological images of human colorectal cancer made available by
Kather JN et al (2016), and taken from the Institute of Pathology of the University of Heidelberg in Mannheim 
(https://zenodo.org/record/53169#.XFoqfs9KjOS). The images are magnified by 20x, contain 3 channels RGB
and 150x150 pixels (corresponding to 74x74 µm, 0.495 µm per pixel). 
  
The dataset contains 8 distinct classes (named 'tumor', 'stroma', 'complex', 'lympho', 'debris',
'mucosa', 'adipose' and 'empty'). Each image corresponds to one class, and each 
class contains 625 images. 


Citation: Kather JN, Weis CA, Bianconi F, Melchers SM, Schad LR, Gaiser T, Marx A, Zollner F: Multi-class texture 
analysis in colorectal cancer histology (2016)


### II. The model architecture (VGG16 base)

I use a pre-trained VGG16 model base and add a densely connected classifier 
(DC) on top of it. The classification is done in a last densely connected classifier
which is trained on the labelled data. In this way, it is possible to save 
computation time by using the pre-trained weights from the model base as input weights to 
the training of the data, instead of having to train the weights from scratch.

First, 'features' (predictions) are extracted from the model base by running 
the training images only once through the model base. Then, the resulting output is used 
as input to a densely connected classifier (using Relu function), followed by a layer of 
dropout regularization. The last densely connected classifier (using a sigmoid 
function) finally trains the images on the 8 categories of labelled data. 
Dropout regularization removes outcomes of the densely connected classifier if its
probability exceeds a given threshold (see below).
For the optimization, an RMSprop is used as optimizer and a categorical crossentropy 
as loss function.

The dataset of 5000 images is distributed evenly over the 8 classes, so that each class 
contains 625 images. For each class, the images are split into 425 training, 100 
validation and 100 test images.  

We vary the number of nodes in the densely connected classifier as well as the dropout 
regularization threshold.  


|  fixed features of the CNN |  |
|----- |-----   | 
|  total nb of images (per class)|   625     |
|  nb of training images (per class)|  425      |
|  nb of validation images (per class)|  100    |
|  nb of test images (per class)|  100    |
|  nb of images per batch (for validation) | 25      |
| nb of epochs  |  250     |


|  hyperparameters |  |
|----- |-----   | 
| nb of nodes   | 50, 55, 60 |
| dropout threshold   | 0.5, 0.55, 0.60 |

 
 from which I extract the weights  and stack additional layer to   
he densely connected classifier on top of the

### III. Results: The accuracy of the model varies with different hyperparameters 

###### 50 nodes and dropout probability 0.5 

![Test 50nodes](Nodes 50 layer 1/Accuracy_VGG16_nodesL1_50_nodesL2_0_epochs250_dropout0.5.png)


###### 55 nodes and dropout probability 0.5 
![Test 55nodes](Nodes 55 layer 1/Accuracy_VGG16_nodesL1_55_nodesL2_0_epochs250_dropout0.5.png)



