### Sustainable roof mapping fromsatellite image inventory

### Introduction
This repository contains the source code for the a sequential framework of the master thesis 'Sustainable roof mapping fromsatellite image inventory'. The proposed sequential framework in the thesis detects and calcualtes pontential green roof areas from a given DHM and Optical inventory.

The sequential framework consist of:
- A set of processing functions for generating tiles from a given bigger raster file (raster_processor.py).
- Developed FCN U-net architecture models with their training, testing and predicition methods.(U_NET.py, U_Net_leakt.py and U_NET_relu.py)
- A main file used to input tiles form the proposed raster processor that uses the trained model to predicted roof areas from the DHM inventory to detect potential green roof areas (main.py).

### Language and tools
The sequential framework is developed using Python. For the libraries [tensorflow](https://www.tensorflow.org/) API, [keras](https://keras.io/) and [GDAL](https://gdal.org/).

-  [tensorflow](https://www.tensorflow.org/) and [keras](https://keras.io/) are used as a libraries to build the U-Net CNN and train the model.
- [GDAL](https://gdal.org/) is used to calcualte slope of the predicted roofs to classify the pixels as potential green roof areas.
- 
Additionally, the project is written using Python 3.7.2 and follows a recommended folder structure as follows:

```
.
├── requirements.txt                # Libraries to installed used by the framework
├── data                            # Folder containing raw and processed data
│   ├── geo                         # Folder containing the raw tile
│   │   └── raster_1.tiff
│   │   └── raster_1.elevation.tiff
│   ├──  train_val
│   │       └──  test               # folders containing   train dataset
│   │       └──  train              # folders containing  test dataset
├── logs                            # Contains the logs from the keras dashboard
├── tmp                             # Contains tmp files used by the mai.py sequential model 
├── model                           # Container the stored model after training
├── U_Net.py                        # Container of the U_Net.py model
├── U_Net_leakypy                   # Container of the U_Net_leaky.py model
├── U_Net_relu.py                   # Container of the U_Net_relu.pt model
├── main.py                         # Container of  sequential framework scripts
├── elevation_processor.py          # Container of  GDAL scripts over the  predicted tiles
├── predicted                       # Folder with the predicted green roof areas from the sequential model
│       ├── merged__.shp            # Shapefile with the potential green roof areas extracted by the model
└── README.md
```

## Installation
The following environment requires the usage of the following:

1. Python 3.7.2

For installing dependencies `pip` is used. The dependencies used in this project can be found in the *requirements.txt* file. In order to install them use `pip install -r requirements.txt`.

### Processing the data
To process the data and create the train and test dataset locate the raster file into the data/geo and polygon shapefile is located at data/geo. Also that directory data/train_val is structured as it is shown above. 
Finally run `python dataset_processor.py` to create the 544x544 tiles. Make in consideration that parent tile must be multiple of 544.

Optical and elevtion inventories together with CPH shapefile were downlaoded from:
[data provider](https://download.kortforsyningen.dk/content/vilk%C3%A5r-og-betingelser)

### Training the model
Developed models consist of: U_Net.py,  U_Net_relu.py and  U_Net_leaky.py. After spliting the dataset, models are trained by running `python U_Net.py`, for instance for the U_net model. Dropout and learning parameters are set to 0.25 and  2*math.pow(10, -4), respectively. These values are configurable by modifing its constant value in their respective files.

In order to use the evalue the model run: `python U_Net.py evaluate`. Make sure WEIGHT_FILE contains the trained weights of the CNN so those are loaded into the built model. Other configurable parameters are:
- MODEL_DIR = 'dirrectory to save the model.
- DATA_DIR = directory where data to train is stored
- TRAIN_DIR = ' directory where train set is stored
- VAL_DIR =  directory where validation set is stored
- CHECK_POINT_PATH = directory where weights are stored


### Plotting values

Developed models allow the plotting of their predicted values for a visual evaluaion of those. To plot the predicted values, make sure weights are stored in the weight directory and the test set contains the tiles to predict. Run `python U_Net.py evaluate`, for instance to evalute the results of the U_net model.



### Result

Result images are stored in the results folder. Test contained FP, FN, TP, TN plots of the predictions in a different set of tiles. Then results/predicted containt the predicted green roof areas with.


### Green roof potential areas extraction

Main.py containg the main script to execute the whole sequential model. The sequential model uses a U_net_leakyRelu model for roof segmentation and Gdal functionalities to evaluate the predicted segments. To run the script make sure data is available in the IN_DIR constant variable and weights to load too WEIGHT_FILE. Finally run `python main.py ` for leakyrelu model or  `python main.py relu` for U_net_relu model.

Predicted values are stored in a shapefile called merged_XX.shp


