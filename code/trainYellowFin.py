import os
#sys.path.append('/home/alzbeta/3DUnetCNN')

config = dict()
# directories
config["output_foder"] = os.path.abspath("/home/alzbeta/data/Ischemic_stroke_lesion/3DUnet_v2/output/EXP13/") # outputs path
config["data_folder"] = os.path.abspath("/home/alzbeta/data/Ischemic_stroke_lesion/PREPROCESSED/TRAINING-2/") # training data path
config["tensorboar_log_dir"] = os.path.join(config["output_foder"], "log")
config["model_folder"] = os.path.join(config["output_foder"], "model")
if not os.path.exists(config["model_folder"]):
    os.makedirs(config["model_folder"])
config["model_file"] = os.path.join(config["model_folder"], "model.h5")
config["wieghts_file_bestval"] = os.path.join(config["model_folder"], "BEST_val.h5")
config["wieghts_file_lasttrain"] = os.path.join(config["model_folder"], "LAST_train.h5")
config["training_file"] = os.path.join(config["output_foder"], "training_ids.pkl")
config["validation_file"] = os.path.join(config["output_foder"], "validation_ids.pkl")
config["logging_file"] = os.path.join(config["output_foder"], "training.log")
config["overwrite"] = False  # If True, will overwite previous files. If False, will use previously written files.

# model settings
config["input_size"] = (128,128) # size of image (x,y)
config["nr_slices"] = (32,) # all inputs will be resliced to this number of slices (z axis), must be power of 2
config["modalities"] = ["CT", "CBF", "CBV", "MTT", "Tmax"]#["CBF", "CBV", "MTT", "Tmax"]
config["mean"] = [144.85851701, 49.06570323, 6.42394785, 2.36531961, 1.16228806] # mean of each modality in the dataset
config["std"] = [530.26314967, 127.95098693, 16.96937125, 4.24672666, 3.35688005] # std of each modality in the dataset
config["labels"] = 1 # the label numbers on the input image (exclude background)
config["threshhold"] = 0.5 # threshold used to convert output heat map to output mask with 0 and 1 only, i.e. >thresh => 1

# training settings
config["batch_size"] = 4
config["n_epochs"] = 2000  # cutoff the training after this many epochs 
config["test_size"] = 0.2  # portion of the data that will be used for validation

# load the dataset from disk
from utils import NiiDatasetLoader
from utils import Resize
preprocessor = [Resize(config["input_size"]+config["nr_slices"])]
(data, labels) = NiiDatasetLoader(preprocessor).load(config["data_folder"], config["modalities"])

# normalize data
def normalize_data(data, mean, std):
    for i in range(len(data)):
        for j in range(len(mean)):
            data[i][j] -= mean[j]
            data[i][j] /= std[j]
    return data 
data = normalize_data(data, config["mean"], config["std"])

# chck if output folder exist if not create
if not os.path.exists(config["output_foder"]):
    os.makedirs(config["output_foder"])

# partition the data into training and testing splits
import pickle
if not os.path.exists(config["validation_file"]) or config["overwrite"]:
    from sklearn.model_selection import train_test_split
    indices = [x for x in range(len(data))]
    training_indices, validation_indices = train_test_split(indices, test_size=config["test_size"])
    # save validation indexes
    with open(config["validation_file"], "wb") as opened_file:
        pickle.dump(validation_indices, opened_file)
    # save training indexes
    with open(config["training_file"], "wb") as opened_file:
        pickle.dump(training_indices, opened_file) 
else:
    # load validation indexes
    with open(config["validation_file"], "rb") as opened_file:
        validation_indices = pickle.load(opened_file)
    # load training indexes
    with open(config["training_file"], "rb") as opened_file:
        training_indices = pickle.load(opened_file)
trainX = [data[i] for i in training_indices]
trainY = [labels[i] for i in training_indices]
testX = [data[i] for i in validation_indices]
testY = [labels[i] for i in validation_indices]

# create model
from model import isensee2017_model
input_shape=(len(config["modalities"]),)+config["input_size"]+config["nr_slices"]
model = isensee2017_model(input_shape=input_shape,
                          n_labels=config["labels"],
                          initial_learning_rate=0.005,
                          optimizer="YellowFin")
print("[INFO] model compiled")

# create training callbacks
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
callbacks = [ModelCheckpoint(monitor='val_loss',
                             filepath=config["model_file"],
                             save_best_only=True,
                             save_weights_only=False),
             ModelCheckpoint(monitor='val_loss',
                             filepath=config["wieghts_file_bestval"],
                             save_best_only=True,
                             save_weights_only=True),
             ModelCheckpoint(monitor='val_loss',
                             filepath=config["wieghts_file_lasttrain"],
                             save_best_only=False,
                             save_weights_only=True),
             TensorBoard(log_dir=config["tensorboar_log_dir"]),
             CSVLogger(config["logging_file"], append=True)]

# training
from utils import BatchGenerator
import numpy as np
model.fit_generator(generator=BatchGenerator(trainX, trainY, config["batch_size"], augment = True),
                    steps_per_epoch=np.ceil(float(len(trainX)) / config["batch_size"]),
                    epochs=config["n_epochs"],
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=BatchGenerator(testX, testY, config["batch_size"], augment = False),
                    validation_steps=np.ceil(float(len(testX)) / config["batch_size"])
                    )


