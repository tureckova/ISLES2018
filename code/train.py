import os
#sys.path.append('/home/alzbeta/3DUnetCNN')

config = dict()
# directories
config["output_foder"] = os.path.abspath("/home/alzbeta/data/Ischemic_stroke_lesion/3DUnet_v2/output/EXP18/") # path to folder where the outputs should be saved
config["data_folder"] = os.path.abspath("/home/alzbeta/data/Ischemic_stroke_lesion/PREPROCESSED/TRAINING-2/") #path to folder containing the training data
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
config["mean"] = [144.85851701, 49.06570323, 6.42394785, 2.36531961, 1.16228806]#preprocessed
config["std"] = [530.26314967, 127.95098693, 16.96937125, 4.24672666, 3.35688005]#preprocessed
#config["mean"] = [80.18678446, 83.45001279, 11.93870835, 2.52698151, 1.2582182]
#config["std"] = [304.3473741, 228.48690137, 32.85048644, 4.60773065, 3.45508374]
config["labels"] = 1 # the label numbers on the input image (exclude background)

# training settings
config["batch_size"] = 4
config["n_epochs"] = 2000  # cutoff the training after this many epochs 
config["initial_learning_rate"] = 5e-3
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
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

# create or load model
if not os.path.exists(config["model_file"]):
    from model import isensee2017_model
    input_shape=(len(config["modalities"]),)+config["input_size"]+config["nr_slices"]
    model = isensee2017_model(input_shape=input_shape,
                              n_labels=config["labels"],
                              initial_learning_rate=config["initial_learning_rate"],
                              optimizer="Adam")
else:
    #from utils import ModelLoader
    #model = ModelLoader(config["model_file"])
    from model import isensee2017_model_dil as isensee2017_model
    input_shape=(len(config["modalities"]),)+config["input_size"]+config["nr_slices"]
    model = isensee2017_model(input_shape=input_shape,
                              n_labels=config["labels"],
                              initial_learning_rate=config["initial_learning_rate"],
                              optimizer="Adam")
    model.load_weights("/data/alzbeta/Ischemic_stroke_lesion/3DUnet_v2/output/EXP11/weights3/model_weights391-0.41.h5")

print("[INFO] model compiled")

# create polynomial learning rate scheduler
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, CSVLogger
import math
def poly_step_decay(epoch):
    # initialize the base initial learning rate, drop factor, and epochs to drop every
    initLR = config["initial_learning_rate"]
    endLR = config["initial_learning_rate"]*1e-7
    dropEvery = config["n_epochs"]
    power = 0.9
    # compute learning rate for the current epoch
    dropEvery = dropEvery * math.ceil((epoch+1) / dropEvery)
    LR = (initLR - endLR) * (1 - epoch / dropEvery) ** (power) + endLR
    # return the learning rate
    return float(LR)

# create training callbacks
callbacks = [LearningRateScheduler(poly_step_decay, verbose=1),
             ModelCheckpoint(monitor='val_loss',
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


