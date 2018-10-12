import os
import nibabel as nib
import numpy as np
import pickle
import glob

config = dict()

# model settings
config["input_size"] = (128,128) # size of image (x,y)
config["nr_slices"] = (32,) # all inputs will be resliced to this number of slices (z axis), must be power of 2
config["all_modalities"] = ["CT", "CBF", "CBV", "MTT", "Tmax"]#["CBF", "CBV", "MTT", "Tmax"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["label"] = (1,) 
config["n_base_filters"] = 16
config["threshhold"] = 0.1 # threshold used to convert output heat map to output mask with 0 and 1 only, i.e. >thresh => 1
config["mean"] = [144.85851701, 49.06570323, 6.42394785, 2.36531961, 1.16228806]#preprocessed
config["std"] = [530.26314967, 127.95098693, 16.96937125, 4.24672666, 3.35688005]#preprocessed
#config["mean"] = [80.18678446, 83.45001279, 11.93870835, 2.52698151, 1.2582182]
#config["std"] = [304.3473741, 228.48690137, 32.85048644, 4.60773065, 3.45508374]


def normalize_data(data, mean, std):
    for i in range(data.shape[0]):
        data[i] -= mean[i]
        data[i] /= std[i]
    return data   

def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    #print('weighted_dice_coefficient call')
    return np.mean(2. * (np.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(np.sum(y_true,
                                                            axis=axis) + np.sum(y_pred,
                                                                               axis=axis) + smooth))

from utils import NiiImagesLoader, Resize, ModelLoader
import nibabel

def predict_one_case(subject_folder, mean, std, caseID, model):
    # load data
    preprocessor = [Resize(config["input_size"]+config["nr_slices"])] # prepare preprocessor for resizing
    data, label = NiiImagesLoader(preprocessor).load(subject_folder, config["training_modalities"]) # load data
    print("input shape: ", data.shape)
    # normalize input image
    data = normalize_data(data, mean, std)
    # create mirrored copy of input
    data2 = np.flip(data, axis=(2))
    # expand dimension of batch size
    data = np.expand_dims(data, axis=0)
    data2 = np.expand_dims(data2, axis=0)
    # predict output
    prediction = model.predict(data)[0,0]
    prediction2 = model.predict(data2)[0,0]
    # mirror the output back
    prediction2 = np.flip(prediction2, axis=(1))
    # load CT image to get SMIR ID, original size, header and afiine
    MTT_path = glob.glob(os.path.join(subject_folder, "*MTT.*.nii"))[0] # get right ID for SMIR
    CT_path = glob.glob(os.path.join(subject_folder, "*CT.*.nii"))[0] # get right header for SMIR
    CT = nib.load(CT_path)
    # transpose label mat to mask
    prediction = np.mean(np.array([prediction, prediction2]), axis=0)
    label_map_data = np.zeros(prediction.shape, np.int8)
    label_map_data[prediction > config["threshhold"]] = 1
    # write prediction to niftiimage into prediction_path folder
    prediction = Resize(CT.shape, interpolation = "nearest").preprocess(label_map_data)
    predNifti = nib.Nifti1Image(prediction, CT.affine, CT.header)
    print("Output prediction: ", prediction.shape)
    # predNifti.set_data_dtype('short')
    if not os.path.exists(config["prediction_path"]):
        os.makedirs(config["prediction_path"])
    prediction_path = os.path.join(config["prediction_path"], "SMIR.prediction"+ config["output_foder"].split("/")[-1] + "_case" + caseID + "." + MTT_path.split(".")[-2] + ".nii")
    predNifti.to_filename(prediction_path)
    if config["test_data"]!=True:
        # evaluate dice coeficient
        dice = weighted_dice_coefficient(label, prediction)
        print("Dice: ", dice)
        return dice

def main():
    # choose which data use for evaluation
    if config["test_data"]==True:
        validation_indices = [i for i in range(1,63)]
    elif config["train_data"]==True:
        validation_indices = [i for i in range(1,95)]
    else:
        # load validation indexes
        with open(config["validation_file"], "rb") as opened_file:
            validation_indices = pickle.load(opened_file)
    dice = []
    # load model and wieghts
    model = ModelLoader(config["model_file"])
    model.load_weights(config["wieghts_file"])
    for i in validation_indices:
        subject_folder = os.path.join(config["data_folder"],"case_" + str(i))
        print("Procesing folder: " + subject_folder)
        dice.append(predict_one_case(subject_folder, config["mean"], config["std"], str(i), model))
    print("Validation keys: ", validation_indices)
    print("Dice: ", dice)
    if dice[0] != None:
        print("Average Dice: ", np.mean(dice))

if __name__ == "__main__":
    for fold in ["fold_01", "fold_02", "fold_03", "fold_04", "fold_05"]:
        # directories
        config["main_folder"] = os.path.abspath("/home/alzbeta/data/Ischemic_stroke_lesion/3DUnet_v2/output/cross_val_01/")
        config["output_foder"] = os.path.join(config["main_folder"], fold) # path to folder where the outputs should be saved
        config["data_folder"] = os.path.abspath("/home/alzbeta/data/Ischemic_stroke_lesion/PREPROCESSED/TRAINING-2/") #path to training data
        config["training_file"] = os.path.join(config["output_foder"], "training_ids.pkl")
        config["validation_file"] = os.path.join(config["output_foder"], "validation_ids.pkl")
        config["model_folder"] = os.path.join(config["output_foder"], "model")
        if not os.path.exists(config["model_folder"]):
            os.makedirs(config["model_folder"])
        config["model_file"] = os.path.join(config["model_folder"], "model.h5")
        config["wieghts_file_bestval"] = os.path.join(config["model_folder"], "BEST_val.h5")
        config["wieghts_file_lasttrain"] = os.path.join(config["model_folder"], "LAST_train.h5")
        config["wieghts_file"] = config["wieghts_file_lasttrain"]
        config["prediction_path"] = os.path.join(config["main_folder"], "prediction/")

        config["test_data"]=False # if true uses all test data and do not count dice index
        if config["test_data"]==True:
            config["data_folder"] = os.path.abspath("/home/alzbeta/data/Ischemic_stroke_lesion/PREPROCESSED/TESTING-2/")
            config["prediction_path"] = os.path.join(config["output_foder"], "prediction_test/")

        config["train_data"]=False # if true uses all training data to evaluate dice
        if config["train_data"]==True:
            config["prediction_path"] = os.path.join(config["output_foder"], "prediction_train/")
        main()
