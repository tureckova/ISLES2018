import os
config = dict()
config["output_foder"] = os.path.abspath("/home/alzbeta/data/Ischemic_stroke_lesion/3DUnet_v2/output/EXP12/")  # outputs path
config["model_file"] = os.path.join(config["output_foder"], "model.h5")
config["wieghts_file"] = os.path.join(config["output_foder"], "model_weights.h5")

config["input_size"] = (256,256) # size of image (x,y)
config["nr_slices"] = (32,) # all inputs will be resliced to this number of slices (z axis), must be power of 2
config["all_modalities"] = ["CT", "CBF", "CBV", "MTT", "Tmax"]#["CBF", "CBV", "MTT", "Tmax"]
config["training_modalities"] = config["all_modalities"] 

def normalize_data(data, mean, std):
    for i in range(data.shape[0]):
        data[i] -= mean[i]
        data[i] /= std[i]
    return data 

mean = [144.85851701, 49.06570323, 6.42394785, 2.36531961, 1.16228806]
std = [530.26314967, 127.95098693, 16.96937125, 4.24672666, 3.35688005]

from utils import NiiImagesLoader, Resize, ModelLoader
import nibabel as nib
prediction = nib.load("/home/alzbeta/data/Ischemic_stroke_lesion/3DUnet_v2/output/EXP12/prediction_train/SMIR.predictionEXP12_case58.345952.nii").get_fdata()
subject_folder = "/home/alzbeta/data/Ischemic_stroke_lesion/PREPROCESSED/TRAINING-2/case_58/"
data, label = NiiImagesLoader().load(subject_folder, config["training_modalities"]) # load data
data = normalize_data(data, mean, std)
import cv2
cv2.imwrite("prediction.png", cv2.rotate(prediction[:,:,0]*255, rotateCode = 2))

for i, modality in enumerate(config["all_modalities"]):
    cv2.imwrite(modality + ".png", cv2.rotate(data[i,:,:,0]*255, rotateCode = 2))

cv2.imwrite("GT.png", cv2.rotate(label[:,:,0]*255, rotateCode = 2))
