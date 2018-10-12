import numpy as np
import os
import glob
from utils import NiiImagesLoader

config = dict()
config["data_folder"] = os.path.abspath("/home/alzbeta/data/Ischemic_stroke_lesion/TRAINING-2/") #path to folder containing the training data
config["all_modalities"] = ["CT", "CBF", "CBV", "MTT", "Tmax"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities

def normalize_data(data, mean, std):
    data -= mean[:, np.newaxis, np.newaxis, np.newaxis]
    data /= std[:, np.newaxis, np.newaxis, np.newaxis]
    return data

def get_normalization(data_storage):
    means = list()
    stds = list()
    for index in range(len(data_storage)):
        data = data_storage[index]
        means.append(data.mean(axis=(1, 2, 3)))
        stds.append(data.std(axis=(1, 2, 3)))
    mean = np.asarray(means).mean(axis=0)
    std = np.asarray(stds).mean(axis=0)
    return mean, std

def main():
    data = []
    # load data
    for subject_folder in glob.glob(os.path.join(config["data_folder"], "*")):
        dataIm, labels = NiiImagesLoader().load(subject_folder, config["training_modalities"])
        data.append(dataIm)
    mean, std = get_normalization(data)
    print("Mean: ", mean)
    print("STD: ", std)


if __name__ == "__main__":
    main()
