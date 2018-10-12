# import the necessary packages
import numpy as np
import cv2
import os
import glob
import nibabel as nib
import numpy as np

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        # store the image preprocessor
        self.preprocessors = preprocessors

        # if the preprocessors are None, initialize them as an
        # empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        # initialize the list of features and labels
        data = []
        labels = []

        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract the class label assuming
            # that our path has the following format:
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            # check to see if our preprocessors are not None
            if self.preprocessors is not None:
                # loop over the preprocessors and apply each to
                # the image
                for p in self.preprocessors:
                    image = p.preprocess(image)

            # treat our processed image as a "feature vector"
            # by updating the data list followed by the labels
            data.append(image)
            labels.append(label)

            # show an update every `verbose` images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1,
                    len(imagePaths)))

        # return a tuple of the data and labels
        return (np.array(data), np.array(labels))

class NiiDatasetLoader:
    def __init__(self, preprocessors=None):
        # store the image preprocessor
        self.preprocessors = preprocessors

        # if the preprocessors are None, initialize them as an
        # empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, data_folder, modalities):
        # initialize the list of features and labels
        data = []
        labels = []
        for folder in sorted(glob.glob(os.path.join(data_folder, "*"))):
            # show an update info
            print('[INFO] Processing foder: ' + folder)
            img = []
            for modality in modalities:
                filename = glob.glob(os.path.join(folder, '*%s*.nii' % modality))[0]
                image = nib.load(filename).get_fdata()
                # check to see if our preprocessors are not None
                if self.preprocessors is not None:
                    # loop over the preprocessors and apply each to the image
                    for p in self.preprocessors:
                        image = p.preprocess(image)
                img.append(image)
            data.append(np.array(img))
            filename = glob.glob(os.path.join(folder, '*OT*.nii'))
            if filename == []:
                label = None
            else:
                label= nib.load(filename[0]).get_fdata()
                # check to see if our preprocessors are not None
                if self.preprocessors is not None:
                    # loop over the preprocessors and apply each to the image
                    for p in self.preprocessors:
                        label = p.preprocess(label)
                label = np.expand_dims(label, axis=0)
            labels.append(np.array(label))
        return (data, labels)

class NiiImagesLoader:
    def __init__(self, preprocessors=None):
        # store the image preprocessor
        self.preprocessors = preprocessors

        # if the preprocessors are None, initialize them as an
        # empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, folder, modalities):
        # show an update info
        print('[INFO] Processing foder: ' + folder)
        # initialize the list of features and labels
        data = []
        images = [] 
        for modality in modalities:
            filename = glob.glob(os.path.join(folder, '*%s*.nii' % modality))[0]
            img = nib.load(filename).get_fdata()
            # check to see if our preprocessors are not None
            if self.preprocessors is not None:
                # loop over the preprocessors and apply each to the image
                for p in self.preprocessors:
                    img = p.preprocess(img)
            data.append(img)
        filename = glob.glob(os.path.join(folder, '*OT*.nii'))
        if filename == []:
            label = None
        else:
            label= nib.load(filename[0]).get_fdata()
        return (np.array(data), label)

