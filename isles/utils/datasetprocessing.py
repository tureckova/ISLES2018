import shutil
from .sitk_utils import resample_to_spacing, calculate_origin_offset
import numpy as np

class Resize:
    def __init__(self, new_shape,  interpolation="linear"):
        # inputs:
        # --------------
        # new_shape - new  shape of the output image, in tuple
        # interpolation - used interpolation, either "linear" ot "nearest" (the second is suitable for label map)
        self.new_shape = new_shape
        self.interpolation = interpolation
    def preprocess(self, image,):
        # inputs:
        # --------------
        # image - image to be resized (np.array, 3 dims: x,y,z)
        zoom_level = np.divide(self.new_shape, image.shape)
        new_spacing = np.divide((1.,1.,1.), zoom_level)
        new_data = resample_to_spacing(image, (1.,1.,1.), new_spacing, interpolation=self.interpolation)
        return new_data
