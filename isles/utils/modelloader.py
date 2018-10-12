from keras.models import load_model
from utils import weighted_dice_coefficient_loss

def ModelLoader(model_file):
    print("Loading pre-trained model")
    custom_objects = {'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss}
    try:
        from keras_contrib.layers import InstanceNormalization
        custom_objects["InstanceNormalization"] = InstanceNormalization
    except ImportError:
        pass
    try:
        return load_model(model_file, custom_objects=custom_objects)
    except ValueError as error:
        if 'InstanceNormalization' in str(error):
            raise ValueError(str(error) + "\n\nPlease install keras-contrib to use InstanceNormalization:\n"
                                          "'pip install git+https://www.github.com/keras-team/keras-contrib.git'")
        else:
            raise error
