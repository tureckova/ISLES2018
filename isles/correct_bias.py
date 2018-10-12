import glob
import os
import warnings
import shutil

import SimpleITK as sitk
import numpy as np
from nipype.interfaces.ants import N4BiasFieldCorrection


def correct_bias(in_file, out_file, image_type=sitk.sitkFloat64):
    """
    Corrects the bias using ANTs N4BiasFieldCorrection. If this fails, will then attempt to correct bias using SimpleITK
    :param in_file: input file path
    :param out_file: output file path
    :return: file path to the bias corrected image
    """
    correct = N4BiasFieldCorrection()
    correct.inputs.input_image = in_file
    correct.inputs.output_image = out_file
    correct.inputs.shrink_factor = 0
    try:
        done = correct.run()
        return done.outputs.output_image
    except IOError:
        warnings.warn(RuntimeWarning("ANTs N4BIasFieldCorrection could not be found."
                                     "Will try using SimpleITK for bias field correction"
                                     " which will take much longer. To fix this problem, add N4BiasFieldCorrection"
                                     " to your PATH system variable. (example: EXPORT PATH=${PATH}:/path/to/ants/bin)"))
        input_image = sitk.ReadImage(in_file, image_type)
        output_image = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
        sitk.WriteImage(output_image, out_file)
        return os.path.abspath(out_file)

def get_image(subject_folder, name):
    file_card = os.path.join(subject_folder, "*" + name + "*.nii")
    try:
        return glob.glob(file_card)[0]
    except IndexError:
        return None
        #raise RuntimeError("Could not find file matching {}".format(file_card))

def normalize_image(in_file, out_file, bias_correction=True):
    if bias_correction:
        correct_bias(in_file, out_file)
    else:
        shutil.copy(in_file, out_file)
    return out_file

def convert_ISLES_folder(in_folder, out_folder, modalities,
                         no_bias_correction_modalities=None):
    for name in modalities:
        image_file = get_image(in_folder, name)
        if image_file is not None:
            out_file = os.path.abspath(os.path.join(out_folder, image_file.split('/')[-1]))
            perform_bias_correction = no_bias_correction_modalities and name not in no_bias_correction_modalities
            normalize_image(image_file, out_file, bias_correction=perform_bias_correction)
            print("File: ", image_file, " DONE")


def convert_ISLES_data(ISLES_folder, out_folder, overwrite=False, modalities=["CT.", "CBF.", "CBV.", "MTT.", "Tmax.", "OT."], no_bias_correction_modalities=("OT.",)):
    """
    Preprocesses the ISLES data and writes it to a given output folder. Assumes the original folder structure.
    :param ISLES_folder: folder containing the original ISLES data
    :param out_folder: output folder to which the preprocessed data will be written
    :param overwrite: set to True in order to redo all the preprocessing
    :param modalities: all modalities to process
    :param no_bias_correction_modalities: performing bias correction could reduce the signal of certain modalities. If
    concerned about a reduction in signal for a specific modality, specify by including the given modality in a list
    or tuple.
    :return:
    """
    for subject_folder in glob.glob(os.path.join(ISLES_folder, "*")):
        if os.path.isdir(subject_folder):
            subject = os.path.basename(subject_folder)
            new_subject_folder = os.path.join(out_folder, os.path.basename(os.path.dirname(subject_folder)),
                                              subject)
            if not os.path.exists(new_subject_folder) or overwrite:
                if not os.path.exists(new_subject_folder):
                    os.makedirs(new_subject_folder)
                convert_ISLES_folder(subject_folder, new_subject_folder, modalities,
                                     no_bias_correction_modalities=no_bias_correction_modalities)

convert_ISLES_data("/home/alzbeta/data/Ischemic_stroke_lesion/TESTING-2/", "/home/alzbeta/data/Ischemic_stroke_lesion/PREPROCESSED/")
