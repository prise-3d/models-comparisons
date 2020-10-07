# main imports
import numpy as np
import pandas as pd
import sys, os, argparse

# image processing
from PIL import Image
from ipfml import utils
from ipfml.processing import transform, segmentation

import matplotlib.pyplot as plt

# model imports
import joblib

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
from modules.utils import data as dt

from data_attributes import get_image_features

zones_indices  = cfg.zones_indices

def write_progress(progress):
    barWidth = 180

    output_str = "["
    pos = barWidth * progress
    for i in range(barWidth):
        if i < pos:
           output_str = output_str + "="
        elif i == pos:
           output_str = output_str + ">"
        else:
            output_str = output_str + " "

    output_str = output_str + "] " + str(int(progress * 100.0)) + " %\r"
    print(output_str)
    sys.stdout.write("\033[F")

def main():

    parser = argparse.ArgumentParser(description="Read and compute entropy data file")

    parser.add_argument('--model', type=str, help='model file')
    parser.add_argument('--method', type=str, help='method name to used', choices=cfg.features_choices_labels, default=cfg.features_choices_labels[0])
    parser.add_argument('--interval', type=str, help='Interval value to keep from svd', default='"0, 200"')
    parser.add_argument('--kind', type=str, help='Kind of normalization level wished', choices=cfg.normalization_choices)
    parser.add_argument('--imnorm', type=int, help="specify if image is normalized before computing something", default=0, choices=[0, 1])
    parser.add_argument('--folder', type=str, help='folder where scene data are stored')
    parser.add_argument('--save', type=str, help='filename where to save input data')
    parser.add_argument('--label', type=str, help='label to use when saving thresholds')

    args = parser.parse_args()

    p_model    = args.model
    p_method   = args.method
    p_interval = list(map(int, args.interval.split(',')))
    #p_n_stop   = args.n_stop
    p_imnorm   = args.imnorm
    p_folder   = args.folder
    p_mode     = args.kind
    p_save     = args.save
    p_label    = args.label

    p_n_stop = 1
    begin, end = p_interval

    # 1. get scene name
    scene_path = p_folder

    # 2. load model and compile it

    # TODO : check kind of model
    model = joblib.load(p_model)
    # model.compile(loss='binary_crossentropy',
    #               optimizer='rmsprop',
    #               metrics=['accuracy'])


    estimated_thresholds = []
    n_estimated_thresholds = []
    zones_list = np.arange(16)

    # 4. get estimated thresholds using model and specific method
    images_path = sorted([os.path.join(scene_path, img) for img in os.listdir(scene_path) if cfg.scene_image_extension in img])
    number_of_images = len(images_path)
    image_indices = [ dt.get_scene_image_quality(img_path) for img_path in images_path ]

    image_counter = 0

    # append empty list
    for _ in zones_list:
        estimated_thresholds.append(None)
        n_estimated_thresholds.append(0)

    blocks_first_features = []

    for img_i, img_path in enumerate(images_path):

        blocks = segmentation.divide_in_blocks(Image.open(img_path), (200, 200))

        for index, block in enumerate(blocks):
            
            if estimated_thresholds[index] is None:
                # normalize if necessary
                if p_imnorm:
                    block = np.array(block) / 255.
                
                # check if prediction is possible
                data = np.array(get_image_features(p_method, np.array(block)))

                if p_mode == 'svdn':
                    data = utils.normalize_arr_with_range(data)

                data = data[begin:end]

                if img_i == 0:
                    blocks_first_features.append(data)
                else:
                    data = data - blocks_first_features[index]

                    #data = np.expand_dims(data, axis=0)
                    #print(data.shape)
                    
                    prob = model.predict(np.array(data).reshape(1, -1))[0]
                    #print(index, ':', image_indices[img_i], '=>', prob)

                    if prob < 0.5:
                        n_estimated_thresholds[index] += 1

                        # if same number of detection is attempted
                        if n_estimated_thresholds[index] >= p_n_stop:
                            estimated_thresholds[index] = image_indices[img_i]
                    else:
                        n_estimated_thresholds[index] = 0

        # write progress bar
        write_progress((image_counter + 1) / number_of_images)
        
        image_counter = image_counter + 1
    
    # default label
    for i, _ in enumerate(zones_list):
        if estimated_thresholds[i] == None:
            estimated_thresholds[i] = image_indices[-1]

    # 6. save estimated thresholds into specific file
    print(estimated_thresholds)
    print(p_save)
    if p_save is not None:
        with open(p_save, 'a') as f:
            f.write(p_label + ';')

            for t in estimated_thresholds:
                f.write(str(t) + ';')
            f.write('\n')
    

if __name__== "__main__":
    main()