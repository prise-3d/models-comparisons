# main imports
import numpy as np
import pandas as pd
import sys, os, argparse
import math

# image processing
from PIL import Image
from ipfml import utils
from ipfml.processing import transform, segmentation

import matplotlib.pyplot as plt

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg


# utils information
zone_width, zone_height = (200, 200)
scene_width, scene_height = (800, 800)
nb_x_parts = math.floor(scene_width / zone_width)
zones_indices  = cfg.zones_indices

def extract_thresholds_from_file(filename):
    # extract thresholds
    thresholds_dict = {}
    with open(filename) as f:
        thresholds_line = f.readlines()

        for line in thresholds_line:
            data = line.split(';')
            del data[-1] # remove unused last element `\n`
            current_scene = data[0]
            thresholds_scene = data[1:]

            thresholds_dict[current_scene] = [ int(threshold) for threshold in  thresholds_scene ]

    return thresholds_dict

def reconstruct_image(images_paths, thresholds, output, step):

    # to avoid issue
    images_paths = sorted(images_paths)

    # 1. compute zone start index
    zones_coordinates = []
    for zone_index in cfg.zones_indices:
        x_zone = (zone_index % nb_x_parts) * zone_width
        y_zone = (math.floor(zone_index / nb_x_parts)) * zone_height

        zones_coordinates.append((x_zone, y_zone))

    images_zones = []
    line_images_zones = []

    # 2. get image using threshold by zone
    for id, t in enumerate(thresholds):

        t_index = int(t / step) - 1 # -1 because index start at 0 in image_paths list
        image_path = images_paths[t_index]
        selected_image = Image.open(image_path)

        x_zone, y_zone = zones_coordinates[id]
        zone_image = np.array(selected_image)[y_zone:y_zone+zone_height, x_zone:x_zone+zone_width]
        line_images_zones.append(zone_image)

        if int(id + 1) % int(scene_width / zone_width) == 0:
            images_zones.append(np.concatenate(line_images_zones, axis=1))
            line_images_zones = []


    # 3. reconstructed the image using these zones
    reconstructed_image = np.concatenate(images_zones, axis=0)

    # 4. Save the image with generated name based on scene
    reconstructed_pil_img = Image.fromarray(reconstructed_image)

    reconstructed_pil_img.save(output)
    
    

def main():

    parser = argparse.ArgumentParser(description="Read and display simulation from multiple models")

    parser.add_argument('--folder', type=str, help='dataset path with scenes images files', required=True)
    parser.add_argument('--thresholds', type=str, help='file with specific thresholds (using only scene from this file', required=True)
    parser.add_argument('--output', type=str, help='output folder with figures for each scene', required=True)

    args = parser.parse_args()

    p_folder = args.folder
    p_thresholds = args.thresholds
    p_output = args.output

    if not os.path.exists(p_output):
        os.makedirs(p_output)

    # 1. retrieve human_thresholds
    human_thresholds = extract_thresholds_from_file(p_thresholds)

    for scene in sorted(os.listdir(p_folder)):
        
        # get all estimated thresholds for this scene
        scene_path = os.path.join(p_folder, scene)
        images_path = sorted([ os.path.join(scene_path, img) for img in os.listdir(scene_path) ])

        # create image with these information
        if scene in human_thresholds:
            thresholds = human_thresholds[scene]
            output_image = os.path.join(p_output, scene + '.png')
            reconstruct_image(images_path, thresholds, output_image, step=20)
            print('Image reconstructed save at {0}'.format(output_image))


if __name__== "__main__":
    main()