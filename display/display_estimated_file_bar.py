# main imports
import numpy as np
import pandas as pd
import sys, os, argparse

# image processing
from PIL import Image
from ipfml import utils
from ipfml.processing import transform, segmentation

import matplotlib.pyplot as plt

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg

dataset_folder = cfg.dataset_path
scenes_list    = cfg.scenes_names
zones_indices  = cfg.zones_indices


def display_thresholds_comparisons(scene, thresholds_file, thresholds, zones_learned, y_lim):
    
    colors = ['C0', 'C1', 'C2', 'C3']
    
    plt.figure(figsize=(25, 20))
    plt.rc('xtick', labelsize=22)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=22)    # fontsize of the tick labels
    
    # display each thresholds data from file comparisons
    for index, i in enumerate(thresholds):
        
        data = thresholds[i]

        if index == 0:
            plt.bar(data, lw=6, label=i, color='tab:red')
        else:
            plt.bar(data, lw=3, label=i)
    
    plt.xticks(zones_indices, label=[ str(i) for i in (zones_indices + 1) ])
    
    if zones_learned:

        for i in cfg.zones_indices:
            if i in zones_learned:
                
                # plt.bar([i, i], [y_lim[0], y_lim[1]], '--', color='black', alpha=0.5)
                plt.gca().get_xticklabels()[i].set_color('black')
            else:
                # plt.bar([i, i], [y_lim[0], y_lim[1]], '-.', color='red', alpha=0.5)
                plt.gca().get_xticklabels()[i].set_color('red')


    plt.title('Comparisons of estimated thresholds for ' + scene, fontsize=30)
    plt.legend(fontsize=26)
    plt.xlabel('Image zone indices', fontsize=28)
    plt.ylabel('Number of samples', fontsize=28)
    #plt.tick_params(labelsize=24)
    plt.ylim(y_lim[0], y_lim[1])

    plt.savefig(thresholds_file + '_bar', transparent=True)

def main():

    parser = argparse.ArgumentParser(description="Read and compute entropy data file")

    parser.add_argument('--simulation', type=str, help='obtained thresholds model data to compare', required=True)
    parser.add_argument('--learned_zones', type=str, help='learned zones file', required=True)
    parser.add_argument('--scene', type=str, help='scene path', required=True)
    parser.add_argument('--thresholds', type=str, help='file with specific thresholds (using only scene from this file', required=True)

    args = parser.parse_args()

    p_simulation = args.simulation
    p_learned_zones = args.learned_zones
    p_scene  = args.scene
    p_thresholds = args.thresholds

    # get y lim
    images_indices = [ int(img.split('_')[-1].replace('.png', ''))
                for img in sorted(os.listdir(p_scene)) 
                if cfg.scene_image_extension in img ]

    y_lim = images_indices[0], images_indices[-1]

        # 1. retrieve human_thresholds
    human_thresholds = {}

    # extract thresholds
    with open(p_thresholds) as f:
        thresholds_line = f.readlines()

        for line in thresholds_line:
            data = line.split(';')
            del data[-1] # remove unused last element `\n`
            current_scene = data[0]
            thresholds_scene = data[1:]

            # TODO : check if really necessary
            if current_scene != '50_shades_of_grey':
                human_thresholds[current_scene] = [ int(threshold) for threshold in  thresholds_scene ]

    _, scene_name = os.path.split(p_scene)

    # get only necessary thresholds
    current_human_thresholds = human_thresholds[scene_name]

    # get all estimated thresholds
    estimated_thresholds = {}
    estimated_thresholds['Ground truth'] = current_human_thresholds

    # read line by line file to estimate threshold entropy stopping criteria
    with open(p_simulation, 'r') as f:
        lines = f.readlines()

        for line in lines:

            data = line.split(';')

            del data[-1]

            method_name = data[0]
            estimated_thresholds[method_name] = [int(d) for d in data[1:] ]
            
    # get trained zones used
    # 5. check if learned zones
    zones_learned = None

    if p_learned_zones is not None:
        with open(p_learned_zones, 'r') as f:
            lines = f.readlines()

            for line in lines:
                data = line.split(';')
                del data[-1]

                if data[0] == scene_name:
                    zones_learned = [ int(d) for d in data[1:] ]

    display_thresholds_comparisons(scene_name, p_simulation, estimated_thresholds, zones_learned, y_lim)

if __name__== "__main__":
    main()