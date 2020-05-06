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


def display_thresholds_comparisons(scene, thresholds, zones_learned, y_lim):
    
    colors = ['C0', 'C1', 'C2', 'C3']
    
    plt.figure(figsize=(25, 20))
    plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
    
    # display each thresholds data from file comparisons
    for i in thresholds:
        data = thresholds[i]
        plt.plot(data, label=i)
    
    plt.xticks(zones_indices)
    
    if zones_learned:

        for i in cfg.zones_indices:
            if i in zones_learned:
                
                plt.plot([i, i], [y_lim[0], y_lim[1]], '--', color='grey', alpha=0.3)
                plt.gca().get_xticklabels()[i].set_color('gray')
            else:
                plt.plot([i, i], [y_lim[0], y_lim[1]], '-.', color='red', alpha=0.3)
                plt.gca().get_xticklabels()[i].set_color('red')


    plt.title('Comparisons of estimated thresholds for ' + scene, fontsize=22)
    plt.legend(fontsize=20)
    plt.ylim(y_lim[0], y_lim[1])


    plt.show()

def main():

    parser = argparse.ArgumentParser(description="Read and compute entropy data file")

    parser.add_argument('--thresholds', type=str, help='thresholds model data to compare')
    parser.add_argument('--learned_zones', type=str, help='learned zones file')
    parser.add_argument('--scene', type=str, help='Scene index to use', choices=cfg.scenes_indices)

    args = parser.parse_args()

    p_thresholds = args.thresholds
    p_learned_zones = args.learned_zones
    p_scene  = args.scene

    scenes_list = cfg.scenes_names
    scenes_indices = cfg.scenes_indices

    scene_index = scenes_indices.index(p_scene.strip())
    scene = scenes_list[scene_index]
    scene_path = os.path.join(cfg.dataset_path, scene)

    # get y lim
    images_indices = [ int(img.split('_')[-1].replace('.png', ''))
                for img in sorted(os.listdir(scene_path)) 
                if cfg.scene_image_extension in img ]

    y_lim = images_indices[0], images_indices[-1]

    human_thresholds = []
    estimated_thresholds = {}

    # 3. retrieve human_thresholds
    # construct zones folder
    zones_list = []

    for index in zones_indices:

        index_str = str(index)

        while len(index_str) < 2:
            index_str = "0" + index_str
        
        zones_list.append(cfg.zone_folder + index_str)

    for zone in zones_list:
            zone_path = os.path.join(scene_path, zone)

            with open(os.path.join(zone_path, cfg.seuil_expe_filename), 'r') as f:
                human_thresholds.append(int(f.readline()))

    estimated_thresholds['Ground truth'] = human_thresholds

    # read line by line file to estimate threshold entropy stopping criteria
    with open(p_thresholds, 'r') as f:
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

                if data[0] == scene:
                    zones_learned = [ int(d) for d in data[1:] ]

    display_thresholds_comparisons(scene, estimated_thresholds, zones_learned, y_lim)

if __name__== "__main__":
    main()
