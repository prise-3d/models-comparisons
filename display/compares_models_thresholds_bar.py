# main imports
import numpy as np
import pandas as pd
import sys, os, argparse

# image processing
from PIL import Image
from ipfml import utils
from ipfml.processing import transform, segmentation

import matplotlib.pyplot as plt
import seaborn as sns

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg


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

def display_thresholds_comparisons(scene, output, thresholds, zones_learned, y_lim, human_available):
    
    colors = ['C0', 'C1', 'C2', 'C3']
    
    fig = plt.figure(figsize=(25, 20))
    plt.rc('xtick', labelsize=22)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=22)    # fontsize of the tick labels
    ax = fig.add_axes([0,0,1,1])
    
    width = 0.35
    # display each thresholds data from file comparisons
    for index, i in enumerate(thresholds):
        
        data = thresholds[i]

        if human_available and index == 0:
            #ax.bar(cfg.zones_indices, data, lw=6, label=i, color='tab:red')
            ax.bar(cfg.zones_indices + width * index, data, width, label=i, color='tab:red')
        else:
            ax.bar(cfg.zones_indices + width * index, data, width, label=i)
    
    ax.set_xticks(zones_indices)
    ax.set_xticklabels([ str(i + 1) for i in (zones_indices) ])
    
    if zones_learned:

        for i in cfg.zones_indices:
            if i in zones_learned:
                
                plt.plot([i, i], [y_lim[0], y_lim[1]], '--', color='black', alpha=0.5)
                plt.gca().get_xticklabels()[i].set_color('black')
            else:
                plt.plot([i, i], [y_lim[0], y_lim[1]], '-.', color='red', alpha=0.5)
                plt.gca().get_xticklabels()[i].set_color('red')


    ax.set_title('Comparisons of estimated thresholds for ' + scene, fontsize=30)
    ax.legend(fontsize=26)
    ax.set_xlabel('Image zone indices', fontsize=28)
    ax.set_ylabel('Number of samples', fontsize=28)
    #plt.tick_params(labelsize=24)
    ax.set_ylim(y_lim[0], y_lim[1])

    plt.savefig(output, transparent=True)

def main():

    parser = argparse.ArgumentParser(description="Read and display simulation from multiple models")

    parser.add_argument('--folder', type=str, help='dataset path with scenes images files', required=True)
    parser.add_argument('--data', type=str, help='folder with obtained thresholds model data to compare', required=True)
    parser.add_argument('--learned_zones', type=str, help='learned zones file', required=True)
    parser.add_argument('--thresholds', type=str, help='file with specific thresholds (using only scene from this file', required=True)
    parser.add_argument('--labels', type=str, help='labels list to display in figure', required=True)
    parser.add_argument('--output', type=str, help='output folder with figures for each scene', required=True)

    args = parser.parse_args()

    p_folder = args.folder
    p_data = args.data
    p_learned_zones = args.learned_zones
    p_thresholds = args.thresholds
    p_labels = args.labels.split(',')
    p_output = args.output

    if not os.path.exists(p_output):
        os.makedirs(p_output)

    # get y lim
    y_lim = 0, 10000

    # 1. retrieve human_thresholds
    human_thresholds = extract_thresholds_from_file(p_thresholds)

    # 2. Extract common learned zones for each scene
    zones_learned = None

    if p_learned_zones is not None:

        zones_learned = {}

        with open(p_learned_zones, 'r') as f:
            lines = f.readlines()

            for line in lines:
                data = line.split(';')
                del data[-1]

                zones_learned[data[0]] = [ int(d) for d in data[1:] ]

    # 3. extract models thresholds
    models_thresholds = {}

    for i, simu in enumerate(sorted(os.listdir(p_data))):
        method_label = p_labels[i]
        simu_path = os.path.join(p_data, simu)
        models_thresholds[method_label] = extract_thresholds_from_file(simu_path)

    # 4. extract information for each expected scene

    for scene in sorted(os.listdir(p_folder)):
        
        # get all estimated thresholds for this scene
        estimated_thresholds = {}

        # get only necessary thresholds
        if scene in human_thresholds:   
            current_human_thresholds = human_thresholds[scene]
            estimated_thresholds['Ground truth'] = current_human_thresholds

        for method_name in p_labels:
            estimated_thresholds[method_name] = models_thresholds[method_name][scene]

        human_enable = False
        if 'Ground truth' in estimated_thresholds:
            human_enable = True    

        output_figure = os.path.join(p_output, scene + '.png')

        scenes_zones = zones_learned[scene] if scene in zones_learned else None

        # create figure with these information
        display_thresholds_comparisons(scene, output_figure, estimated_thresholds, scenes_zones, y_lim, human_enable)
        print('Figure is saved at {0}'.format(output_figure))

if __name__== "__main__":
    main()