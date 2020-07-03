import os
import numpy as np

dataset = 'dataset'
output_data = 'previous_thresholds'

scenes = os.listdir(dataset)
zones = np.arange(16)

for scene in scenes:
    
    scene_path = os.path.join(dataset, scene)

    scene_thresholds = []
    for zone in zones:

        index_str = str(zone)

        if len(index_str) < 2:
            index_str = '0' + index_str

        zone_name = 'zone' + index_str

        zone_path = os.path.join(scene_path, zone_name)

        threshold_file = os.path.join(zone_path, 'seuilExpe')

        with open(threshold_file, 'r') as f:
            threshold = int(f.readline())
            scene_thresholds.append(threshold)

    print(scene, scene_thresholds)

    with open(output_data, 'a') as f:
        
        f.write(scene + ';')
        for threshold in scene_thresholds:
            f.write(str(threshold) + ';')

        f.write('\n')



