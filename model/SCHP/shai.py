
import os
import cv2
import numpy as np
# import matplotlib.pyplot as plt
import json
import random



for root, dirs, files in os.walk("/home/chongzheng_p23/data/Datasets/UniFashion/YOOX/YOOX-Images"):
    for file in files:
        if file.endswith(".jpg"):
            source_file_path = os.path.join(root, file)
            print(source_file_path)
            save = root.replace("YOOX-Images","YOOX-SCHP")
            print(save)
            print(root)
            # img_name = source_file_path.split("/")[-1].split(".")[0]
            # print(img_name)