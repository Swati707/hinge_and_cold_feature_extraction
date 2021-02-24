# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 22:23:46 2020

@author: swati
"""
from hinge_feature_extraction import *
from cold_feature_extraction import *
import argparse
import numpy as np
import os
from tqdm import tqdm

if __name__ == '__name__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input_folder", type=str, default=r"<IMAGES FOLDER IMAGES ARE STORED>")
    parser.add_argument("--output_folder", type=str, default=r"<FOLDER WHERE FEATURES ARE STORED")
    parser.add_argument("--sharpness_factor", type=int, default=10)
    parser.add_argument("--bordersize", type=int, default=3)
    parser.add_argument("--show_images", type=bool, default=False)
    parser.add_argument("--is_binary", type=bool, default=True)
    opt = parser.parse_args()
    
    input_folder = opt.input_folder
    output_folder = opt.output_folder
    
    class_dirs = os.listdir(input_folder)
    
    class_dirs.sort()
    print(class_dirs)

    hinge_feature_vectors = []
    cold_feature_vectors = []
    labels = []
    label_names = []
    ecount = 0
    
    cold = Cold(opt)
    hinge = Hinge(opt)
    
    for i, class_dir in enumerate(class_dirs):
        img_filenames = os.listdir(os.path.join(input_folder, class_dir))
        for img_filename in tqdm(img_filenames):
            try:
                img_path = os.path.join(input_folder, class_dir, img_filename)
                
                h_f = hinge.get_hinge_features(img_path)
                c_f = cold.get_cold_features(img_path)
                
                hinge_feature_vectors.append(h_f)
                cold_feature_vectors.append(c_f)
                label_names.append(class_dir)
                labels.append(i)
            except Exception as inst:
                ecount += 1
                if ecount % 20 == 0:
                    print(inst, f'error count: {ecount}')
                continue
        
        print(f"[STATUS] processed folder: {class_dir}")
        
    np.save(os.path.join(output_folder, f"hinge_features.npy"), hinge_feature_vectors)
    np.save(os.path.join(output_folder, f"cold_features.npy"), cold_feature_vectors)
    np.savez(os.path.join(output_folder, f"labels"), label = labels, label_name = label_names)
    
    print(f"Saved all hinge and cold features")
           



