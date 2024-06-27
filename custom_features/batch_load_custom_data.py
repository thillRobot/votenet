# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Batch mode in loading Scannet scenes with vertices and ground truth labels
for semantic and instance segmentations

Usage example: python ./batch_load_custom_data.py
"""
import os
import sys
import datetime
import numpy as np
from load_custom_data import export
import pdb

CUSTOM_DIR = 'CustomFeatures'
TRAIN_SCAN_NAMES = [line.rstrip() for line in open(os.path.join(CUSTOM_DIR,'custom_train.txt'))]
VAL_SCAN_NAMES   = [line.rstrip() for line in open(os.path.join(CUSTOM_DIR,'custom_val.txt'))]
TEST_SCAN_NAMES  = [line.rstrip() for line in open(os.path.join(CUSTOM_DIR,'custom_test.txt'))]
TRAIN_SCAN_NAMES = TRAIN_SCAN_NAMES+VAL_SCAN_NAMES+TEST_SCAN_NAMES # added by TH
#LABEL_MAP_FILE = 'meta_data/custom-labels.combined.tsv'
DONOTCARE_CLASS_IDS = np.array([])
OBJ_CLASS_IDS = np.array([0,1,2])
MAX_NUM_POINT = 50000
OUTPUT_FOLDER = './CustomFeatures/data'

def export_one_scan(scan_name, output_filename_prefix):    
    #mesh_file = os.path.join(CUSTOM_DIR, scan_name, scan_name + '_vh_clean_2.ply')
    part_type = scan_name.split('_')[1]

    pcd_file = os.path.join(CUSTOM_DIR,'pcds', part_type, scan_name + '.pcd')
    agg_file = os.path.join(CUSTOM_DIR,'labels', part_type, scan_name + '.aggregation.json')
    seg_file = os.path.join(CUSTOM_DIR,'labels', part_type, scan_name + '.segs.json')
    box_file = os.path.join(CUSTOM_DIR,'labels', part_type, scan_name + '.boxes.txt')

    pcd_vertices, semantic_labels, instance_labels, instance_bboxes, instance2semantic = \
        export(pcd_file, agg_file, seg_file, box_file, None)
       
    mask = np.logical_not(np.in1d(semantic_labels, DONOTCARE_CLASS_IDS))
    pcd_vertices = np.asarray(pcd_vertices)[mask,:]
    semantic_labels = semantic_labels[mask]
    instance_labels = instance_labels[mask]

    num_instances = len(np.unique(instance_labels))
    #print('Num of instances: ', num_instances)

    bbox_mask = np.in1d(instance_bboxes[:,-1], OBJ_CLASS_IDS)
    instance_bboxes = instance_bboxes[bbox_mask,:]
    #print('Num of care instances: ', instance_bboxes.shape[0])

    N = pcd_vertices.shape[0]
    if N > MAX_NUM_POINT:
        choices = np.random.choice(N, MAX_NUM_POINT, replace=False)
        pcd_vertices = pcd_vertices[choices, :]
        semantic_labels = semantic_labels[choices]
        instance_labels = instance_labels[choices]

    np.save(output_filename_prefix+'_vert.npy', pcd_vertices)
    np.save(output_filename_prefix+'_sem_label.npy', semantic_labels)
    np.save(output_filename_prefix+'_ins_label.npy', instance_labels)
    np.save(output_filename_prefix+'_bbox.npy', instance_bboxes)
    #print('output vertices, semantic labels, instance labels, and instance boxes saved as .npy files')    

def batch_export():
    if not os.path.exists(OUTPUT_FOLDER):
        print('Creating new data folder: {}'.format(OUTPUT_FOLDER))                
        os.mkdir(OUTPUT_FOLDER)        
        
    for scan_name in TRAIN_SCAN_NAMES:
        print('-'*20+'begin')
        print(datetime.datetime.now())
        print(scan_name)
        output_filename_prefix = os.path.join(OUTPUT_FOLDER, scan_name) 
        if os.path.isfile(output_filename_prefix+'_vert.npy'):
            print('File already exists. skipping.')
            print('-'*20+'done')
            continue
        try:            
            export_one_scan(scan_name, output_filename_prefix)
        except Exception as e: 
            print(e)
            print('Failed export scan: %s'%(scan_name))            
        print('-'*20+'done')

if __name__=='__main__':    
    batch_export()
