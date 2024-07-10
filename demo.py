# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Demo of using VoteNet 3D object detector to detect objects from a point cloud.
"""

import os
import sys
import numpy as np
import argparse
import importlib
import time
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sunrgbd', help='Dataset: sunrgbd or scannet [default: sunrgbd]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--input_file', default='input_pc_custom_features.pcd')
parser.add_argument('--input_dir', default='demo_files')
parser.add_argument('--show_results', type=bool, default=True)

FLAGS = parser.parse_args()

import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pc_util import random_sampling, read_ply, read_pcd
from ap_helper import parse_predictions

import open3d as o3d

def preprocess_point_cloud(point_cloud):
    ''' Prepare the numpy point cloud (N,3) for forward pass '''
    point_cloud = point_cloud[:,0:3] # do not use color for now
    floor_height = np.percentile(point_cloud[:,2],0.99)
    height = point_cloud[:,2] - floor_height
    point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)
    point_cloud = random_sampling(point_cloud, FLAGS.num_point)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0) # (1,40000,4)
    return pc

if __name__=='__main__':
    
    # Set file paths and dataset config
    demo_dir = os.path.join(BASE_DIR, FLAGS.input_dir) 
    if FLAGS.dataset == 'sunrgbd':
        sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
        from sunrgbd_detection_dataset import DC # dataset config
        checkpoint_path = os.path.join(demo_dir, 'pretrained_votenet_on_sunrgbd.tar')
        pc_path = os.path.join(demo_dir, 'input_pc_sunrgbd.ply')
    elif FLAGS.dataset == 'scannet':
        sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
        from scannet_detection_dataset import DC # dataset config
        checkpoint_path = os.path.join(demo_dir, 'pretrained_votenet_on_scannet.tar')
        pc_path = os.path.join(demo_dir, 'input_pc_scannet.ply')
    elif FLAGS.dataset == 'custom':
        sys.path.append(os.path.join(ROOT_DIR, 'custom_features'))
        from custom_features_dataset import DC # dataset config
        checkpoint_path = os.path.join(demo_dir, 'pretrained_votenet_on_custom_features.tar')
        pc_path = os.path.join(demo_dir, FLAGS.input_file)
        DC.show_results=FLAGS.show_results
    else:
        print('Unkown dataset %s. Exiting.'%(DATASET))
        exit(-1)

    eval_config_dict = {'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
        'use_old_type_nms': False, 'cls_nms': False, 'per_class_proposal': False,
        'conf_thresh': 0.5, 'dataset_config': DC}
    
    # Init the model and optimzier
    MODEL = importlib.import_module('votenet') # import network module
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = MODEL.VoteNet(num_proposal=256, input_feature_dim=1, vote_factor=1,
        sampling='seed_fps', num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr).to(device)
    print('Constructed model.')

    # use alternate checkpoint path if provided
    if FLAGS.checkpoint_path is not None and os.path.isfile(FLAGS.checkpoint_path):
        checkpoint_path=FLAGS.checkpoint_path
    checkpoint = torch.load(checkpoint_path)  
    
    # Load checkpoint
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    print("Loaded checkpoint %s (epoch: %d)"%(checkpoint_path, epoch))
   
    # Load and preprocess input point cloud 
    net.eval() # set model to eval mode (for bn and dp)
    if pc_path.endswith('.ply'):
        point_cloud = read_ply(pc_path)
    elif pc_path.endswith('.pcd'):
        point_cloud = read_pcd(pc_path)

    print('Loaded point cloud data from: %s'%(pc_path))

    pc = preprocess_point_cloud(point_cloud)

    # Model inference
    inputs = {'point_clouds': torch.from_numpy(pc).to(device)}
    tic = time.time()
    with torch.no_grad():
        end_points = net(inputs)
    toc = time.time()
    print('Inference time: %f'%(toc-tic))
    end_points['point_clouds'] = inputs['point_clouds']
    pred_map_cls = parse_predictions(end_points, eval_config_dict)

    # show the predictions in the terminal
    num_objects=len(pred_map_cls[0]) # number of detected objects 
    print('Finished detection. %d object detected.'%num_objects)
 
    for pred_cls in pred_map_cls[0]:
        print('pred_cls: %d, %s'%(pred_cls[0], DC.class2type[pred_cls[0]]))
    #print('end_points keys:', end_points.keys())

    #print('end_points sem_cls_scores:', end_points['sem_cls_scores'][0,:,:])
    #print('sem_cls_scores shape:', np.asarray(end_points['sem_cls_scores'].cpu()).shape )

    #print('end_points heading_scores:', end_points['heading_scores'][0,:,:])
    #print('heading_scores shape:', np.asarray(end_points['heading_scores'].cpu()).shape )

    dump_dir = os.path.join(demo_dir, '%s_results'%(FLAGS.dataset))
    if not os.path.exists(dump_dir): os.mkdir(dump_dir) 
    MODEL.dump_results(end_points, dump_dir, DC, True)
    print('Dumped detection results to folder %s'%(dump_dir))

    # show the results in an figure window

    # show the input pointcloud in grey
    origin_base = o3d.geometry.TriangleMesh.create_coordinate_frame()
    origin=copy.deepcopy(origin_base).scale(0.5, center=(0,0,0))

    fpath = os.path.join(dump_dir,'000000_pc.ply')
    pcd = o3d.io.read_point_cloud(fpath)
    pcd.paint_uniform_color((.3, .3, .3))
    display_items=[origin, pcd]
    # add additional items to show to this list
    display_results=['000000_pred_confident_nms_bbox.ply']
    
    for result in display_results:

        fpath = os.path.join(dump_dir,result)
        pcd = o3d.io.read_point_cloud(fpath)
        mesh = o3d.io.read_triangle_mesh(fpath)
        pcd.paint_uniform_color((1, .1, .1))
        print(f"Pointcloud loaded pointcloud from: {fpath}")
        display_items.append(pcd)
        display_items.append(mesh)

    o3d.visualization.draw_geometries(display_items)
