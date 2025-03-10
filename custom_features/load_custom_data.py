# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Load CustomFeatures scenes with vertices and ground truth labels
for semantic and instance segmentations
"""

# python imports
import math
import os, sys, argparse
import inspect
import json
import pdb

try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
import custom_utils

from model_util_custom import CustomDatasetConfig

DATASET_CONFIG = CustomDatasetConfig()

# function to get instance segmentation information from json
def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])    
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1# instance ids are 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs

# function to get semantic segmentation information from json
def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def export(pcd_file, agg_file, seg_file, box_file, output_file=None, config=DATASET_CONFIG):
    """ points are XYZ RGB (RGB in 0-255),
    instance label as 1-#instance,
    box as (cx,cy,cz,dx,dy,dz,alpha,beta,gamma,semantic_label)
    """
    
    #label_map=config.type2class # dont use this map
    #label_map = {'inside_fillet':2, 'inside_corner':3}
    #label_map = {'inside_corner':1, 'outside_corner':2, 'inside_fillet':3, 'outside_fillet':4}
    label_map = {'inside_corner':1, 'outside_corner':2, 'inside_outside_corner':3, 'inside_fillet':4, 'outside_fillet':5 }
    #label_map = {'inside_corner':1, 'outside_corner':2, 'inside_outside_corner':3, 'inside_fillet':4, 'outside_fillet':5 }
    
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(pcd_file)
    pcd_vertices=pcd.points
    #mesh_vertices = scannet_utils.read_mesh_vertices_rgb(mesh_file)
    #print('%d vertices loaded from pcd file'%len(pcd_vertices))

    ## Load scene axis alignment matrix # skip axis alignment for now
    #lines = open(meta_file).readlines()
    #for line in lines:
    #    if 'axisAlignment' in line:
    #        axis_align_matrix = [float(x) \
    #            for x in line.rstrip().strip('axisAlignment = ').split(' ')]
    #        break
   
    #axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
    #pts = np.ones((pcd_vertices.shape[0], 4))
    #pts[:,0:3] = pcd_vertices[:,0:3]
    #pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
    #pcd_vertices[:,0:3] = pts[:,0:3]

    # Load semantic and instance labels
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
   
    seg_to_verts, num_verts = read_segmentation(seg_file)
   
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
    object_id_to_label_id = {}
    for label, segs in label_to_segs.items():   
        label_id = label_map[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
   
    instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
    num_instances = len(np.unique(list(object_id_to_segs.keys())))
    for object_id, segs in object_id_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            instance_ids[verts] = object_id
            if object_id not in object_id_to_label_id:
                object_id_to_label_id[object_id] = label_ids[verts][0]
          
    # instance boxes do not need to be re-computed, can be loaded directly from file
    instance_bboxes = np.zeros((num_instances,10))
    lines = open(box_file).readlines()
    for idx,line in enumerate(lines):
        instance_bboxes[idx,0:9]=line.split(' ')[0:9]
        instance_bboxes[idx,9]=label_map[line.split(' ')[9].split('\n')[0]] # this last \n split is a hack
    
    # compute instance boxes from pcd vertices (NOT USED)    
    # for obj_id in object_id_to_segs:
    #     label_id = object_id_to_label_id[obj_id]
    #     print('label id:', label_id)
    #     obj_pc = np.asarray(pcd_vertices)[instance_ids==obj_id, 0:3]
    #     if len(obj_pc) == 0: continue
    #     # Compute axis aligned box
    #     # An axis aligned bounding box is parameterized by
    #     # (cx,cy,cz) and (dx,dy,dz) and label id
    #     # where (cx,cy,cz) is the center point of the box,
    #     # dx is the x-axis length of the box.
    #     xmin = np.min(obj_pc[:,0])
    #     ymin = np.min(obj_pc[:,1])
    #     zmin = np.min(obj_pc[:,2])
    #     xmax = np.max(obj_pc[:,0])
    #     ymax = np.max(obj_pc[:,1])
    #     zmax = np.max(obj_pc[:,2])
    #     bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2,
    #         xmax-xmin, ymax-ymin, zmax-zmin, label_id])
    #     # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
    #     instance_bboxes[obj_id-1,:] = bbox 
   
    if output_file is not None:
        np.save(output_file+'_vert.npy', pcd_vertices)
        np.save(output_file+'_sem_label.npy', label_ids)
        np.save(output_file+'_ins_label.npy', instance_ids)
        np.save(output_file+'_bbox.npy', instance_bboxes)

    return pcd_vertices, label_ids, instance_ids,\
        instance_bboxes, object_id_to_label_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan_path', required=True, help='path to scannet scene (e.g., data/ScanNet/v2/scene0000_00')
    parser.add_argument('--output_file', required=True, help='output file')
    #parser.add_argument('--label_map_file', required=True, help='path to scannetv2-labels.combined.tsv')
    opt = parser.parse_args()

    scan_name = os.path.split(opt.scan_path)[-1]
    pcd_file = os.path.join(opt.scan_path, scan_name + '.pcd')
    agg_file = os.path.join(opt.scan_path, scan_name + '.aggregation.json')
    seg_file = os.path.join(opt.scan_path, scan_name + '.segs.json')
    box_file = os.path.join(opt.scan_path, scan_name + '.boxes.txt')

    export(pcd_file, agg_file, seg_file, box_file, opt.output_file)

if __name__ == '__main__':
    main()
