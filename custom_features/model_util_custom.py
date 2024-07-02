# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from box_util import get_3d_box

class CustomDatasetConfig(object):
    def __init__(self):
        self.num_class = 4
        self.num_heading_bin = 36
        self.num_size_cluster = 4
        #self.type2class={'inside_fillet':0, 'inside_corner':1}
        self.type2class={'inside_corner':0, 'outside_corner':1, 'inside_fillet':2, 'outside_fillet':3}
        self.class2type = {self.type2class[t]:t for t in self.type2class}
        
        self.classids = np.array([1,2,3,4]) # (see nyuids in scannet example) # non overlapping for debugging only
        self.id2class = {classid: i for i,classid in enumerate(list(self.classids))}

        #self.mean_size_arr = np.load(os.path.join(ROOT_DIR,'scannet/meta_data/scannet_means.npz'))['arr_0']
        self.mean_size_arr = np.asarray([
                                        [ 1.0, 1.0, 1.0 ],
                                        [ 1.0, 1.0, 1.0 ],
                                        [ 5.0, 1.0, 1.0 ],
                                        [ 5.0, 1.0, 1.0 ]
                                        ])

        print('mean_size_arr:', type(self.mean_size_arr))

        self.type_mean_size = {}
        for i in range(self.num_size_cluster):
            self.type_mean_size[self.class2type[i]] = self.mean_size_arr[i,:]

    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optional] also small regression number from  
            class center angle to current angle.
           
            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle

            code borrowed from 'model_util_sunrgbd.py'
        '''
        num_class = self.num_heading_bin
        angle = angle%(2*np.pi)
        assert(angle>=0 and angle<=2*np.pi)
        angle_per_class = 2*np.pi/float(num_class)
        shifted_angle = (angle+angle_per_class/2)%(2*np.pi)
        class_id = int(shifted_angle/angle_per_class)
        residual_angle = shifted_angle - (class_id*angle_per_class+angle_per_class/2)
        return class_id, residual_angle
    
    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class.
        code borrowed from 'model_util_sunrgbd.py'
        '''
        
        num_class = self.num_heading_bin
        angle_per_class = 2*np.pi/float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle>np.pi:
            angle = angle - 2*np.pi
        return angle    

    def size2class(self, size, type_name):
        ''' Convert 3D box size (l,w,h) to size class and size residual '''
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual
    
    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''        
        return self.mean_size_arr[pred_cls, :] + residual

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle*-1
        return obb

def rotate_aligned_boxes(input_boxes, rot_mat):    
    centers, lengths = input_boxes[:,0:3], input_boxes[:,3:6]    
    new_centers = np.dot(centers, np.transpose(rot_mat))
    sem_classes=input_boxes[:,7]
           
    dx, dy = lengths[:,0]/2.0, lengths[:,1]/2.0
    new_x = np.zeros((dx.shape[0], 4))
    new_y = np.zeros((dx.shape[0], 4))
    
    for i, crnr in enumerate([(-1,-1), (1, -1), (1, 1), (-1, 1)]):        
        crnrs = np.zeros((dx.shape[0], 3))
        crnrs[:,0] = crnr[0]*dx
        crnrs[:,1] = crnr[1]*dy
        crnrs = np.dot(crnrs, np.transpose(rot_mat))
        new_x[:,i] = crnrs[:,0]
        new_y[:,i] = crnrs[:,1]
    
    new_dx = 2.0*np.max(new_x, 1)
    new_dy = 2.0*np.max(new_y, 1)    
    new_lengths = np.stack((new_dx, new_dy, lengths[:,2]), axis=1)
                  
    return np.concatenate([new_centers, new_lengths, sem_classes], axis=1)

def rotate_oriented_boxes(input_boxes, rot_angles):    
    new_centers, new_lengths = input_boxes[:,0:3], input_boxes[:,3:6]
    new_angles = input_boxes[:,6:9]+rot_angles # simply add to the angles indepently
    sem_classes = input_boxes[:,9:10]
           
    return np.concatenate([new_centers, new_lengths, new_angles, sem_classes], axis=1)