# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sys
import os
import open3d as o3d
import copy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from box_util import get_3d_box
from box_util import rotx, roty, rotz

class CustomDatasetConfig(object):
    def __init__(self):
        self.num_class = 4
        self.num_heading_bin = 36 # 10 deg heading bins
        self.num_size_cluster = 4

        self.type2class={'outside_corner':0, 'inside_outside_corner':1, 'inside_fillet':2, 'outside_fillet':3 }
        #self.type2class={'inside_corner':0, 'outside_corner':1, 'inside_outside_corner':2, 'inside_fillet':3, 'outside_fillet':4 }
        self.class2type = {self.type2class[t]:t for t in self.type2class}
        
        self.classids = np.array([1,2,3,4]) # (see nyuids in scannet example) # non overlapping for debugging only
        #self.classids = np.array([1,2,3,4,5]) # (see nyuids in scannet example) # non overlapping for debugging only
        self.id2class = {classid: i for i,classid in enumerate(list(self.classids))}

        #self.mean_size_arr = np.load(os.path.join(ROOT_DIR,'scannet/meta_data/scannet_means.npz'))['arr_0']
        self.mean_size_arr = np.asarray([
                                        [ 1.0, 1.0, 1.0 ],
                                        [ 1.0, 1.0, 1.0 ],
                                        [ 5.0, 1.0, 1.0 ],
                                        [ 5.0, 1.0, 1.0 ],
                                        ])
        # self.mean_size_arr = np.asarray([
        #                                 [ 1.0, 1.0, 1.0 ],
        #                                 [ 1.0, 1.0, 1.0 ],
        #                                 [ 5.0, 1.0, 1.0 ],
        #                                 [ 5.0, 1.0, 1.0 ],
        #                                 [ 1.0, 1.0, 1.0 ]
        #                                 ])

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

    def param2obb(self, center, 
                  xheading_class, xheading_residual,
                  yheading_class, yheading_residual,
                  zheading_class, zheading_residual, 
                  size_class, size_residual):
        xheading_angle = self.class2angle(xheading_class, xheading_residual)
        yheading_angle = self.class2angle(yheading_class, yheading_residual)
        zheading_angle = self.class2angle(zheading_class, zheading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((9,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6:9] = [xheading_angle, yheading_angle, zheading_angle*-1]
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
    
    # centers = input_boxes[:,0:3] 
    # lengths = input_boxes[:,3:6]
    # angles = input_boxes[:,6:9]
    # sem_classes = input_boxes[:,9:10]
    #print(type(input_boxes))
    #print(np.asarray(input_boxes).shape)

    output_boxes=[]
    for idx, box in enumerate(input_boxes):

        center = box[0:3] 
        l,w,h = box[3:6]
        angles = box[6:9]
        sem_class = box[9:10]

        x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
        y_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];  
        z_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];

        corners = np.vstack([x_corners,y_corners,z_corners])

        corners[0,:] = corners[0,:] + center[0];
        corners[1,:] = corners[1,:] + center[1];
        corners[2,:] = corners[2,:] + center[2];

        #bbox0=o3d.geometry.OrientedBoundingBox().create_from_points(o3d.utility.Vector3dVector(np.transpose(corners)))
        #bbox0.color=[1,.2,.2]

        #point_base=o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        #cpoint0=copy.deepcopy(point_base).translate(center[:])
        #cpoint0.paint_uniform_color([ 1, .2, .2])

        #graphic for debugging rotation
        #origin_base = o3d.geometry.TriangleMesh.create_coordinate_frame()
        #origin=copy.deepcopy(origin_base).scale(0.25, center=(0,0,0))

        #origin0=copy.deepcopy(origin_base).scale(0.25, center=(0,0,0))
        #origin0=origin0.translate(center)

        # generate rotation matrices
        Rx = rotx(rot_angles[0])  
        Ry = roty(rot_angles[1]) 
        Rz = rotz(rot_angles[2]) 

        corners = np.matmul(Rx, corners) # apply three rotations seperately (for debugging)
        corners = np.matmul(Ry, corners)
        corners = np.matmul(Rz, corners)

        center = np.matmul(Rx, center)
        center = np.matmul(Ry, center)
        center = np.matmul(Rz, center)

        #bbox1=o3d.geometry.OrientedBoundingBox().create_from_points(o3d.utility.Vector3dVector(np.transpose(corners)))
        #bbox1.color=[.2,1,.2]

        #cpoint1=copy.deepcopy(point_base).translate(center[:])
        #cpoint1.paint_uniform_color([ .2, 1, .2])

        angles=np.asarray(angles)+np.asarray(rot_angles)    

        #origin1=copy.deepcopy(origin_base).scale(0.25, center=(0,0,0))
        #origin1=origin1.rotate(np.transpose(Rz))
        #origin1=origin1.translate(center)

        #o3d.visualization.draw_geometries([origin, bbox0, cpoint0, origin0, bbox1, cpoint1, origin1]) 

        output_box=np.concatenate([center, [l, w, h], angles, sem_class])
        output_boxes.append(output_box)
    
    #print(type(output_boxes))
    #print(np.asarray(output_boxes).shape)
    
    return np.asarray(output_boxes)