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
        self.num_class = 5
        self.num_heading_bin = 12 # previously used 36 -> 10 deg heading bins
        self.num_size_cluster = 5

        self.type2class={'inside_corner':0, 'outside_corner':1, 'inside_outside_corner':2, 'inside_fillet':3, 'outside_fillet':4 }
        self.class2type = {self.type2class[t]:t for t in self.type2class}
        
        self.classids = np.array([1,2,3,4,5]) # do not use 0, 0 is for unallocated instances
        self.id2class = {classid: i for i,classid in enumerate(list(self.classids))}

        #self.mean_size_arr = np.load(os.path.join(ROOT_DIR,'scannet/meta_data/scannet_means.npz'))['arr_0']
        self.mean_size_arr = np.asarray([ # consider computing mean size arr from training set as discussed in 'tips' 
                                        [ 1.0, 1.0, 1.0 ],
                                        [ 1.0, 1.0, 1.0 ],
                                        [ 2.0, 2.0, 2.0 ],
                                        [ 5.0, 1.0, 1.0 ],
                                        [ 5.0, 1.0, 1.0 ],
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
        obb[3:6] = box_size                                          # double check this conversion here  
        obb[6:9] = [xheading_angle, yheading_angle, -zheading_angle] # results in standard coordinate frame
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


def rotate_oriented_boxes(input_boxes, rot_angles, show_boxes=False):
    
    output_boxes=[] 
    for idx, box in enumerate(input_boxes):
        
        center = box[0:3] 
        l,w,h = box[3:6]
        angles = box[6:9]
        sem_class = box[9:10]
      
        if show_boxes:
          show_oriented_boxes([box],point_cloud)

        # generate rotation matrices
        Rx = rotx(rot_angles[0])  
        Ry = roty(rot_angles[1]) 
        Rz = rotz(rot_angles[2]) 

        # pre-multiply by R for fixed-axis rotations
        center = np.matmul(Rx, center)
        center = np.matmul(Ry, center)
        center = np.matmul(Rz, center)

        angles=np.asarray(angles)+np.asarray(rot_angles)
        
        #if sem_class==0: 
            #output_box=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            #print('semantic class:0, no feature')
        #else:         
        output_box=np.concatenate([center, [l, w, h], angles, sem_class])
        
        if show_boxes:
          show_oriented_boxes([output_box], point_cloud) 
        
        output_boxes.append(output_box)

    return np.asarray(output_boxes)


def show_oriented_boxes(input_boxes, point_cloud=None):

    origin_base = o3d.geometry.TriangleMesh.create_coordinate_frame()
    origin=copy.deepcopy(origin_base).scale(0.5, center=(0,0,0))
    draw_items=[origin]
    
    np.set_printoptions(precision=3)
    print(f'input_boxes: {input_boxes[0:5,:]}')

    part_angles=input_boxes[0,6:9]

    for idx, box in enumerate(input_boxes):
        center = box[0:3] 


        l,w,h = box[3:6]
        angles = box[6:9]


        sem_class = box[9:10]
        box=np.concatenate([center, [l, w, h], angles, sem_class])

        # generate corner points from the box parameters
        x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
        y_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];  
        z_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
        corners = np.vstack([x_corners,y_corners,z_corners])

        # fix for local rotation issue for features starting orientation (rotated feats, parts in dataset generation)
        feat_angles=angles-part_angles # subtracting the orientation of the first feature (assumed as part orientation)
        # print('feat_angles:', feat_angles) # this only works with the part is not also rotated in dataset generation
        
        # generate rotation matrices for feature rotation 
        Rx = rotx(feat_angles[0])  
        Ry = roty(feat_angles[1]) 
        Rz = rotz(feat_angles[2])

        # show a coordinate frame at the center point of the bounding boxes
        origin0=copy.deepcopy(origin_base).scale(0.5, center=(0,0,0)) 
        # rotate the frames to the box orientation
        #vertices=np.transpose(o3d.utility.Vector3dVector(origin0.vertices))
        vertices=o3d.utility.Vector3dVector(origin0.vertices)
        #vertices=np.transpose(vertices)# rotate the feature coordinate frame vertices to local orientation (part level)


        vertices=np.matmul(vertices,Rx) # post-multiply for moving axis rotation (feature frame rotation)
        vertices=np.matmul(vertices,Ry) 
        vertices=np.matmul(vertices,Rz)
        
        # generate rotation matrices for part rotation 
        Rx = rotx(part_angles[0])  
        Ry = roty(part_angles[1]) 
        Rz = rotz(part_angles[2])
        # rotate the features by the part orientation 
        #vertices=o3d.utility.Vector3dVector(origin0.vertices)
        vertices=np.transpose(vertices)
        vertices=np.matmul(Rx, vertices) 
        vertices=np.matmul(Ry, vertices) 
        vertices=np.matmul(Rz, vertices)
        
        origin0.vertices=o3d.utility.Vector3dVector(np.transpose(vertices))
        #origin0.vertices=o3d.utility.Vector3dVector(vertices)
        # before translating them to the box centers
        origin0.translate(center)
        # rotate the corner points
        corners = np.matmul(Rx, corners) # apply three rotations seperately (for debugging)
        corners = np.matmul(Ry, corners)
        corners = np.matmul(Rz, corners)
       # corners = np.matmul(np.transpose(corners),Rx) # apply three rotations seperately (for debugging)
       # corners = np.matmul(corners,Ry)
       # corners = np.matmul(corners,Rz)
       # corners = np.transpose(corners)
        # then move them to the box center
        corners[0,:] = corners[0,:] + center[0]; # this is not the numpy way at all
        corners[1,:] = corners[1,:] + center[1];
        corners[2,:] = corners[2,:] + center[2];

        # show all the corners of the bounding box 
        point_base=o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
        for i,corner in enumerate(np.transpose(corners)):
            point=copy.deepcopy(point_base).scale(1.0, center=(0,0,0))
            point=point.translate(corner)
            if i==1:
                point.paint_uniform_color([0, 0, 0])
            else:  
                point.paint_uniform_color([1, .2, .2])
            draw_items.append(point)
       
        #origin0.rotate(Rx, center=center) # o3d can do rotations also, handles rotation center  
        #origin0.rotate(Ry, center=center) 
        #origin0.rotate(Rz) 
        # origin0.rotate(Rx, center=(0,0,0))  
        # origin0.rotate(Ry, center=(0,0,0)) 
        # origin0.rotate(Rz, center=(0,0,0)) 
        # origin0.translate(center)
        

    # if available, show the point cloud as well
    if point_cloud is not None:

        cloud0=o3d.geometry.PointCloud() # complete part pointcloud 
        points=np.asarray(point_cloud)
        cloud0.points=o3d.utility.Vector3dVector(point_cloud[:,0:3])
        cloud0.paint_uniform_color([.6,.6,.6])
        draw_items.append(cloud0)        
        
    draw_items.append(origin0)
    o3d.visualization.draw_geometries(draw_items) 

