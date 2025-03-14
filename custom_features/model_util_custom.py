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
        obb[3:6] = box_size
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


#def rotate_oriented_boxes(input_boxes, rot_angles, point_cloud=None, show_boxes=False):
def rotate_oriented_boxes(input_boxes, rot_angles, show_boxes=False):
    # centers = input_boxes[:,0:3] 
    # lengths = input_boxes[:,3:6]
    # angles = input_boxes[:,6:9]
    # sem_classes = input_boxes[:,9:10]
    #print(np.asarray(input_boxes).shape)
    #print('input_boxes inside rotate_oriented_boxes:', input_boxes[:2,:])   
    #print(type(input_boxes), input_boxes.shape)
    #print('input_boxes: ', input_boxes[:2,:])    
    
    output_boxes=[] 
    for idx, box in enumerate(input_boxes):
        if idx<2:
            print('box: ',idx ,' inside rotate_oriented_boxes:')
            print(box)   
        
        center = box[0:3] 
        l,w,h = box[3:6]
        angles = box[6:9]
        sem_class = box[9:10]
      
        #input_box=np.concatenate([center, [l, w, h], angles, sem_class])
        #print('input_box: ', np.asarray(input_box))
        
        #print('idx: ',idx, ', box: ', box, ', box type:', type(box))

       # print('l: ', l)
      #  x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
      #  y_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];  
      #  z_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];

      #  corners = np.vstack([x_corners,y_corners,z_corners])

      #  corners[0,:] = corners[0,:] + center[0];
      #  corners[1,:] = corners[1,:] + center[1];
      #  corners[2,:] = corners[2,:] + center[2];

        if show_boxes:
          show_oriented_boxes([box],point_cloud)
          #  bbox0=o3d.geometry.OrientedBoundingBox().create_from_points(o3d.utility.Vector3dVector(np.transpose(corners)))
          #  bbox0.color=[1,.2,.2]

          #  point_base=o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
          #  cpoint0=copy.deepcopy(point_base).translate(center[:])
          #  cpoint0.paint_uniform_color([ 1, .2, .2])

          #  origin_base = o3d.geometry.TriangleMesh.create_coordinate_frame()
          #  origin=copy.deepcopy(origin_base).scale(0.25, center=(0,0,0))

          #  origin0=copy.deepcopy(origin_base).scale(0.25, center=(0,0,0))
          #  origin0=origin0.translate(center)

        #if point_cloud is not None:
        #    cloud0=o3d.geometry.PointCloud() # complete part pointcloud 
        #    cloud0.points=o3d.utility.Vector3dVector(np.asarray())

        # generate rotation matrices
        Rx = rotx(rot_angles[0])  
        Ry = roty(rot_angles[1]) 
        Rz = rotz(rot_angles[2]) 

      #  corners = np.matmul(Rx, corners) # apply three rotations seperately (for debugging)
      #  corners = np.matmul(Ry, corners)
      #  corners = np.matmul(Rz, corners)

        center = np.matmul(Rx, center)
        center = np.matmul(Ry, center)
        center = np.matmul(Rz, center)

      #  center = np.matmul(center, Rx)
      #  center = np.matmul(center, Ry)
      #  center = np.matmul(center, Rz)
       
        U=[1,0,0] # vector from center C to point P on local x axis
        U = np.matmul(Rx, U)
        U = np.matmul(Ry, U)
        U = np.matmul(Rz, U)
        
        Umag=(U[0]**2+U[1]**2+U[2]**2)**(1/2)
        Udir=[U[0]/Umag,U[1]/Umag,U[2]/Umag]

        V=[0,1,0] # vector from center C to point P on local y axis
        V = np.matmul(Rx, V)
        V = np.matmul(Ry, V)
        V = np.matmul(Rz, V)
        
        Vmag=(V[0]**2+V[1]**2+V[2]**2)**(1/2)
        Vdir=[V[0]/Vmag,V[1]/Vmag,V[2]/Vmag]
        
        W=[0,0,1] # vector from center C to point P on local z axis
        W = np.matmul(Rz, W)
        W = np.matmul(Ry, W)
        W = np.matmul(Rz, W)
        
        Wmag=(W[0]**2+W[1]**2+W[2]**2)**(1/2)
        Wdir=[W[0]/Wmag, W[1]/Wmag, W[2]/Wmag]
        
        if idx<2:

            print('angles: ', angles)
            print('rot_angles: ', rot_angles)
            print('angles+rot_angles: ', np.asarray(angles)+np.asarray(rot_angles))
            
            print('U: ', U)
           # print('Umag: ', Umag)
            print('Udir: ', Udir)
        
            print('V: ', V)
           # print('Vmag: ', Vmag)
            print('Vdir: ', Vdir)
            
            print('W: ', W)
           # print('Wmag: ', Wmag)
            print('Wdir: ', Wdir)
        
        #angles=Pdir
        angles=np.asarray(angles)+np.asarray(rot_angles)

        #if show_boxes:
          #  bbox1=o3d.geometry.OrientedBoundingBox().create_from_points(o3d.utility.Vector3dVector(np.transpose(corners)))
          #  bbox1.color=[.2,1,.2]

          #  cpoint1=copy.deepcopy(point_base).translate(center[:])
          #  cpoint1.paint_uniform_color([ .2, 1, .2])

          #  origin1=copy.deepcopy(origin_base).scale(0.25, center=(0,0,0))
          #  origin1=origin1.rotate(np.transpose(Rz))
          #  origin1=origin1.translate(center)

          #  o3d.visualization.draw_geometries([origin, bbox0, cpoint0, origin0, bbox1, cpoint1, origin1]) 

        output_box=np.concatenate([center, [l, w, h], angles, sem_class])
        if idx < 2:
            print('output_box ', idx, ': ', output_box)
        
        if show_boxes:
          show_oriented_boxes([output_box], point_cloud) 
        
        output_boxes.append(output_box)

    return np.asarray(output_boxes)


def show_oriented_boxes(input_boxes, point_cloud=None):

    origin_base = o3d.geometry.TriangleMesh.create_coordinate_frame()
    origin=copy.deepcopy(origin_base).scale(0.5, center=(0,0,0))
    draw_items=[origin]
    
    part_angles=input_boxes[0,6:9]

    for idx, box in enumerate(input_boxes):
        center = box[0:3] 
        l,w,h = box[3:6]
        angles = box[6:9]
        sem_class = box[9:10]
        box=np.concatenate([center, [l, w, h], angles, sem_class])
        #print('box: ', box)

        # generate corner points from the box parameters
        x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
        y_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];  
        z_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
        corners = np.vstack([x_corners,y_corners,z_corners])
        #print('corners: ', corners)

       # feat_angles=angles-part_angles
        #vertices=np.matmul(vertices,Rx) 
        # generate rotation matrices for feature roation purposes
        feat_angles=angles-part_angles
       # print('feat_angles:', feat_angles)
        Rx = rotx(feat_angles[0])  
        Ry = roty(feat_angles[1]) 
        Rz = rotz(feat_angles[2])

        # show a coordinate frame at the center point of the bounding boxes
        origin0=copy.deepcopy(origin_base).scale(0.25, center=(0,0,0)) 
        # rotate the frames to the box orientation
        #vertices=np.transpose(o3d.utility.Vector3dVector(origin0.vertices))
        vertices=o3d.utility.Vector3dVector(origin0.vertices)
        #vertices=np.transpose(vertices)
        vertices=np.matmul(vertices,Rx) 
        vertices=np.matmul(vertices,Ry) 
        vertices=np.matmul(vertices,Rz)
       # Rx = rotx(angles[0])  
       # Ry = roty(angles[1]) 
       # Rz = rotz(angles[2]) 

        # generate rotation matrices for display purposes
        Rx = rotx(part_angles[0])  
        Ry = roty(part_angles[1]) 
        Rz = rotz(part_angles[2])

        #vertices=o3d.utility.Vector3dVector(origin0.vertices)
        vertices=np.transpose(vertices)
        vertices=np.matmul(Rx, vertices) 
        vertices=np.matmul(Ry, vertices) 
        vertices=np.matmul(Rz, vertices)
        
        
        #vertices=np.matmul(vertices,Ry) 
        #vertices=np.matmul(vertices,Rz)
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
        #print('corners: ', corners, type(corners))

        # try:
        #     bbox0=o3d.geometry.OrientedBoundingBox().create_from_points(o3d.utility.Vector3dVector(np.transpose(corners)))
        #     bbox0.color=[1,.2,.2]
        #     draw_items.append(bbox0)
        # except:
        #     print('bounding box failed')    

        # show center point of the bounding box
        point_base=o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        cpoint0=copy.deepcopy(point_base).translate(center[:])
        cpoint0.paint_uniform_color([ 1, .2, .2])
        draw_items.append(cpoint0)

        for corner in np.transpose(corners):
            point=copy.deepcopy(point_base).scale(0.5, center=(0,0,0))
            point=point.translate(corner)
            point.paint_uniform_color([1, .2, .2])
            draw_items.append(point)

       
        #origin0.rotate(Rx, center=center) 
        #origin0.rotate(Ry, center=center) 
        #origin0.rotate(Rz) 
       
       # origin0.rotate(Rx, center=(0,0,0))  
       # origin0.rotate(Ry, center=(0,0,0)) 
       # origin0.rotate(Rz, center=(0,0,0)) 
       # origin0.translate(center)
        
        draw_items.append(origin0)

    # if available, show the point cloud as well
    if point_cloud is not None:

        cloud0=o3d.geometry.PointCloud() # complete part pointcloud 
        points=np.asarray(point_cloud)
        cloud0.points=o3d.utility.Vector3dVector(point_cloud[:,0:3])
        cloud0.paint_uniform_color([.6,.6,.6])
        draw_items.append(cloud0)        
        
    o3d.visualization.draw_geometries(draw_items)     
