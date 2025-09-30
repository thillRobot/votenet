# trainingpart.py  
# thillRobot - March 19, 2025
# this program contains the part definition function for CustomFeatures dataset
# the functions used creating the part geomerty are also in this file
       
import open3d as o3d
import numpy as np
import math
import os
import copy
import shutil
import zipfile
import argparse
import json
import random
from open3d.t.geometry import TriangleMesh
import params # custom params file
import itertools

parser=argparse.ArgumentParser(
  prog='label_pointcloud',
  description='convert .pcd pointclouds from to annotated numpy .npy files')
parser.add_argument('--start-index', help="First index of new labels", type=int, required=False, default=0)
parser.add_argument('--number-classes', help="Number of classes to label", type=int, required=False, default=4)
parser.add_argument('--show-set', help="show entire dataset after generation", action="store_true")
parser.add_argument('--show-each', help="show each part after generation", action="store_true")
parser.add_argument('--no-archive', help="skip compressing archive with dataset and generation files", action="store_false")
parser.add_argument('--single-part', help="create a single part, skip dataset creation ", action="store_true")
parser.add_argument('--filename', help="filename for single part creation", type=str, required=False, default="single_part")

#custom color rgb codes
light_grey=(.65, .65, .65)
medium_grey=(.45, .45, .45)
dark_grey=(.25, .25, .25)
ttu_gold=(255/255, 221/255, 0)
ttu_purple=(79/255, 41/255, 132/255)
saftey_orange=(255/255, 95/255, 21/255)
bright_green=(170/255, 255/255, 0)
turquoise=(64/255,224/255,208/255)
dark_turquoise=(0/255,206/255,209/255)
dark_red=(139/255, 0, 0)
black=(0,0,0)


# global functions defined here for now, im not sure i like this ...
def hex_to_rgb(hexcode):
    
    h=hexcode.lstrip('#') # remove the # from the color code
    rgb=tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4)) # convert to rgb, (from SO)

    return rgb

IBM_COLORS={ # 40 is light and 80 is dark (it would be cool to automate this from the IBM website)
            'red40':'#ff8389', 'red50':'#fa4d56', 'red60':'#da1e28','red70':'#a2191f','red80':'#750e13',
            'magenta40':'#ff7eb6', 'magenta50':'#ee5396', 'magenta60':'#d02670','magenta70':'#9f1853','magenta80': '#740937',
            'purple40':'#be95ff' ,'purple50':'#a56eff' , 'purple60':'#8a3ffc','purple70':'#6929c4', 'purple80': '#491d8b',
            'blue40':'#78a9ff', 'blue50':'#4589ff', 'blue60':'#0f62fe','blue70':'#0043ce', 'blue80': '#002d9c',
            'teal40':'#08bdba', 'teal50':'#009d9a', 'teal60':'#007d79', 'teal70':'#005d5d', 'teal80':'#004144', 
            'green40':'#42be65', 'green50':'#24a148', 'green60':'#198038', 'green70':'#0e6027', 'green80':'#044317',
            'gray40':'#a8a8a8', 'gray50':'#8d8d8d', 'gray60':'#6f6f6f', 'gray70':'#525252', 'gray80':'#393939',
            'black':'#000000'
            }  

colormap=[] # make a color map of the IBM colors in the order shown above          

for color in IBM_COLORS.keys():
    if 'gray' not in color and 'black' not in color:
        colormap.append(IBM_COLORS[color])

# colors for feature classes, for visualization only
class_colors={'inside_fillet':hex_to_rgb(IBM_COLORS['blue40']), 
              'outside_fillet':hex_to_rgb(IBM_COLORS['green40']), 
              'inside_corner':hex_to_rgb(IBM_COLORS['red40']),
              'outside_corner':hex_to_rgb(IBM_COLORS['purple40']),
              'inside_outside_corner':hex_to_rgb(IBM_COLORS['teal40'])}



class TrainingPart:
    def __init__(self):

        self.part_cloud=o3d.geometry.PointCloud() # complete part pointcloud 
        #self.feature_cloud=o3d.geometry.PointCloud() # feature cloud for display only       
        self.feature_clouds=[] 
        # dictionary for parameter value indicies, does this belong in params.py ?
         
        self.param_indices={
                            'l0':0, 'l1':0, 'l2':0, 'l3':0, 
                            'w0':0, 'w1':0, 'w2':0, 'w3':0, 
                            't0':0, 't1':0, 't2':0, 't3':0, 
                            'l01':0, 'l12':0, 'l23':0, 'w01':0, 'w12':0, 
                            'point_density':0, 'fillet_radius':0, 'corner_radius':0
                           }                            
        
        #self.part_type=part_type # this may be unused
        self.part_type='part' # this is a tmp placeholder
        # define the names of the different types of parts (not feature classes), should be in params.py?
        self.all_parts=[]

        self.square_corners=True # equal aspect ratio on feature bounding boxes, should be in params.py for sure
        self.square_edges=True
        
        self.data_dir=os.environ[params.data_dir] # define location of files from bash environment var
        self.labels_dir=os.path.join(self.data_dir, params.labels_dir)
        self.points_dir=os.path.join(self.data_dir, params.points_dir)
        self.pcds_dir=os.path.join(self.data_dir, params.pcds_dir)

        self.labels_file='labels.txt'      # tmp name only, this may be unused
        self.archive_file=params.archive_file

        # class names to be used by VoteNet (FeatureNet), to params.py also
        self.feature_classes=['inside_corner', 'outside_corner', 'inside_outside_corner', 'inside_fillet', 'outside_fillet']        
        
        # segmentation labels are kept in a segmentation file, custom votenet dataloader requires instance segmentation labels
        self.part_segments=[]
        self.feature_segments=[]
        self.seg_indices=[]
   
        self.segment_data={           # these are stock params I think
                    "params": {
                    "kThresh": "0.0001",
                    "segMinVerts": "20",
                    "minPoints": "750",
                    "maxPoints": "300000",
                    "thinThresh": "0.05",
                    "flatThresh": "0.001",
                    "minLength": "0.02",
                    "maxLength": "1"
                  }
                  ,
                  "sceneId": self.part_type,
                  "segIndices": self.seg_indices
                  }
        
        self.instance_data={"sceneId": "scene",
                            "appId": "generate_dataset.py",
                            "segGroups": [],
                            "segmentsFile": "customfeatures.part.segs.json"}
        self.instance_groups=[]     

        self.update_params(self.param_indices)
        self.part_num=0  # tracks number parts in the dataset (a single object)
        self.part_set_num=0 # part number in the set of a single part type
        self.clear_part()
        

    def update_params(self, indices, verbose=False):

        self.param_indices=indices # get indices from trainingset.py from params.py
        # parameter lists moved to a param file (params.py) to clean things up and make is easy to re-create datasets
        # parameters for inside_fillet: l1, l2, l3, w1, w2, w3, t1, t2, t3, th12, th13, of12
        self.n_l=params.n_l       
        self.n_w=params.n_w        
        self.n_t=params.n_t             
        self.n_lw=params.n_lw
         
        #th12=(90.0, 0.0, 0.0) # rotations between plate1 and plate2, about the xyz axes, unused for now
        #th13=(90.0, 0.0, 0.0) # angles between plate1 and plate3, about the xyz axes
        #self.n_th=len(th12)       
        
        # these hard coded blocks should be removed, replace with loop
        # lookup using index values from param dictionary  
        #self.l0=params.l0[self.param_indices['l0']]
        self.l0=params.l0[indices['l0']]
        self.l1=params.l1[indices['l1']]
        self.l2=params.l2[indices['l2']]
        self.l3=params.l3[indices['l3']]

        self.w0=params.w0[indices['w0']]
        self.w1=params.w1[indices['w1']]
        self.w2=params.w2[indices['w2']]
        self.w3=params.w3[indices['w3']]

        self.t0=params.t0[indices['t0']]
        self.t1=params.t1[indices['t1']]
        self.t2=params.t2[indices['t2']]
        self.t3=params.t3[indices['t3']]

        self.l01=params.l01[indices['l01']]
        self.l12=params.l12[indices['l12']]
        self.l23=params.l23[indices['l23']]
        self.w01=params.w01[indices['w01']]
        self.w12=params.w12[indices['w12']] 
       
        self.point_density=params.point_density[indices['point_density']]
        self.fillet_radius=params.fillet_radius[indices['fillet_radius']]       
        self.corner_radius=params.fillet_radius[indices['corner_radius']]       
         
        self.fudge=params.fudge

        self.max_noise=params.max_noise
        self.sigma_noise=params.sigma_noise
        self.mu_noise=params.mu_noise
        
        self.num_holes=params.num_holes         # artificial voids (circular holes) in part clouds
        self.min_hole_radius=params.min_hole_radius
        self.max_hole_radius=params.max_hole_radius

        self.corner_fraction=params.corner_fraction     # corner params
        self.corner_edge_fraction=params.corner_edge_fraction
        self.edge_fraction=params.edge_fraction
        self.fillet_radius_fraction=params.fillet_radius_fraction
        self.fillet_edge_fraction=params.fillet_edge_fraction

        if verbose:
          print('l1=%f, l2=%f, l3=%f,'%(self.l1, self.l2, self.l3)) 
          print('w1=%f, w2=%f, w3=%f,'%(self.w1, self.w2, self.w3)) 
          print('t1=%f, t2=%f, t3=%f,'%(self.t1, self.t2, self.t3)) 
          print('w12=%f, l12=%f, l23=%f,'%(self.w12, self.l12, self.l23)) 

        # this can be replaced with a loop also 
        self.l0_indices=params.param_indices['l0']
        self.l1_indices=params.param_indices['l1'] 
        self.l2_indices=params.param_indices['l2']
        self.l3_indices=params.param_indices['l3']
        
        self.w0_indices=params.param_indices['w0']
        self.w1_indices=params.param_indices['w1']      
        self.w2_indices=params.param_indices['w2']    
        self.w3_indices=params.param_indices['w3']    

        self.t0_indices=params.param_indices['t0']
        self.t1_indices=params.param_indices['t1']       
        self.t2_indices=params.param_indices['t2']       
        self.t3_indices=params.param_indices['t3']       
 
        self.l01_indices=params.param_indices['l01']
        self.l12_indices=params.param_indices['l12']       
        self.l23_indices=params.param_indices['l23']
        self.w01_indices=params.param_indices['w01']      
        self.w12_indices=params.param_indices['w12']     
       
        self.point_density_indices=params.param_indices['point_density']
        self.fillet_radius_indices=params.param_indices['fillet_radius']

        # training, validation, and test split fractions 
        self.train_frac=params.train_frac
        self.val_frac=params.val_frac

    # erase all items and points to start a new part with same object
    def clear_part(self):

        self.plate_num=0
        self.feature_num=0
        
        self.part_items=[] # separate meshes and pointcloud objects, starts with an origin only
        self.part_cloud.clear() # clear the complete part pointcloud
        self.part_segments=[]
        self.feature_segments=[]
        self.seg_indices=[]
        self.instance_groups.clear()
        self.feature_clouds=[]

    # function to create a plate as a mesh box with two colored faces as triangle meshes
    def add_plate(self, size, position, orientation, colors=(light_grey, medium_grey, dark_grey) ):
        
       # colors=[self.hex_to_rgb(IBM_COLORS['black']),
       #         self.hex_to_rgb(IBM_COLORS['gray40']),
       #         self.hex_to_rgb(IBM_COLORS['gray40'])]
        colors=[hex_to_rgb(IBM_COLORS['black']),
                hex_to_rgb(IBM_COLORS['gray40']),
                hex_to_rgb(IBM_COLORS['gray40'])]
        # create plate as triangle mesh box     
        plate=o3d.geometry.TriangleMesh.create_box(width=size[0], height=size[1], depth=size[2])
        sarea=plate.get_surface_area()
        
        # calculate the points for this plate to maintain uniform surface point density across all parts
        num_points=int(self.point_density*sarea)

        # rotate plate first
        rot_xyz = plate.get_rotation_matrix_from_xyz(orientation)
        plate_rot=copy.deepcopy(plate).rotate(rot_xyz, center=(0,0,0))
        # translate plate second  
        plate_trans=copy.deepcopy(plate_rot).translate(position) 
        plate_trans_lines=o3d.geometry.LineSet.create_from_triangle_mesh(plate_trans)
        plate.paint_uniform_color(colors[0])
        # sample the mesh and add the plate points to the part pointcloud       
        cloud=o3d.geometry.TriangleMesh.sample_points_uniformly(plate_trans, num_points)           
        self.part_cloud=combine_pointclouds(self.part_cloud, cloud)
        self.part_segments.append(cloud)
        
        #instance_num=self.plate_num+1
        # the points in the plate are unannotated, this could be done much faster than point by point in a for loop, fix this soon
        for i,point in enumerate(cloud.points):
            #self.seg_indices.append(instance_num)
            self.seg_indices.append(0) # zero is for unannotated points
            
        # update the instance data , SKIP ADDING PLATES, add class instances only 
        #label='plate'
        #self.instance_data["sceneId"]='scene%s'%self.part_name
        #self.instance_groups.append({"id": instance_num,
        #                             "objectId": instance_num,
        #                             "segments": [instance_num],
        #                                 orientation=(0,0,0), 
        #                                 orientation=(0,0,0), 
        #                             "label": label})
        #self.instance_data['segGroups']=self.instance_groups

        # create a plane as triangle mesh
        plane1=o3d.geometry.TriangleMesh()
        plane1_vertices=np.asarray([ 
                                    [0, 0, -self.fudge[2]], 
                                    [size[0], 0, -self.fudge[2]], 
                                    [size[0], size[1], -self.fudge[2]], 
                                    [0, size[1], -self.fudge[2]] 
                                   ])

        plane1_triangles=np.asarray([[0,1,2], [2,3,0]])
        plane1.vertices=o3d.utility.Vector3dVector(plane1_vertices)
        plane1.triangles=o3d.utility.Vector3iVector(plane1_triangles)       
        plane1.paint_uniform_color(colors[1])
        # rotate plane1 first
        R = plane1.get_rotation_matrix_from_xyz(orientation)
        plane1_rot=copy.deepcopy(plane1).rotate(R, center=(0,0,0))
        # translate plane2 second
        plane1_trans=copy.deepcopy(plane1_rot).translate(position)
        
        # create a second plane as a triangle mesh
        plane2=o3d.geometry.TriangleMesh()
        plane2_vertices=np.asarray([
                                    [0, 0, size[2]], 
                                    [size[0], 0, size[2]], 
                                    [size[0], size[1], size[2]], 
                                    [0, size[1], size[2]] 
                                   ])

        plane2_triangles=np.asarray([[0,1,2], [2,3,0]])
        plane2.vertices=o3d.utility.Vector3dVector(plane2_vertices)
        plane2.triangles=o3d.utility.Vector3iVector(plane2_triangles)       
        plane2.paint_uniform_color(colors[2])
        # rotate plane1 first
        R = plane2.get_rotation_matrix_from_xyz(orientation)
        plane2_rot=copy.deepcopy(plane2).rotate(R, center=(0,0,0))
        # translate plane2 second
        plane2_trans=copy.deepcopy(plane2_rot).translate(position)
        
        # add the graphics items to the part object
        self.plate_num+=1
        #self.part_items.append(plate_trans)
        self.part_items.append(plate_trans_lines)
        #self.part_items.append(plane1_trans)
        #self.part_items.append(plane2_trans)  

        return plate_trans, plate_trans_lines, plane1_trans, plane2_trans
 
    
    def add_fillet(self, length, radius, position, orientation, quadrant, show_box=False):

        # create the mesh to represent the round fillet shape        
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
        R = cylinder.get_rotation_matrix_from_xyz(orientation)
        cylinder_rot = copy.deepcopy(cylinder).rotate(R, center=(0,0,0)) # rotate first
        cylinder_trans = copy.deepcopy(cylinder_rot).translate(position) # translate second

        sarea=cylinder.get_surface_area()
        
        # calculate the points for this plate to maintain uniform surface point density across all parts
        num_points=int(self.point_density*sarea)

        # create points on the cylinder through uniform sampling
        cloud=o3d.geometry.TriangleMesh.sample_points_uniformly(cylinder_trans, num_points)           

        # keep only the points in the quarter of the cylinder near the edge (seam)
        bbox=o3d.geometry.AxisAlignedBoundingBox()
        
        cylinder_size=cylinder.get_max_bound()-cylinder.get_min_bound()
        bbox.min_bound=[cylinder.get_min_bound()[0]/2, # cut the boxes in half and shift by 1/4
                        cylinder.get_min_bound()[1]/2,
                        cylinder.get_min_bound()[2]+self.fudge[2]] # trim both ends of the cylinder
        bbox.max_bound=[cylinder.get_max_bound()[0]/2,
                        cylinder.get_max_bound()[1]/2,
                        cylinder.get_max_bound()[2]-self.fudge[2]] 

        if quadrant==1:
            bbox.min_bound=bbox.min_bound+[cylinder_size[0]/4, cylinder_size[1]/4, 0]
            bbox.max_bound=bbox.max_bound+[cylinder_size[0]/4, cylinder_size[1]/4, 0]
        elif quadrant==2: 
            bbox.min_bound=bbox.min_bound+[-cylinder_size[0]/4, cylinder_size[1]/4, 0]
            bbox.max_bound=bbox.max_bound+[-cylinder_size[0]/4, cylinder_size[1]/4, 0]
        elif quadrant==3:
            bbox.min_bound=bbox.min_bound+[-cylinder_size[0]/4, -cylinder_size[1]/4, 0]
            bbox.max_bound=bbox.max_bound+[-cylinder_size[0]/4, -cylinder_size[1]/4, 0]
        else: 
            bbox.min_bound=bbox.min_bound+[cylinder_size[0]/4, -cylinder_size[1]/4, 0]
            bbox.max_bound=bbox.max_bound+[cylinder_size[0]/4, -cylinder_size[1]/4, 0]
        
        # convert to oriented bounding box because only OrientedBoundingBox can be rotated
        bbox_o=o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(bbox)
        bbox.color=(.2,.2,.2)
        bbox_o.color=(.2,.2,.2)

        #R=bbox_o.get_rotation_matrix_from_xyz(orientation)
        bbox_o.rotate(R,center=(0,0,0)).translate(position)
        
        #indices=np.asarray(bbox.get_point_indices_within_bounding_box(cloud.points))
        indices=np.asarray(bbox_o.get_point_indices_within_bounding_box(cloud.points))
        
        fillet_cloud=o3d.geometry.PointCloud()
        # only add the feature if points were found in the bounding box        
        if len(indices)>0:
        
            fillet_cloud.points=o3d.utility.Vector3dVector(np.asarray(cloud.points)[indices][:])
            fillet_cloud.paint_uniform_color((0, 0, 0)) 
        else:
            print('no fillet points found')      
 
        # the points on the cylinder are unannotated at first
        # this could be faster than point by point in a for loop, fix this soon
        for i,point in enumerate(fillet_cloud.points):
            #self.seg_indices.append(instance_num)
            self.seg_indices.append(0) # zero is for unannotated points

        self.part_cloud=combine_pointclouds(self.part_cloud, fillet_cloud)
        self.part_segments.append(fillet_cloud)
        if show_box:
            #self.part_items.append(cylinder_trans)
            self.part_items.append(bbox_o)

        return fillet_cloud


    def add_corner(self, radius, position, orientation, octant):

        # create the mesh to represent the round fillet shape        
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        R = sphere.get_rotation_matrix_from_xyz(orientation)
        sphere_rot = copy.deepcopy(sphere).rotate(R, center=(0,0,0)) # rotate first
        sphere_trans = copy.deepcopy(sphere_rot).translate(position) # translate second

        sarea=sphere.get_surface_area()
        
        # calculate the points for this plate to maintain uniform surface point density across all parts
        num_points=int(self.point_density*sarea)

        # create points on the cylinder through uniform sampling
        cloud=o3d.geometry.TriangleMesh.sample_points_uniformly(sphere_trans, num_points)           

        # keep only the points in the quarter of the cylinder near the edge (seam)
        bbox=o3d.geometry.AxisAlignedBoundingBox()
        
        sphere_size=sphere.get_max_bound()-sphere.get_min_bound()
        bbox.min_bound=[sphere.get_min_bound()[0]/2, # cut the boxes in half and shift by 1/4
                        sphere.get_min_bound()[1]/2,
                        sphere.get_min_bound()[2]/2] # trim both ends of the cylinder
        bbox.max_bound=[sphere.get_max_bound()[0]/2,
                        sphere.get_max_bound()[1]/2,
                        sphere.get_max_bound()[2]/2] 

        if octant==1:
            bbox.min_bound=bbox.min_bound+[sphere_size[0]/4, sphere_size[1]/4,sphere_size[1]/4]
            bbox.max_bound=bbox.max_bound+[sphere_size[0]/4, sphere_size[1]/4,sphere_size[1]/4]
        elif octant==2:                                                        
            bbox.min_bound=bbox.min_bound+[-sphere_size[0]/4, sphere_size[1]/4,sphere_size[1]/4]
            bbox.max_bound=bbox.max_bound+[-sphere_size[0]/4, sphere_size[1]/4,sphere_size[1]/4]
        elif octant==3:                                                        
            bbox.min_bound=bbox.min_bound+[-sphere_size[0]/4, -sphere_size[1]/4,sphere_size[1]/4]
            bbox.max_bound=bbox.max_bound+[-sphere_size[0]/4, -sphere_size[1]/4,sphere_size[1]/4]
        elif octant==4:   
            bbox.min_bound=bbox.min_bound+[sphere_size[0]/4, -sphere_size[1]/4,sphere_size[1]/4]
            bbox.max_bound=bbox.max_bound+[sphere_size[0]/4, -sphere_size[1]/4,sphere_size[1]/4]
        elif octant==5:
            bbox.min_bound=bbox.min_bound+[sphere_size[0]/4, sphere_size[1]/4,-sphere_size[1]/4]
            bbox.max_bound=bbox.max_bound+[sphere_size[0]/4, sphere_size[1]/4,-sphere_size[1]/4]
        elif octant==6:                                                        
            bbox.min_bound=bbox.min_bound+[-sphere_size[0]/4, sphere_size[1]/4,-sphere_size[1]/4]
            bbox.max_bound=bbox.max_bound+[-sphere_size[0]/4, sphere_size[1]/4,-sphere_size[1]/4]
        elif octant==7:                                                        
            bbox.min_bound=bbox.min_bound+[-sphere_size[0]/4, -sphere_size[1]/4,-sphere_size[1]/4]
            bbox.max_bound=bbox.max_bound+[-sphere_size[0]/4, -sphere_size[1]/4,-sphere_size[1]/4]
        else:   
            bbox.min_bound=bbox.min_bound+[sphere_size[0]/4, -sphere_size[1]/4,-sphere_size[1]/4]
            bbox.max_bound=bbox.max_bound+[sphere_size[0]/4, -sphere_size[1]/4,-sphere_size[1]/4]
                                                                                 
        bbox_o=o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(bbox)
        bbox.color=(.2,.2,.2)
        bbox_o.color=(.2,.2,.2)

        #R=bbox_o.get_rotation_matrix_from_xyz(orientation)
        bbox_o.rotate(R).translate(position)
        
        #indices=np.asarray(bbox.get_point_indices_within_bounding_box(cloud.points))
        indices=np.asarray(bbox_o.get_point_indices_within_bounding_box(cloud.points))
        
        corner_cloud=o3d.geometry.PointCloud()
        # only add the feature if points were found in the bounding box        
        if len(indices)>0:
        
            corner_cloud.points=o3d.utility.Vector3dVector(np.asarray(cloud.points)[indices][:])
            corner_cloud.paint_uniform_color((0, 0, 0)) 
        else:
            print('no corner points found')      
 
        # the points on the cylinder are unannotated at first
        # this could be faster than point by point in a for loop, fix this soon
        for i,point in enumerate(corner_cloud.points):
            #self.seg_indices.append(instance_num)
            self.seg_indices.append(0) # zero is for unannotated points

        self.part_cloud=combine_pointclouds(self.part_cloud, corner_cloud)
        self.part_segments.append(corner_cloud)
        #self.part_items.append(sphere_trans)
        #self.part_items.append(bbox_o)

        return corner_cloud
    

    # function to get ALL points from a cloud inside an aligned bounding box
    def get_points_within_bbox(self,cloud,bbox):
        
        bounds=bbox.get_max_bound()
        [xmax, ymax, zmax]=bbox.get_max_bound()
        [xmin, ymin, zmin]=bbox.get_min_bound()
        
        indices=[] 
        points=[]
        for idx,point in enumerate(cloud.points):
            if point[0]>xmin and point[0]<xmax and point[1]>ymin and point[1]<ymax and point[2]>zmin and point[2]<zmax:
                indices.append(idx)
                points.append(point)
        return points, indices 

    # function to create a bounding box and sample points within from the part mesh
    def sample_part(self, size, position, orientation, show_points=True, show_bbox=False):
              
        bbox=o3d.geometry.AxisAlignedBoundingBox()       
        cloud=o3d.geometry.PointCloud()       
         
        # draw bbox 1 - this box will contain the feature points
        if size[0]>0 and size[1]>0 and size[2]>0:
            # this might be an extra step, but it works for now
            box=o3d.geometry.TriangleMesh.create_box( width=size[0], height=size[1], depth=size[2] ) 
            R = box.get_rotation_matrix_from_xyz(orientation) # the rotation and translation could be applied directly to the bounding box object
            box_rot=copy.deepcopy(box).rotate(R, center=(0,0,0))
            box_trans=copy.deepcopy(box_rot).translate(position)
          
            bbox_vertices=np.asarray(box_trans.vertices) # this AxisAlignedBoundingBox could probably be defined directly here (fix this later)
            bbox=bbox.create_from_points(o3d.utility.Vector3dVector(bbox_vertices))
            #print('bbox extent:',bbox.get_extent())           
            
            # sampling is not appropriate here, we should take all the point in the box, parts have already been sampled once 
            # sample the pointclouds inside the bounding volume of interest, this should return all the points in the box
            # !note: if add_noise() is used after the features are created and sampled, it will appear is some points are missed.
            
            #indices=np.asarray(bbox.get_point_indices_within_bounding_box(self.part_cloud.points))     
            # replace with function from scratch to test....  
            points, indices = self.get_points_within_bbox(self.part_cloud, bbox) # homemade function seems to work 
            # store the indices in the segmentation index list
            for i,index in enumerate(indices):
                #self.seg_indices[index]=self.max_plates+(self.feature_num)+1 # shift the index so the feature segs follow the part segs
                self.seg_indices[index]=self.feature_num+1 # shift the index so the feature segs follow the part segs

            # only add the feature if points were found in the bounding box        
            if len(indices)>0:
            
                cloud.points=o3d.utility.Vector3dVector(np.asarray(self.part_cloud.points)[indices][:])
                cloud.paint_uniform_color((0, 0, 0)) 
                if show_points: 
                    self.part_items.append(cloud)
             
            #else:
                #print('no points found, pointcloud not added')
            
            # add the box even if no points were found
            if show_bbox:
                bbox.color=(.2,.2,.2)
                bbox_o=o3d.geometry.OrientedBoungingBox.create_from_axis_aligned_bounding_box(bbox)     
                self.part_items.append(bbox_o)
 
        else:
            #print('bounding box volume <= 0, feature not added')
            #print('size=[%.2f, %.2f, %.2f]'%(size[0],size[1],size[2]))
            indices=[] 
        return bbox, cloud, indices

    
    def add_feature(self, bbox, cloud, label, position=(0,0,0), orientation=(0,0,0), feature_orientation=(0,0,0), orientation_flag='xaxis' ):
      
        feature_cloud=cloud
        #feature_cloud.paint_uniform_color(self.hex_to_rgb(IBM_COLORS['gray40']))
        feature_cloud.paint_uniform_color(hex_to_rgb(IBM_COLORS['gray40']))
        self.feature_clouds.append(feature_cloud)
        # get a bounding box from the input points
        bbox=o3d.geometry.AxisAlignedBoundingBox.create_from_points(cloud.points)
        bbox_o=o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(bbox) # only OrientedBoundingBox can be rotated
        #bbox_o.color=(self.hex_to_rgb(IBM_COLORS['magenta40']))
        bbox_o.color=(hex_to_rgb(IBM_COLORS['magenta40']))
        #bbox.color=(.2,.2,.2)
        
        center=o3d.geometry.TriangleMesh()
        center.vertices=o3d.utility.Vector3dVector([np.asarray(bbox.get_center())])
        R = center.get_rotation_matrix_from_xyz(orientation) 
        center.rotate(R,center=(0,0,0))  # rotate then translate center point for label
        center.translate(position)
        [cx, cy, cz]=np.asarray(center.vertices[0])
        [dx, dy, dz]=bbox.get_extent() # only AxisAlignedBoundingBox has get_extent
       
        corners=np.asarray(bbox.get_box_points()) 
        
        # caclulate distance and angle between pointA and pointB, and between pointA and pointC
        AB=np.sqrt((corners[0][0]-corners[1][0])**2+(corners[0][1]-corners[1][1])**2+(corners[0][2]-corners[1][2])**2)                
        AC=np.sqrt((corners[0][0]-corners[2][0])**2+(corners[0][1]-corners[2][1])**2+(corners[0][2]-corners[2][2])**2)                

        corner0=(corners[0][0], corners[0][1], corners[0][2]) 
        corner1=(corners[1][0], corners[1][1], corners[1][2])
        corner2=(corners[2][0], corners[2][1], corners[2][2])

        point_base=o3d.geometry.TriangleMesh.create_sphere(radius=0.05)

        #pointA=copy.deepcopy(point_base).translate((corners[0][0], corners[0][1], corners[0][2]))
        pointA=copy.deepcopy(point_base).translate(corner0)
        pointA.paint_uniform_color([ 1, .1, .1])
        pointB=copy.deepcopy(point_base).translate(corner1)
        pointB.paint_uniform_color([.1,  1, .1])
        pointC=copy.deepcopy(point_base).translate(corner2)
        pointC.paint_uniform_color([.1, .1,  1])
        self.part_items.append(pointA)
        self.part_items.append(pointB)
        self.part_items.append(pointC)

        # add the part rotation to the feature orientation angles
        # angles add if using 'fixed-axis' rotations with respect to base frame
        [al, bt, gm] = np.asarray(orientation)+np.asarray(feature_orientation) 
        # calculating the feature heading is not required, boxes are aligned 
        # this in only a good assumption because all features normal for now and the part is not rotated yet
        min_length=0.75
        if np.asarray([dx,dy,dz]).max() > min_length or label=='inside_corner':
            #self.part_items.append(bbox)
            self.part_items.append(bbox_o) # show the oriented bounding box around the feature
            self.feature_num+=1

            labels_file='%s/scene%s.boxes.txt' % (self.part_type, self.part_name)  
            pcd_file='%s/scene%s_%06d_%s.pcd' % (label, self.part_name, self.feature_num, label)
            points_file='%s/scene%s_%06d_%s.npy' % (label, self.part_name, self.feature_num, label) 

            # this seems to fix the multiple feature orientation issues, feature box sizes must be defined wrt global frame
            if orientation_flag == 'yaxis':
                tmp=dx # swap x and y box dimensions, this is very counter intutive to me
                dx=dy
                dy=tmp
            elif orientation_flag == 'zaxis':
                tmp=dx # swap x and z box dimensions  
                dx=dz
                dz=tmp  

            #  write a label entry for each feature bounding box
            # label_header='# format: [x y z dx dy dz alpha beta gamma class_name]\n'
            box_info='%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %s'%(cx,cy,cz,dx,dy,dz,al,bt,gm,label) 
            with open(os.path.join(self.labels_dir, labels_file), 'a', encoding="utf-8") as f:
                f.write(box_info+'\n')
                f.close  

            # add the feature segment to the list for segmentation labels
            self.feature_segments.append(cloud)
            # update the instance data 
            instance_num=self.feature_num
            self.instance_data["sceneId"]='scene%s'%self.part_name
            self.instance_groups.append({"id": instance_num,
                                         "objectId": instance_num,
                                         "segments": [instance_num],
                                         "label": label,
                                         'box': box_info})
            self.instance_data['segGroups']=self.instance_groups
            
           # if self.save_feature_files:      
           #     # stack the data to be written to the feature points file and feature pcd files (segmented features)
           #     intensities=np.zeros( np.asarray(cloud.points).shape[0] )
           #     data=np.column_stack( (np.asarray(cloud.points), intensities) ) 
           #     #data=np.column_stack( ( points, np.asarray(cloud.colors)) ) # leave out colors for now
           #     # save the pointcloud and the points file
           # 
           #     o3d.io.write_point_cloud( os.path.join(self.pcds_dir, pcd_file), cloud )                             
           #     np.save( os.path.join(self.points_dir, points_file), data )
             
            self.segment_data["sceneId"]='scene%s'%self.part_name
            self.segment_data['segIndices']=self.seg_indices    
            # save the segmentation information as json file
            segment_file='%s/scene%06d_%s.segs.json' % (self.part_type, self.part_set_num, self.part_type)            
             
            with open(os.path.join(self.labels_dir,segment_file), 'w', encoding='utf-8') as f:
                json.dump(self.segment_data, f, ensure_ascii=False, indent=4)

            # also save the instance information as json, this uses the segmentation indices also         
            self.instance_data['segmentsFile']='customfeatures.scene%s.segs.json'%self.part_name

            instance_file='%s/scene%06d_%s.aggregation.json' % (self.part_type, self.part_set_num, self.part_type)          
            with open(os.path.join(self.labels_dir,instance_file), 'w', encoding='utf-8') as f:
                json.dump(self.instance_data, f, ensure_ascii=False, indent=4)
        #else:
            #print('feature shorter than min length, not added')
   
 
    def save_part(self, filename): 
        
        try:         
            part_pcd_file='%s/%s.pcd' % (self.part_type, filename)
            part_points_file='%s/%s.npy' % (self.part_type, filename)            
            
            # stack the data to be written to the full part points and pcd files
            # print('colors: ',type(cloud.colors), np.asarray(cloud.colors).shape )
            part_intensities=np.zeros( np.asarray(self.part_cloud.points).shape[0] )
            part_data=np.column_stack( (np.asarray(self.part_cloud.points), part_intensities) ) 
            #data=np.column_stack( ( points, np.asarray(cloud.colors)) ) # leave out colors for now
            # save the pointcloud and the points file
            o3d.io.write_point_cloud( os.path.join(self.pcds_dir, part_pcd_file), self.part_cloud )                             
            np.save( os.path.join(self.points_dir, part_points_file), part_data )

            # add the part_name to all_parts list to be used in prepare_splits afterwards                
            self.all_parts.append('%s'%filename)
            self.part_filename=filename     

        except Exception as e:
            print('save_part() failed:', e)
            print('part_pcd_file: ', part_pcd_file)
            print('part_points_file:', part_points_file)


    def create_1plate_partA(self, position=(0,0,0), orientation=(0,0,0), scale=1, filename='use_part_name'):

        self.part_type='1plateA' 
        self.part_name='%06d_%s'%(self.part_set_num, self.part_type)
        self.max_plates=1
        
        if filename=='use_part_name':
            filename='scene%s'%(self.part_name)
        
        fudge_x=self.fudge[0]
        fudge_y=self.fudge[1]
        fudge_z=self.fudge[2]
  
        plate1, plate1_lines, plane11, plane12 = self.add_plate(size=(self.l1,self.w1,self.t1), position=(0,0,0), orientation=(0,0,0))

        # delete random regions of points from the cloud (spherical regions for now)
        self.delete_regions(self.part_cloud, num_regions=self.num_holes, min_radius=self.min_hole_radius, max_radius=self.max_hole_radius)
        
        #self.corner_fraction=.25
        lf1=self.l1*self.corner_fraction      # use the respective side length times the corner fraction
        wf1=self.w1*self.corner_fraction
        hf1=self.t1*self.corner_fraction 
        
        self.square_corners=True
        # define 8 corner features
        if self.square_corners:    
            c1=np.asarray([lf1, wf1, hf1]).min() # use the smallest side to define the corner size
            f1_size=(c1, c1, c1)
        else:
            f1_size=(lf1, wf1, hf1)


        # vertices of plate [x,y,z , roll, pitch, yaw] # testing using angles now, training issues
        corners=[[-fudge_x,                  -fudge_y,                   -fudge_z, 0, 0, 0],         # bottom four corners
                [self.l1-f1_size[0]+fudge_x, -fudge_y,                   -fudge_z, 0, 0, 0],
                [self.l1-f1_size[0]+fudge_x, self.w1-f1_size[1]+fudge_y, -fudge_z, 0, 0, 0],
                [-fudge_x,                   self.w1-f1_size[1]+fudge_y, -fudge_z, 0, 0, 0],
                [-fudge_x,                   -fudge_y,                   self.t1-f1_size[2]+fudge_z,0, 0, 0],# top four corners, on x axis
                [self.l1-f1_size[0]+fudge_x, -fudge_y,                   self.t1-f1_size[2]+fudge_z, 0, 0, 0], 
                [self.l1-f1_size[0]+fudge_x, self.w1-f1_size[1]+fudge_y, self.t1-f1_size[2]+fudge_z, 0, 0, 0],
                [-fudge_x,                   self.w1-f1_size[1]+fudge_y, self.t1-f1_size[2]+fudge_z, 0, 0, 0]] # on xaxis
        
        for corner in corners:     
            # sample and create a pointcloud with a bounding box for feature1
            
            f1_bbox, f1_cloud, f1_indices = self.sample_part(size=f1_size, 
                                                             position=corner[0:3], 
                                                             orientation=(0,0,0) )
            f1_orientation=(0,0,0)
            
            # only add the feature if there are points returned from sampling to avoid segmentation index error
            if len(f1_indices)>0:
                self.add_feature(f1_bbox, f1_cloud, 'outside_corner', position=position,
                                                                      orientation=orientation,
                                                                      feature_orientation=f1_orientation) 
                # pass in feature orientation so the label has updated info

        # edges defined by indices of plate1_vertices, orientations in last 3 columns
      #  plate1_xedges=[[0,1,0,0,0], # edges parallel to x axis
      #                 [3,2,-np.pi/2,0,0],
      #                 [7,6,-2*np.pi/2,0,0],
      #                 [4,5,-3*np.pi/2,0,0]]

      #  plate1_yedges=[[0,3,0,0,np.pi/2], # edges parallel to y axis
      #                 [4,7,np.pi/2,0,np.pi/2],
      #                 [5,6,2*np.pi/2,0,np.pi/2],
      #                 [1,2,3*np.pi,0,np.pi/2]]     

      #  plate1_zedges=[[0,4,0,-np.pi/2,0], # edges parallel to z axis
      #                 [1,5,0,-np.pi/2,-np.pi/2],
      #                 [2,6,0,-np.pi/2,-2*np.pi/2],
      #                 [3,7,0,-np.pi/2,-3*np.pi/2]]                                       

        plate1_xedges=[[0,1,0,0,0], # edges parallel to x axis
                       [3,2,0,0,np.pi],
                       [7,6,0,np.pi,np.pi],
                       [4,5,0,np.pi,0]]

        plate1_yedges=[[0,3,0,0,np.pi/2], # edges parallel to y axis
                       [4,7,np.pi,0,3*np.pi/2],
                       [5,6,np.pi,0,np.pi/2],
                       [1,2,0,0,3*np.pi/2]]     

        plate1_zedges=[[0,4,0,-np.pi/2,0], # edges parallel to z axis
                       [1,5,0,-np.pi/2,-np.pi/2],
                       [2,6,0,-np.pi/2,-2*np.pi/2],
                       [3,7,0,-np.pi/2,-3*np.pi/2]]                                       

        for edge in plate1_xedges:

            lf2=self.l1*(1-2*self.corner_fraction)
            wf2=self.w1*self.edge_fraction
            hf2=self.t1*self.edge_fraction

            if self.square_edges:    

                e2=np.asarray(f1_size).min() # use the smallest side to define the edge width
                l2=self.l1-2*e2
                f2_size=(l2, e2, e2)
                xf2=corners[edge[0]][0]+c1
                yf2=corners[edge[0]][1]
                zf2=corners[edge[0]][2]

            else:    
                f2_size=(lf2, wf2, hf2)

                xf2=corners[edge[0]][0]+lf1
                yf2=corners[edge[0]][1]
                zf2=corners[edge[0]][2]

            f2_bbox, f2_cloud, f2_indices = self.sample_part(size=f2_size, 
                                                             position=(xf2, yf2, zf2), 
                                                             orientation=(0,0,0))
            f2_orientation=edge[2:] 
            # only add the feature if there are points returned from sampling to avoid segmentation index error
            if len(f2_indices)>0:
                # pass in local feature orientation and part orientation so the label has updated info
                self.add_feature(f2_bbox, f2_cloud, 'outside_fillet', position=position,
                                                                      orientation=orientation,
                                                                      feature_orientation=f2_orientation,
                                                                      orientation_flag='xaxis') 
                                                             
        for edge in plate1_yedges:

            lf2=self.l1*self.edge_fraction
            wf2=self.w1*(1-2*self.corner_fraction)
            hf2=self.t1*self.edge_fraction

            if self.square_edges:    
                w2=self.w1-2*e2
                f2_size=(l2, e2, e2)
                xf2=corners[edge[0]][0]
                yf2=corners[edge[0]][1]+c1
                zf2=corners[edge[0]][2]

                f2_size=(e2, w2, e2)
            else:    
                f2_size=(lf2, wf2, hf2)

                xf2=corners[edge[0]][0]
                yf2=corners[edge[0]][1]+wf1
                zf2=corners[edge[0]][2]

            f2_bbox, f2_cloud, f2_indices = self.sample_part(size=f2_size, 
                                                             position=(xf2, yf2, zf2), 
                                                             orientation=(0,0,0))   
            f2_orientation=edge[2:]
            # only add the feature if there are points returned from sampling to avoid segmentation index error
            if len(f2_indices)>0:
                #self.add_feature(f2_bbox, f2_cloud, 'outside_fillet', f2_orientation) # pass in final part orientation so the label has updated info
                self.add_feature(f2_bbox, f2_cloud, 'outside_fillet', position=position,
                                                                      orientation=orientation,
                                                                      feature_orientation=f2_orientation,
                                                                      orientation_flag='yaxis') 

      # skip z edges for debuggin 
      #  for edge in plate1_zedges:

      #      lf2=self.l1*self.edge_fraction
      #      wf2=self.w1*self.edge_fraction
      #      hf2=self.t1*(1-2*self.corner_fraction)

      #      if self.square_edges:    

      #          h2=self.t1-2*e2
      #          f2_size=(e2, e2, h2)
      #          xf2=corners[edge[0]][0]
      #          yf2=corners[edge[0]][1]
      #          zf2=corners[edge[0]][2]+c1

      #      else:    
      #          f2_size=(lf2, wf2, hf2)
      #          xf2=corners[edge[0]][0]
      #          yf2=corners[edge[0]][1]
      #          zf2=corners[edge[0]][2]+hf1

      #      f2_bbox, f2_cloud, f2_indices = self.sample_part(size=f2_size, 
      #                                                       position=(xf2, yf2, zf2), 
      #                                                       orientation=(0,0,0))
      #      f2_orientation=edge[2:]
      #      # only add the feature if there are points returned from sampling to avoid segmentation index error
      #      if len(f2_indices)>0:
      #          #self.add_feature(f2_bbox, f2_cloud, 'outside_fillet', f2_orientation) # pass in final part orientation so the label has updated info
      #          self.add_feature(f2_bbox, f2_cloud, 'outside_fillet', position=position,
      #                                                                orientation=orientation,
      #                                                                feature_orientation=f2_orientation,
      #                                                                orientation_flag='zaxis') 

        # after labeling add noise to the pointcloud
        #self.part_cloud = self.add_noise(self.part_cloud, mu=self.mu_noise, sigma=self.sigma_noise, max_noise=self.max_noise)    

        # after adding all peices and sampling, rotate the part and feature 
        for k,item in enumerate(self.part_items):
            R = item.get_rotation_matrix_from_xyz(orientation) 
            item.rotate(R, center=(0,0,0)) # be sure to use center=(0,0,0)
            item.translate(position)
        
        # rotate the part cloud as well 
        R = self.part_cloud.get_rotation_matrix_from_xyz(orientation) 
        self.part_cloud.rotate(R, center=(0,0,0)) # use center=(0,0,0) here too
        self.part_cloud.translate(position)

        self.save_part(filename)     
  
        # after adding all peices, features (and labels) increment the part counter
        self.part_num+=1
      
        return self.part_items
    

    def create_1plate_partB(self, position=(0,0,0), orientation=(0,0,0), scale=1, filename='use_part_name'):

        self.part_type='1plateB' 
        self.part_name='%06d_%s'%(self.part_set_num, self.part_type)
        self.max_plates=1
        
        if filename=='use_part_name':
            filename='scene%s'%(self.part_name)
        
        fudge_x=self.fudge[0]
        fudge_y=self.fudge[1]
        fudge_z=self.fudge[2]
  
        # make a block from three plates
        plate1, plate1_lines, plane11, plane12 = self.add_plate(size=(self.l1,self.w1,self.t1), position=(0,0,0), orientation=(0,0,0))
        plate2, plate2_lines, plane21, plane22 = self.add_plate(size=(self.l1,self.w2,self.t2), position=(0,0,self.t1), orientation=(0,0,0))

        # delete random regions of points from the cloud
        self.delete_regions(self.part_cloud,
                            num_regions=params.num_holes, 
                            min_radius=params.min_hole_radius, 
                            max_radius=params.max_hole_radius)
        
        # define 6 corner features
        lf1=self.l1*self.corner_fraction
        wf1=self.w1*self.corner_fraction
        hf1=self.t1*self.corner_fraction

        h1=self.t1+self.t2

        # define 8 corner features
        if self.square_corners:    
            c1=np.asarray([lf1, wf1, hf1]).min() # use the smallest side to define the corner size
            c1_size=(c1, c1, c1)
        else:
            c1_size=(lf1, wf1, hf1)
        
        #vertices of plate [x,y,z , roll, pitch, yaw] # angles not used for now
        corners=[[-fudge_x,                     -fudge_y,                   -fudge_z, 0, 0, 0, 0],      # part point 0
                [self.l1-c1_size[0]+fudge_x,    -fudge_y,                   -fudge_z, 0, 0, 0],         # part point 1
                [self.l1-c1_size[0]+fudge_x,    self.w1-c1_size[1]+fudge_y, -fudge_z, 0, 0, 0],         # part point 2
                [-fudge_x,                      self.w1-c1_size[1]+fudge_y, -fudge_z, 0, 0, 0],         # part point 3
                [-fudge_x,                      self.w2-c1_size[1]+fudge_y, self.t1-c1_size[2]+fudge_z,0,0,0],  # part point 4
                [self.l1-c1_size[0]+fudge_x,    self.w2-c1_size[1]+fudge_y, self.t1-c1_size[2]+fudge_z,0,0,0],  # part point 5
                [self.l1-c1_size[0]+fudge_x,    self.w1-c1_size[1]+fudge_y, self.t1-c1_size[2]+fudge_z,0,0,0],  # part point 6
                [-fudge_x,                      self.w1-c1_size[1]+fudge_y, self.t1-c1_size[2]+fudge_z,0,0,0],  # part point 7
                [-fudge_x,                      -fudge_y,                   h1-c1_size[2]+fudge_z, 0, 0, 0],    # part point 8
                [self.l1-c1_size[0]+fudge_x,    -fudge_y,                   h1-c1_size[2]+fudge_z, 0, 0, 0],    # part point 9
                [self.l1-c1_size[0]+fudge_x,    self.w2-c1_size[1]+fudge_y, h1-c1_size[2]+fudge_z, 0, 0, 0],    # part point 10
                [-fudge_x,                      self.w2-c1_size[1]+fudge_y, h1-c1_size[2]+fudge_z, 0, 0, 0]]    # part point 11  

        fx2=2*fudge_x
        fy2=2*fudge_y
        fz2=2*fudge_z
        # trim the intersection zone inside the part    
        # these points will be bounded by a box which will be used to trim the part cloud
        trim_points=[[corners[0][0]+fx2,            corners[0][1]+fy2,              self.t1-c1_size[2]],
                    [corners[1][0]+c1_size[0]-fx2,  corners[1][1]+fy2,              self.t1-c1_size[2]],
                    [corners[5][0]+c1_size[0]-fx2,  corners[4][1]+c1_size[1]-fy2,   self.t1-c1_size[2]],
                    [corners[4][0]+fx2,             corners[5][1]+c1_size[1]-fy2,   self.t1-c1_size[2]],
                    [corners[0][0]+fx2,             corners[0][1]+fy2,              self.t1+c1_size[2]],
                    [corners[1][0]+c1_size[0]-fx2,  corners[1][1]+fy2,              self.t1+c1_size[2]],
                    [corners[5][0]+c1_size[0]-fx2,  corners[4][1]+c1_size[1]-fy2,   self.t1+c1_size[2]],
                    [corners[4][0]+fx2,             corners[5][1]+c1_size[1]-fy2,   self.t1+c1_size[2]]]
        self.trim_part(trim_points)          

        for idx,corner in enumerate(corners):     
            # sample and create a pointcloud with a bounding box for feature1
            
            if  idx==4 or idx==5:
                f1_size=[c1_size[0], c1_size[1]*2, c1_size[2]*2]
                f1_class='inside_outside_corner'
            else:
                f1_size=c1_size
                f1_class='outside_corner'

            f1_bbox, f1_cloud, f1_indices = self.sample_part(size=f1_size, 
                                                             position=corner[0:3], 
                                                             orientation=(0,0,0) )
            f1_orientation=np.asarray(orientation)+(0,0,0)
            # only add the feature if there are points returned from sampling to avoid segmentation index error
            if len(f1_indices)>0:
                self.add_feature(f1_bbox, f1_cloud, f1_class, f1_orientation) # pass in feature orientation so the label has updated info


        plate1_xedges=[[0,1,0,0,0], # edge 0 # edges parallel to x axis
                       [3,2,np.pi/2,0,0], # edge 1
                       [7,6,2*np.pi/2,0,0],
                       [4,5,0,0,0],
                       [11,10,2*np.pi/2,0,0],
                       [8,9,3*np.pi/2,0,0]] # edge 5
     
        plate1_yedges=[[0,3,0,0,-np.pi/2], # edge 6 # edges parallet to y axis
                       [4,7,0,np.pi/2,-np.pi/2], # edge 7
                       [8,11,0,np.pi/2,-np.pi/2],
                       [9,10,0,2*np.pi/2,-np.pi/2],
                       [5,6,0,2*np.pi/2,-np.pi/2],
                       [1,2,0,3*np.pi/2,-np.pi/2]]  #edge 11   

        plate1_zedges=[[0,8,0,-np.pi/2,-np.pi/2], #edge 12 # edges parallel to z axis
                       [1,9,0,-np.pi/2,0],
                       [5,10,0,-np.pi/2,np.pi/2],
                       [2,6,0,-np.pi/2,np.pi/2],
                       [3,7,0,-np.pi/2,2*np.pi/2],
                       [4,11,0,-np.pi/2,2*np.pi/2]]  #edge 17                                         

        for edge in plate1_xedges:

            lf2=self.l1*(1-2*self.corner_fraction)
            wf2=self.w1*self.edge_fraction
            hf2=self.t1*self.edge_fraction

            if self.square_edges:
                e2=np.asarray(c1_size).min()
                l2=self.l1-2*e2
                f2_size=[l2, e2, e2]
                xf2=corners[edge[0]][0]+c1

                if edge[:2]==[4, 5]: 
                    #print('edge == [4 ,5]')
                    yf2=corners[edge[0]][1]+c1
                    zf2=corners[edge[0]][2]+c1
                    f2_class='inside_fillet'
                else:    
                    yf2=corners[edge[0]][1]
                    zf2=corners[edge[0]][2]
                    f2_class='outside_fillet'
            else:
                f2_size=[lf2, wf2, hf2]   
                xf2=corners[edge[0]][0]+lf1 

                if edge[:2]==[4, 5]: 
                    yf2=corners[edge[0]][1]+wf1
                    zf2=corners[edge[0]][2]+hf1
                    f2_class='inside_fillet'
                else:    
                    yf2=corners[edge[0]][1]
                    zf2=corners[edge[0]][2]
                    f2_class='outside_fillet'


            f2_bbox, f2_cloud, f2_indices = self.sample_part(size=f2_size, 
                                                             position=(xf2, yf2, zf2), 
                                                             orientation=(0,0,0) ) 
            f2_orientation=np.asarray(orientation)+(0,0,0)#+edge[2:]
            # only add the feature if there are points returned from sampling to avoid segmentation index error
            if len(f2_indices)>0:
                self.add_feature(f2_bbox, f2_cloud, f2_class, f2_orientation) # pass in final part orientation so the label has updated info
                                                             
        for edge in plate1_yedges:

            lf2=self.l1*self.edge_fraction
            hf2=self.t1*self.edge_fraction

            if self.square_edges:

                w2=self.w2-2*e2
                
                if edge[:2]==[4, 7] or edge[:2]==[5, 6]: 
                    w2=self.w1-self.w2-2*e2
                    yf2=corners[edge[0]][1]+2*c1
                elif edge[:2]==[8, 11] or edge[:2]==[9, 10]:
                    w2=self.w2-2*e2
                    yf2=corners[edge[0]][1]+c1
                else:    
                    w2=self.w1-2*e2
                    yf2=corners[edge[0]][1]+c1
                
                f2_size=[e2, w2, e2]

            else:
                
                if edge[:2]==[4, 7] or edge[:2]==[5, 6]: 
                    wf2=self.w1-self.w2-2*wf1
                    yf2=corners[edge[0]][1]+2*wf1
                elif edge[:2]==[8, 11] or edge[:2]==[9, 10]:
                    wf2=self.w2-2*wf1
                    yf2=corners[edge[0]][1]+wf1
                else:    
                    wf2=self.w1*(1-2*self.corner_fraction)
                    yf2=corners[edge[0]][1]+wf1
                       
                f2_size=[lf2, wf2, hf2]

            xf2=corners[edge[0]][0]
            zf2=corners[edge[0]][2]

            f2_bbox, f2_cloud, f2_indices = self.sample_part(size=f2_size, 
                                                             position=(xf2, yf2, zf2), 
                                                             orientation=(0,0,0) ) 
            f2_orientation=np.asarray(orientation)+(0,0,0)#+edge[2:]
            # only add the feature if there are points returned from sampling to avoid segmentation index error
            if len(f2_indices)>0:
                self.add_feature(f2_bbox, f2_cloud, 'outside_fillet', f2_orientation) # pass in final part orientation so the label has updated info

        for edge in plate1_zedges:

            lf2=self.l1*self.edge_fraction
            wf2=self.w1*self.edge_fraction

            if self.square_edges:
                if edge[:2]==[2, 6] or edge[:2]==[3, 7]: 
                    h2=self.t1-2*e2
                    zf2=corners[edge[0]][2]+c1
                elif edge[:2]==[5, 10] or edge[:2]==[4, 11]:
                    h2=self.t2-2*e2
                    zf2=corners[edge[0]][2]+2*c1
                else:    
                    h2=h1-2*e2
                    zf2=corners[edge[0]][2]+c1

                f2_size=[e2, e2, h2]    
                

            else:    
                if edge[:2]==[2, 6] or edge[:2]==[3, 7]: 
                    hf2=self.t1-2*hf1
                    zf2=corners[edge[0]][2]+hf1
                elif edge[:2]==[5, 10] or edge[:2]==[4, 11]:
                    hf2=self.t2-2*hf1
                    zf2=corners[edge[0]][2]+2*hf1
                else:    
                    hf2=h1-2*hf1
                    zf2=corners[edge[0]][2]+hf1

                f2_size=[lf2, wf2, hf2]

            xf2=corners[edge[0]][0]
            yf2=corners[edge[0]][1]
            
            f2_bbox, f2_cloud, f2_indices = self.sample_part(f2_size, 
                                                             position=(xf2, yf2, zf2), 
                                                             orientation=(0,0,0) )
            f2_orientation=np.asarray(orientation)+(0,0,0)#+edge[2:]
            # only add the feature if there are points returned from sampling to avoid segmentation index error
            if len(f2_indices)>0:
                self.add_feature(f2_bbox, f2_cloud, 'outside_fillet', f2_orientation) # pass in final part orientation so the label has updated info

        # after labeling add noise to the pointcloud 
        self.part_cloud = self.add_noise(self.part_cloud)              

        # after adding all peices and sampling, rotate the part and feature 
        for k,item in enumerate(self.part_items):
            item.translate(position)
            R = item.get_rotation_matrix_from_xyz(orientation) 
            item.rotate(R, center=(0,0,0))
        
        R = self.part_cloud.get_rotation_matrix_from_xyz(orientation) 
        self.part_cloud.rotate(R, center=(0,0,0))
        
        self.save_part(filename)     
  
        # after adding all peices, features (and labels) increment the part counter
        self.part_num+=1
      
        return self.part_items


    def create_1plate_partC(self, position=(0,0,0), orientation=(0,0,0), scale=1, filename='use_part_name'):    

        self.part_type='1plateC' 
        self.part_name='%06d_%s'%(self.part_set_num, self.part_type)
        self.max_plates=1
        
        if filename=='use_part_name':
            filename='scene%s'%(self.part_name)
        
        fudge_x=self.fudge[0]
        fudge_y=self.fudge[1]
        fudge_z=self.fudge[2]
  
        # make a block from two plates
        plate1, plate1_lines, plane11, plane12 = self.add_plate(size=(self.l1,self.w1,self.t1), position=(0,0,0), orientation=(0,0,0))
        plate2, plate2_lines, plane21, plane22 = self.add_plate(size=(self.l1,self.w2,self.t2), position=(0,0,self.t1), orientation=(0,0,0))
        plate3, plate3_lines, plane31, plane32 = self.add_plate(size=(self.l2,self.w1-self.w2,self.t2), position=(0,self.w2,self.t1), orientation=(0,0,0))

        # delete random regions of points from the cloud
        self.delete_regions(self.part_cloud,
                            num_regions=params.num_holes, 
                            min_radius=params.min_hole_radius, 
                            max_radius=params.max_hole_radius)
        
        lf1=self.l1*self.corner_fraction
        wf1=self.w1*self.corner_fraction
        hf1=self.t1*self.corner_fraction

        h1=self.t1+self.t2
        l12=self.l1+self.l2
        w12=self.w1+self.w2
        t12=self.t1+self.t2
        w3=self.w1-self.w2


        # define 13 corner features
        if self.square_corners:    
            c1=np.asarray([lf1, wf1, hf1]).min() # use the smallest side to define the corner size
            c1_size=(c1, c1, c1)
        else:
            c1_size=(lf1, wf1, hf1)

        #vertices of plate [x,y,z , roll, pitch, yaw] # angles not used for now
        corners=[[-fudge_x,                     -fudge_y,                       -fudge_z,                   0, 0, 0], # part point 0
                [self.l1-c1_size[0]+fudge_x,    -fudge_y,                       -fudge_z,                   0, 0, 0], # part point 1
                [self.l1-c1_size[0]+fudge_x,    self.w1-c1_size[1]+fudge_y,     -fudge_z,                   0, 0, 0],# part point 2
                [-fudge_x,                      self.w1-c1_size[1]+fudge_y,     -fudge_z,                   0,0,0],# part point 3
                [self.l2-c1_size[0]-fudge_x,    self.w2-c1_size[2]-fudge_y,     self.t1+fudge_z,            0,0,0],  # part point 4
                [self.l1-c1_size[0]+fudge_x,    self.w2-c1_size[2]+fudge_y,     self.t1-c1_size[2]+fudge_z, 0,0,0], # part point 5
                [self.l1-c1_size[0]+fudge_x,    self.w1-c1_size[1]+fudge_y,     self.t1-c1_size[2]+fudge_z, 0,0,0], # part point 6
                [self.l2-c1_size[0]+fudge_x,    self.w1-c1_size[1]+fudge_y,     self.t1-c1_size[2]+fudge_z, 0,0,0], # part point 7
                [-fudge_x,                      -fudge_y,               t12-c1_size[2]+fudge_z, 0, 0, 0],   # part point 8
                [self.l1-c1_size[0]+fudge_x,    -fudge_y,               t12-c1_size[2]+fudge_z, 0, 0, 0],   # part point 9
                [self.l1-c1_size[0]+fudge_x,    self.w2-c1_size[1]+fudge_y,     t12-c1_size[2]+fudge_z, 0, 0, 0],   # part point 10
                [self.l2-c1_size[0]+fudge_x,    self.w2-c1_size[0]+fudge_y,     t12-c1_size[2]+fudge_z, 0, 0, 0],   # part point 11
                [self.l2-c1_size[0]+fudge_x,    self.w1-c1_size[1]+fudge_y,     t12-c1_size[2]+fudge_z, 0, 0, 0],   # part point 12    
                [-fudge_x,                      self.w1-c1_size[1]+fudge_y,     t12-c1_size[2]+fudge_z, 0, 0, 0]]   # part point 13                       
        
        fx2=3*fudge_x
        fy2=3*fudge_y
        fz2=3*fudge_z
        # trim the intersection zone inside the part - volume 1   
        # these points will be bounded by a box which will be used to trim the part cloud
        trim_points1=[[corners[0][0]+fx2,           corners[0][1]+fy2,              self.t1-c1_size[2]],
                    [corners[1][0]+c1_size[0]-fx2,  corners[1][1]+fy2,              self.t1-c1_size[2]],
                    [corners[5][0]+c1_size[0]-fx2,  corners[5][1]+c1_size[1]-fy2,   self.t1-c1_size[2]],
                    [corners[0][0]+fx2,             corners[4][1]+c1_size[1]-fy2,   self.t1-c1_size[2]],
                    [corners[0][0]+fx2,             corners[0][1]+fy2,              self.t1+c1_size[2]],
                    [corners[1][0]+c1_size[0]-fx2,  corners[1][1]+fy2,              self.t1+c1_size[2]],
                    [corners[5][0]+c1_size[0]-fx2,  corners[5][1]+c1_size[1]-fy2,   self.t1+c1_size[2]],
                    [corners[0][0]+fx2,             corners[4][1]+c1_size[1]-fy2,   self.t1+c1_size[2]]]

        trim_points2=[[corners[4][0]+fx2,           corners[4][1]+fy2,              self.t1-c1_size[2]],
                    [corners[7][0]+c1_size[0]-fx2,  corners[7][1]+fy2,              self.t1-c1_size[2]],
                    [corners[13][0]+c1_size[0]-fx2, corners[13][1]+c1_size[1]-fy2,  self.t1-c1_size[2]],
                    [corners[13][0]+fx2,            corners[4][1]+c1_size[1]-fy2,   self.t1-c1_size[2]],
                    [corners[4][0]+fx2,             corners[4][1]+fy2,              h1-fz2],
                    [corners[7][0]+c1_size[0]-fx2,  corners[7][1]+fy2,              h1-fz2],
                    [corners[13][0]+c1_size[0]-fx2, corners[13][1]+c1_size[1]-fy2,  h1-fz2],
                    [corners[13][0]+fx2,            corners[4][1]+c1_size[1]-fy2,   h1-fz2]]

        self.trim_part(trim_points1)            
        self.trim_part(trim_points2)   
    
    #    corners=[[0, 0, 0, 0, 0, 0],                                         # part point 0
    #            [self.l1-c1_size[0], 0, 0, 0, 0, 0],                         # part point 1
    #            [self.l1-c1_size[0], self.w1-c1_size[1], 0, 0, 0, 0],        # part point 2
    #            [0, self.w1-c1_size[1], 0, 0, 0, 0],                         # part point 3
    #            [self.l2-c1_size[0], self.w2-c1_size[2], self.t1, 0,0,0],                          # part point 4
    #            [self.l1-c1_size[0], self.w2-c1_size[2], self.t1-c1_size[2], 0,0,0],                              # part point 5
    #            [self.l1-c1_size[0], self.w1-c1_size[1], self.t1-c1_size[2], 0,0,0],                              # part point 6
    #            [self.l2-c1_size[0], self.w1-c1_size[1], self.t1-c1_size[2], 0,0,0],                              # part point 7
    #            [0, 0, t12-c1_size[2], 0, 0, 0],                                        # part point 8
    #            [self.l1-c1_size[0], 0, t12-c1_size[2], 0, 0, 0],                                      # part point 9
    #            [self.l1-c1_size[0], self.w2-c1_size[1], t12-c1_size[2], 0, 0, 0],                                # part point 10
    #            [self.l2-c1_size[0], self.w2-c1_size[0], t12-c1_size[2], 0, 0, 0],                           # part point 11
    #            [self.l2-c1_size[0], self.w1-c1_size[1], t12-c1_size[2], 0, 0, 0],                                # part point 12    
    #            [0, self.w1-c1_size[1], t12-c1_size[2], 0, 0, 0]]                                      # part point 13                       

    #            
    #    # trim the intersection zone inside the part - volume 1   
    #    # these points will be bounded by a box which will be used to trim the part cloud
    #    trim_points1=[[corners[0][0]+fudge_x,corners[0][1]+fudge_y,self.t1-c1_size[2]],
    #                [corners[1][0]+c1_size[0]-fudge_x,corners[1][1]+fudge_y,self.t1-c1_size[2]],
    #                [corners[5][0]+c1_size[0]-fudge_x,corners[4][1]+c1_size[1]-fudge_y,self.t1-c1_size[2]],
    #                [corners[4][0]+fudge_x,corners[5][1]+c1_size[1]-fudge_y,self.t1-c1_size[2]],
    #                [corners[0][0]+fudge_x,corners[0][1]+fudge_y,self.t1+c1_size[2]],
    #                [corners[1][0]+c1_size[0]-fudge_x,corners[1][1]+fudge_y,self.t1+c1_size[2]],
    #                [corners[5][0]+c1_size[0]-fudge_x,corners[4][1]+c1_size[1]-fudge_y,self.t1+c1_size[2]],
    #                [corners[4][0]+fudge_x,corners[5][1]+c1_size[1]-fudge_y,self.t1+c1_size[2]]]

    #    trim_points2=[[corners[4][0]+fudge_x,corners[4][1]+fudge_y,self.t1-c1_size[2]],
    #                [corners[7][0]+c1_size[0]-fudge_x,corners[7][1]+fudge_y,self.t1-c1_size[2]],
    #                [corners[13][0]+c1_size[0]-fudge_x,corners[13][1]+c1_size[1]-fudge_y,self.t1-c1_size[2]],
    #                [corners[13][0]+fudge_x,corners[4][1]+c1_size[1]-fudge_y,self.t1-c1_size[2]],
    #                [corners[4][0]+fudge_x,corners[4][1]+fudge_y,h1-fudge_z],
    #                [corners[7][0]+c1_size[0]-fudge_x,corners[7][1]+fudge_y,h1-fudge_z],
    #                [corners[13][0]+c1_size[0]-fudge_x,corners[13][1]+c1_size[1]-fudge_y,h1-fudge_z],
    #                [corners[13][0]+fudge_x,corners[4][1]+c1_size[1]-fudge_y,h1-fudge_z]]

    #    self.trim_part(trim_points1)            
    #    self.trim_part(trim_points2)          
            
        for idx,corner in enumerate(corners):     
            # sample and create a pointcloud with a bounding box for feature1
            f1_position=corner[0:3]

            if idx==4:
                f1_size=[c1_size[0], c1_size[1], c1_size[2]]
                f1_position=[corner[0]+c1, corner[1]+c1, corner[2]]
                f1_class='inside_corner'
            elif idx==5:
                f1_size=[c1_size[0], c1_size[1]*2, c1_size[2]*2]
                f1_position=[corner[0], corner[1], corner[2]]
                f1_class='inside_outside_corner'
            elif idx==7:
                f1_size=[c1_size[0]*2, c1_size[1], c1_size[2]*2]
                f1_position=[corner[0], corner[1], corner[2]]
                f1_class='inside_outside_corner'
            elif idx==11:
                f1_size=[c1_size[0]*2, c1_size[1]*2, c1_size[2]]
                f1_position=[corner[0], corner[1], corner[2]]
                f1_class='inside_outside_corner'    
            else:
                f1_size=c1_size
                f1_class='outside_corner'

            f1_bbox, f1_cloud, f1_indices = self.sample_part(size=f1_size, 
                                                              position=f1_position, 
                                                              orientation=(0,0,0) )
            f1_orientation=np.asarray(orientation)+(0,0,0)
            
            # only add the feature if there are points returned from sampling to avoid segmentation index error
            if len(f1_indices)>0:
                self.add_feature(f1_bbox, f1_cloud, f1_class, f1_orientation) # pass in feature orientation so the label has updated info

        # define 7 edges for parallel to each axis
        plate1_xedges=[[0,1,0,0,0], # edge 0 # edges parallel to x axis
                       [3,2,np.pi/2,0,0], # edge 1
                       [7,6,2*np.pi/2,0,0],   # edge 2
                       [4,5,0,0,0],   # edge 3 
                       [11,10,2*np.pi/2,0,0], # edge 4
                       [13,12,2*np.pi/2,0,0], # edge 5
                       [8,9,3*np.pi/2,0,0]]   # edge 6
     
        plate1_yedges=[[0,3,0,0,-np.pi/2],   # edge 7 # edges parallet to y axis
                       [8,13,0,np.pi/2,-np.pi/2],   # edge 8 
                       [11,12,0,2*np.pi/2,-np.pi/2], # edge 9
                       [4,7,0,0,-np.pi/2],   # edge 10 
                       [5,6,0,2*np.pi/2,-np.pi/2],   # edge 11 
                       [9,10,0,2*np.pi/2,-np.pi/2],  # edge 12   
                       [1,2,0,3*np.pi/2,-np.pi/2]]   #edge 13   

        plate1_zedges=[[0,8,0,-np.pi/2,-np.pi/2], #edge 14 # edges parallel to z axis
                       [1,9,0,-np.pi/2,0], #edge 15 
                       [5,10,0,-np.pi/2,np.pi/2], # edge 16
                       [4,11,0,-np.pi/2,-np.pi/2],  # edge 17 
                       [7,12,0,-np.pi/2,np.pi/2],   # edge 18
                       [2,6,0,-np.pi/2,np.pi/2],   # edge 19
                       [3,13,0,-np.pi/2,2*np.pi/2]]  #edge 20


        for edge in plate1_xedges:

            lf2=self.l1*(1-2*self.corner_fraction)
            wf2=self.w1*self.edge_fraction
            hf2=self.t1*self.edge_fraction

            if self.square_edges:
                e2=np.asarray(c1_size).min()
                l2=self.l1-2*e2
                
                xf2=corners[edge[0]][0]+c1
                yf2=corners[edge[0]][1]
                zf2=corners[edge[0]][2]

                if edge[:2]==[4, 5]:
                    xf2=corners[edge[0]][0]+2*c1 
                    yf2=corners[edge[0]][1]+c1
                    l2=self.l1-self.l2-2*e2
                    f2_class='inside_fillet'
                elif edge[:2]==[7,6] or edge[:2]==[11,10]:
                    xf2=corners[edge[0]][0]+2*c1

                    l2=self.l1-self.l2-2*e2
                    f2_class='outside_fillet'
                elif edge[:2]==[13,12]:

                    l2=self.l2-2*e2
                    f2_class='outside_fillet'    
                else:    
                    f2_class='outside_fillet'

                f2_size=[l2, e2, e2]    

            else:
                f2_size=[lf2, wf2, hf2]   
                xf2=corners[edge[0]][0]+lf1 

                if edge[:2]==[4, 5]: 
                    yf2=corners[edge[0]][1]+wf1
                    zf2=corners[edge[0]][2]+hf1
                    f2_class='inside_fillet'
                else:    
                    yf2=corners[edge[0]][1]
                    zf2=corners[edge[0]][2]
                    f2_class='outside_fillet'


            f2_bbox, f2_cloud, f2_indices = self.sample_part(size=f2_size, 
                                                             position=(xf2, yf2, zf2), 
                                                             orientation=(0,0,0)) 
            f2_orientation=np.asarray(orientation)+(0,0,0)#+edge[2:]
            # only add the feature if there are points returned from sampling to avoid segmentation index error
            if len(f2_indices)>0:
                self.add_feature(f2_bbox, f2_cloud, f2_class, f2_orientation) # pass in final part orientation so the label has updated info
                                                             
        for edge in plate1_yedges:

            lf2=self.l1*self.edge_fraction
            hf2=self.t1*self.edge_fraction

            if self.square_edges:

                w2=self.w2-2*e2
                xf2=corners[edge[0]][0]
                yf2=corners[edge[0]][1]+c1
                zf2=corners[edge[0]][2]

                if edge[:2]==[4, 7]:
                    w2=self.w1-self.w2-2*e2
                    xf2=corners[edge[0]][0]+c1
                    yf2=corners[edge[0]][1]+2*c1 
                    f2_class='inside_fillet'
                elif edge[:2]==[11, 12] or edge[:2]==[5, 6]: 
                    w2=self.w1-self.w2-2*e2
                    yf2=corners[edge[0]][1]+2*c1
                    f2_class='outside_fillet'
                elif edge[:2]==[9, 10]:
                    w2=self.w2-2*e2
                    f2_class='outside_fillet'
                else:    
                    w2=self.w1-2*e2
                    f2_class='outside_fillet'
                
                f2_size=[e2, w2, e2]

            else:
                
                if edge[:2]==[4, 7] or edge[:2]==[5, 6]: 
                    wf2=self.w1-self.w2-2*wf1
                    yf2=corners[edge[0]][1]+2*wf1
                elif edge[:2]==[8, 11] or edge[:2]==[9, 10]:
                    wf2=self.w2-2*wf1
                    yf2=corners[edge[0]][1]+wf1
                else:    
                    wf2=self.w1*(1-2*self.corner_fraction)
                    yf2=corners[edge[0]][1]+wf1
                        
                f2_size=[lf2, wf2, hf2]

                xf2=corners[edge[0]][0]
                zf2=corners[edge[0]][2]
            

            f2_bbox, f2_cloud, f2_indices = self.sample_part(size=f2_size, 
                                                             position=(xf2, yf2, zf2), 
                                                             orientation=(0,0,0) ) 
            f2_orientation=np.asarray(orientation)+(0,0,0)#+edge[2:]
            # only add the feature if there are points returned from sampling to avoid segmentation index error
            if len(f2_indices)>0:
                self.add_feature(f2_bbox, f2_cloud, 'outside_fillet', f2_orientation) # pass in final part orientation so the label has updated info

        for edge in plate1_zedges:

            lf2=self.l1*self.edge_fraction
            wf2=self.w1*self.edge_fraction

            if self.square_edges:

                xf2=corners[edge[0]][0]
                yf2=corners[edge[0]][1]
                zf2=corners[edge[0]][2]+c1

                if edge[:2]==[4, 11]:
                    h2=self.t2-2*e2
                    xf2=corners[edge[0]][0]+c1
                    yf2=corners[edge[0]][1]+c1
                    f2_class='inside_fillet'

                elif edge[:2]==[5, 10] or edge[:2]==[7, 12]: 
                    h2=self.t2-2*e2
                    zf2=corners[edge[0]][2]+2*c1
                    f2_class='outside_fillet'

                elif edge[:2]==[2,6]:
                    h2=self.t1-2*e2
                    zf2=corners[edge[0]][2]+c1
                    f2_class='outside_fillet'

                else:    
                    h2=h1-2*e2
                    zf2=corners[edge[0]][2]+c1
                    f2_class='outside_fillet'

                f2_size=[e2, e2, h2]    
                

            else:    
                if edge[:2]==[2, 6] or edge[:2]==[3, 7]: 
                    hf2=self.t1-2*hf1
                    zf2=corners[edge[0]][2]+hf1
                elif edge[:2]==[5, 10] or edge[:2]==[4, 11]:
                    hf2=self.t2-2*hf1
                    zf2=corners[edge[0]][2]+2*hf1
                else:    
                    hf2=h1-2*hf1
                    zf2=corners[edge[0]][2]+hf1

                f2_size=[lf2, wf2, hf2]

                xf2=corners[edge[0]][0]
                yf2=corners[edge[0]][1]
            
            f2_bbox, f2_cloud, f2_indices = self.sample_part(f2_size, 
                                                             position=(xf2, yf2, zf2), 
                                                             orientation=(0,0,0))
            f2_orientation=np.asarray(orientation)+(0,0,0)#+edge[2:]
            # only add the feature if there are points returned from sampling to avoid segmentation index error
            if len(f2_indices)>0:
                self.add_feature(f2_bbox, f2_cloud, 'outside_fillet', f2_orientation) # pass in final part orientation so the label has updated info
        
        # after labeling add noise to the pointcloud
        self.part_cloud = self.add_noise(self.part_cloud)   

        # after adding all peices and sampling, rotate the part and feature 
        for k,item in enumerate(self.part_items):
            item.translate(position)
            R = item.get_rotation_matrix_from_xyz(orientation) 
            item.rotate(R, center=(0,0,0))
        
        R = self.part_cloud.get_rotation_matrix_from_xyz(orientation) 
        self.part_cloud.rotate(R, center=(0,0,0))
        
        self.save_part(filename)     
  
        # after adding all peices, features (and labels) increment the part counter
        self.part_num+=1
      
        return self.part_items    

  
    def create_2plate_partA(self, position=(0,0,0), orientation=(0,0,0), scale=1, filename='use_part_name'):
       
        self.part_type='2plateA' 
        self.part_name='%06d_%s'%(self.part_set_num, self.part_type)
        self.max_plates=2
        
        if filename=='use_part_name':
            filename='scene%s'%(self.part_name)
        
        fudge_x=self.fudge[0]
        fudge_y=self.fudge[1]
        fudge_z=self.fudge[2]
  
        # add the bottom plate of the part
        plate1, plate1_lines, plane11, plane12 = self.add_plate(size=(self.l1,self.w1,self.t1), 
                                                                position=(0,0,0), 
                                                                orientation=(0,0,0))
 
        x2=self.l12  
        y2=np.asarray([ self.w12+self.t2, self.w1 ]).min() # limit the y offset to the edge of plate (prevents gap)
        z2=self.t1
        plate2, plate2_lines, plane21, plane22 = self.add_plate(size=(self.l2,self.w2,self.t2), 
                                                                position=(x2,y2,z2), 
                                                                orientation=(np.pi/2,0,0))
                      
        # dimensions of feature bounding box based on weld geometry, this part needs to be cleaned up
        #legy=np.asarray([self.t1, self.t2]).min()/2 # calculate leg from smallest thickness plate (ask SC about this)
        legy=self.w2*self.edge_fraction # edge fraction drives the fillet size? this should be changed
        rooty=legy/40  # calculate root from leg (this is not realistic for welding, must barely pass surface to ensure catching points when sampling)  
        legz=legy      # use a square cross section, also 'root' is used incorrectly here, root is the origin of the weld (feature seam)   
        rootz=rooty   
        # fillet radius should be in the parameter set in params.py
        #radius=legy*self.fillet_radius_fraction       
        radius=self.fillet_radius

        # feature 1 lies in the xdirection on the intersection of the two plates
        lf1=np.asarray([self.l2, self.l1-self.l12]).min()-self.fudge[0]
        wf1=radius+radius*self.fillet_edge_fraction
        hf1=wf1 # fillet features have square cross section      

        #print('[lf1,wf1,hf1]:', lf1,  wf1, hf1) 
        #boxl=np.asarray([lf1, lf1+(self.l12+self.l2)]).max()
        boxl=self.l2+self.fudge[0]
        #boxw=wf1
        #boxh=hf1
        # trim the points in the interection of the two plates that are not part of the fillets
        trim_points=[[x2, y2-self.t2-radius, self.t1+radius],
                     [x2, y2+radius, self.t1+radius],
                     [x2, y2+radius, self.t1-self.fudge[2]],
                     [x2, y2-self.t2-radius, self.t1-self.fudge[2]],
                     [x2+boxl, y2-self.t2-radius, self.t1+radius],
                     [x2+boxl, y2+radius, self.t1+radius],
                     [x2+boxl, y2+radius, self.t1-self.fudge[2]],
                     [x2+boxl, y2-self.t2-radius, self.t1-self.fudge[2]]]

        self.trim_part(trim_points)
        
        # add first fillet at feature location, consider the fillet a vertical cylinder, quadrants from xy plane looking top down
        fillet1=self.add_fillet(radius=radius, 
                                length=lf1, 
                                position=[x2+lf1/2, y2+radius, z2+radius], 
                                orientation=[0,90*np.pi/180,0], 
                                quadrant=4)

        # add second fillet at feature location 
        fillet2=self.add_fillet(radius=radius, 
                                length=lf1, 
                                position=[x2+lf1/2, y2-self.t2-radius, z2+radius], 
                                orientation=[0,90*np.pi/180,0], 
                                quadrant=1)

        # trim the points off the bottom of plate1
        trim_points=[[0, 0, 0],
                     [self.l1+self.fudge[0], 0, 0],
                     [self.l1+self.fudge[0], self.w1+self.fudge[1], 0],
                     [0, self.w1+self.fudge[1], 0],
                     [0, 0, self.t1-self.fudge[2]],
                     [self.l1+self.fudge[0], 0, self.t1-self.fudge[2]],
                     [self.l1+self.fudge[0], self.w1+self.fudge[1], self.t1-self.fudge[2]],
                     [0, self.w1+self.fudge[1], self.t1-self.fudge[2]]]

        self.trim_part(trim_points)
        
        # trim the points off the top of plate2 
        boxl=self.l2
        boxw=self.t2-self.fudge[0]
        boxh=2*self.fudge[2]
        trim_points=[[x2+self.fudge[0]/2, y2-boxw, self.t1+boxh+self.w2],
                     [x2+self.fudge[0]/2, y2-self.fudge[0]/2, self.t1+boxh+self.w2],
                     [x2+self.fudge[0]/2, y2-self.fudge[0]/2, self.t1-self.fudge[2]+self.w2],
                     [x2+self.fudge[0]/2, y2-boxw, self.t1-self.fudge[2]+self.w2],
                     [x2+boxl, y2-boxw, self.t1+boxh+self.w2],
                     [x2+boxl, y2-self.fudge[0]/2, self.t1+boxh+self.w2],
                     [x2+boxl, y2-self.fudge[0]/2, self.t1-self.fudge[2]+self.w2],
                     [x2+boxl, y2-boxw, self.t1-self.fudge[2]+self.w2]]

        self.trim_part(trim_points, show_box=True)
        
        # trim the points off the first external vertical edge of plate 2
        boxl=2*self.fudge[0]
        boxw=self.t2-self.fudge[1]
        boxh=self.w2
        trim_points=[[x2-self.fudge[0],        y2-boxw,   self.t1+boxh],
                     [x2-self.fudge[0],        y2-self.fudge[1],        self.t1+boxh],
                     [x2-self.fudge[0],        y2-self.fudge[1],        self.t1],
                     [x2-self.fudge[0],        y2-boxw,   self.t1],
                     [x2+boxl+self.fudge[0],   y2-boxw,   self.t1+boxh],
                     [x2+boxl+self.fudge[0],   y2-self.fudge[1],        self.t1+boxh],
                     [x2+boxl+self.fudge[0],   y2-self.fudge[1],        self.t1],
                     [x2+boxl+self.fudge[0],   y2-boxw,   self.t1]]

        self.trim_part(trim_points)
       
        # trim the points off the second external vertical edge of plate 2
        boxl=2*self.fudge[0]
        boxw=self.t2-self.fudge[1]
        boxh=self.w2
        trim_points=[[x2+self.l2-self.fudge[0]/2,        y2-boxw,   self.t1+boxh],
                     [x2+self.l2-self.fudge[0]/2,        y2-self.fudge[1],        self.t1+boxh],
                     [x2+self.l2-self.fudge[0]/2,        y2-self.fudge[1],        self.t1],
                     [x2+self.l2-self.fudge[0]/2,        y2-boxw,   self.t1],
                     [x2+self.l2+boxl+self.fudge[0],   y2-boxw,   self.t1+boxh],
                     [x2+self.l2+boxl+self.fudge[0],   y2-self.fudge[1],        self.t1+boxh],
                     [x2+self.l2+boxl+self.fudge[0],   y2-self.fudge[1],        self.t1],
                     [x2+self.l2+boxl+self.fudge[0],   y2-boxw,   self.t1]]
        
        self.trim_part(trim_points)
        # use the shadow when feature 2 is not being used (debugging)
        
        # trim the points off the back of plate2, this was a temp hack to force similarity to real scans 
        boxw=0.5*self.w2
        trim_points=[[self.l12,         self.w12-boxw,                  self.t1],
                     [self.l12+self.l2, self.w12-boxw,                  self.t1],
                     [self.l12+self.l2, self.w12-boxw,                  self.t1+self.w2+self.fudge[2]],
                     [self.l12,         self.w12-boxw,                  self.t1+self.w2+self.fudge[2]],
                     [self.l12,         self.w12-self.fudge[1]+self.t2, self.t1],
                     [self.l12+self.l2, self.w12-self.fudge[1]+self.t2, self.t1],
                     [self.l12+self.l2, self.w12-self.fudge[1]+self.t2, self.t1+self.w2+self.fudge[2]],
                     [self.l12,         self.w12-self.fudge[1]+self.t2, self.t1+self.w2+self.fudge[2]]]

        #self.trim_part(trim_points)
        
        self.part_cloud = self.add_noise(self.part_cloud, mu=self.mu_noise, sigma=self.sigma_noise, max_noise=self.max_noise)    
        # delete random regions of points from the cloud (spherical regions for now), this needs to move back down
        self.delete_regions(self.part_cloud, num_regions=self.num_holes, min_radius=self.min_hole_radius, max_radius=self.max_hole_radius)

        dx_max=self.l0*.075
        dx=np.random.rand(1,3)*dx_max-dx_max/2
        theta_max=90*np.pi/180.0
        tmp=np.random.rand(1,1)*theta_max-theta_max/2
        dtheta=tmp[0][0] # fix this soon
        #print(dx)       
        #print(dtheta)
        # add a plate for the table , do this before adding features
   #     plate0, plate0_lines, plane01, plane02 = self.add_plate(size=(self.l0,self.w0,self.t0), 
   #                                                             position=(-self.l0/2+self.l1/2+dx[0][0],-self.w0/2+self.w1/2+dx[0][1],-self.t0), 
   #                                                             orientation=(0,0,dtheta) )
 
   #     #trim points off the bottom of the table plate0
   #     boxl=self.l0*5 # times 5 is a hack to cover entire space below rotated table... fix this but it should not cost points
   #     boxw=self.w0*5
   #     trim_points=[[-boxl/2-self.fudge[0]+dx[0][0], -boxw/2-self.fudge[1]+dx[0][1], -self.fudge[2]],
   #                  [boxl/2+self.fudge[0]+dx[0][0], -boxw/2-self.fudge[1]+dx[0][1], -self.fudge[2]],
   #                  [boxl/2+self.fudge[0]+dx[0][0],  boxw/2+self.fudge[1]+dx[0][1], -self.fudge[2]],
   #                  [-boxl/2-self.fudge[0]+dx[0][0], boxw/2+self.fudge[1]+dx[0][1], -self.fudge[2]],
   #                  [-boxl/2-self.fudge[0]+dx[0][0], -boxw/2-self.fudge[1]+dx[0][1], -self.t0-self.fudge[2]],
   #                  [boxl/2+self.fudge[0]+dx[0][0], -boxw/2-self.fudge[1]+dx[0][1], -self.t0-self.fudge[2]],
   #                  [boxl/2+self.fudge[0]+dx[0][0], boxw/2+self.fudge[1]+dx[0][1], -self.t0-self.fudge[2]],
   #                  [-boxl/2-self.fudge[0]+dx[0][0], boxw/2+self.fudge[1]+dx[0][1], -self.t0-self.fudge[2]]]
   #    # ct=np.cos(orientation[0])
   #    # st=np.sin(orientation[0])
   #    # Rx=[[1, 0, 0], 
   #    #     [0, ct, -st],
   #    #     [0, st, ct]]
   #    # ct=np.cos(orientation[1])
   #    # st=np.sin(orientation[1])
   #    # Ry=[[ct, 0, st],
   #    #     [0,  1, 0], 
   #    #     [-st, 0, ct]]
   #     ct=np.cos(dtheta)
   #     st=np.sin(dtheta)
   #     Rz=[[ct, -st, 0],
   #         [st, ct, 0], 
   #         [0,  0,  1]]
   #     #Rxy=np.matmul(Rx,Ry)
   #     #Rxyz=np.matmul(Rxy,Rz)
   #     #print(trim_points) # this rotated trim is not working for some reason, fix this soon
   #     trim_points_rot=np.matmul(trim_points,Rz) 
   #     self.trim_part(trim_points)
   #     #print(trim_points_rot)       

   #    
   #     #trim the shadow beneath the part from the table
   #     trim_points=[[self.fudge[0],self.fudge[1], self.t1/2],
   #                  [self.l1-self.fudge[0], self.fudge[1], self.t1/2],
   #                  [self.l1-self.fudge[0], self.w1+self.fudge[1], self.t1/2],
   #                  [self.fudge[0], self.w1+self.fudge[1], self.t1/2],
   #                  [self.fudge[0],self.fudge[1],-self.t0-self.fudge[2]],
   #                  [self.l1-self.fudge[0], self.fudge[1], -self.t0-self.fudge[2]],
   #                  [self.l1-self.fudge[0], self.w1+self.fudge[1], -self.t0-self.fudge[2]],
   #                  [self.fudge[0], self.w1+self.fudge[1], -self.t0-self.fudge[2]] ]
   #     
   #     self.trim_part(trim_points, show_box=True)
        
        
        
        #trim_points_rot=np.matmul(trim_points,Rz)
        #print('[lf1,wf1,hf1]:', lf1,  wf1, hf1) 
        # sample and create a pointcloud with a bounding box for feature1
        feat1_bbox, feat1_cloud, feat1_indices = self.sample_part(size=(lf1, wf1, hf1), 
                                                                  position=(x2+self.fudge[0]/2, self.w12+self.t2-self.fudge[1], self.t1-rooty), 
                                                                  orientation=(0,0,0) )
        self.add_feature(feat1_bbox, feat1_cloud, 'inside_fillet', position=position, orientation=orientation) 
        
        # feature 2 is the same but on the opposite side - f2 disabled
        # sample and create a pointcloud with a bounding box for feature 2
        feat2_bbox, feat2_cloud, feat2_indices = self.sample_part(size=(lf1, wf1, hf1), 
                                                                  position=(x2+self.fudge[0]/2, self.w12-wf1+self.fudge[1], self.t1-rooty), 
                                                                  orientation=(0,0,0) )
        f2_orientation=(0,0,np.pi)
        self.add_feature(feat2_bbox, feat2_cloud, 'inside_fillet', position=position, 
                                                                   orientation=orientation,
                                                                   feature_orientation=f2_orientation)

        # after adding all peices and sampling, rotate the part and feature 
        for k,item in enumerate(self.part_items):
            R = item.get_rotation_matrix_from_xyz(orientation) 
            item.rotate(R, center=(0,0,0)) # be sure to use center=(0,0,0)
            item.translate(position)
        
        # rotate the part cloud as well 
        R = self.part_cloud.get_rotation_matrix_from_xyz(orientation) 
        self.part_cloud.rotate(R, center=(0,0,0)) # use center=(0,0,0) here too
        self.part_cloud.translate(position)


        #print(len(self.part_cloud.points))
        # after labeling add noise to the pointcloud, testing this above instead
        #self.part_cloud = self.add_noise(self.part_cloud, mu=self.mu_noise, sigma=self.sigma_noise, max_noise=self.max_noise)    
        #print('num points:', len(self.part_cloud.points))

        self.save_part(filename)     
        # after adding all peices, features (and labels) increment the part counter
        self.part_num+=1
      
        return self.part_items


    def create_3plate_partA(self, position=(0,0,0), orientation=(0,0,0), scale=1, filename='use_part_name'):
        
        self.part_type='3plateA' 
        self.part_name='%06d_%s'%(self.part_set_num, self.part_type)
        self.max_plates=3
        
        if filename=='use_part_name':
            filename='scene%s'%(self.part_name)
       
        fudge_x=self.fudge[0]
        fudge_y=self.fudge[1]
        fudge_z=self.fudge[2]

        plate1, plate1_lines, plane11, plane12 = self.add_plate(size=(self.l1,self.w1,self.t1), 
                                                                position=(0,0,0), 
                                                                orientation=(0,0,0))   
        x2=self.l12
        y2=np.asarray([ self.w12+self.t2, self.w1 ]).min() # limit the y offset to the edge of plate (prevents gap)
        z2=self.t1
        plate2, plate2_lines, plane21, plane22 = self.add_plate(size=(self.l2, self.w2, self.t2), 
                                                                position=(x2,y2,z2), 
                                                                orientation=(np.pi/2,0,0))
     
        x3=np.asarray([ x2+self.l23, self.l1 ]).min() # limit the y offset to the edge of plate (prevents gap)
        y3=np.asarray([ self.w12-self.l3, self.l3 ]).min() # limit the y offset to the edge of plate (prevents gap)
        z3=self.t1        
        plate3, plate3_lines, plane31, plane32 = self.add_plate(size=(self.l3,self.w3,self.t3), 
                                                                position=(x3,y3,z3), 
                                                                orientation=(np.pi/2,np.pi/2,0))
        
        # trim the points off the bottom of plate1
        trim_points=[[0, 0, 0],
                     [self.l1+self.fudge[0], 0, 0],
                     [self.l1+self.fudge[0], self.w1+self.fudge[1], 0],
                     [0, self.w1+self.fudge[1], 0],
                     [0, 0, self.t1-self.fudge[2]],
                     [self.l1+self.fudge[0], 0, self.t1-self.fudge[2]],
                     [self.l1+self.fudge[0], self.w1+self.fudge[1], self.t1-self.fudge[2]],
                     [0, self.w1+self.fudge[1], self.t1-self.fudge[2]]]

        self.trim_part(trim_points)
        
        radius=self.fillet_radius 
        # trim the points in the interection of plates 1 and plate 2 that are not part of the fillets
        boxl=self.l2+self.fudge[0]
        trim_points=[[x2, y2-self.t2-radius, self.t1+radius],
                     [x2, y2+radius, self.t1+radius],
                     [x2, y2+radius, self.t1-self.fudge[2]],
                     [x2, y2-self.t2-radius, self.t1-self.fudge[2]],
                     [x2+boxl, y2-self.t2-radius, self.t1+radius],
                     [x2+boxl, y2+radius, self.t1+radius],
                     [x2+boxl, y2+radius, self.t1-self.fudge[2]],
                     [x2+boxl, y2-self.t2-radius, self.t1-self.fudge[2]]]

        self.trim_part(trim_points)
        
        # feature 1 lies in the xdirection on the intersection of the two plates
        lf1=np.asarray([self.l2, self.l1-self.l12]).min()-self.fudge[0]
        wf1=radius+radius*self.fillet_edge_fraction
        hf1=wf1 # fillet features have square cross section      
        # this is still just adding geometry, not labeled features, that is below
        # add first fillet at feature location 
        fillet1=self.add_fillet(radius=radius, 
                                length=lf1, 
                                position=[x2+lf1/2, y2+radius, z2+radius], 
                                orientation=[0,90*np.pi/180,0], 
                                quadrant=4)

        # add fillet2 at feature location
        lf2=self.l1-self.l12 
        fillet2=self.add_fillet(radius=radius, 
                                length=lf1, 
                                position=[x2+lf1/2, y2-self.t2-radius, z2+radius], 
                                orientation=[0,90*np.pi/180,0], 
                                quadrant=1)
        
        lf3=np.asarray([self.l3, self.w12]).min()-radius
        #wf3=wf1                                                    
        #hf3=hf1                                                    
        # trim the points in the interection of plate 1 and plate 3 that are not part of the fillets
        boxl=lf3
        trim_points=[[x3-radius, 0, self.t1+radius],
                     [x3-radius, 0, self.t1-self.fudge[2]],
                     [x3-radius, boxl, self.t1+radius],
                     [x3-radius, boxl, self.t1-self.fudge[2]],
                     [x3+self.t3+radius, 0, self.t1+radius],
                     [x3+self.t3+radius, 0, self.t1-self.fudge[2]],
                     [x3+self.t3+radius, boxl, self.t1+radius],
                     [x3+self.t3+radius, boxl, self.t1-self.fudge[2]]]
        
        self.trim_part(trim_points)
        
        # add fillet3 at feature location, orthogonal to fillet1, fillet2
        fillet3=self.add_fillet(radius=radius, 
                                length=lf3, 
                                position=[self.l12+self.l23-radius, lf3/2, z2+radius], 
                                orientation=[90*np.pi/180.0,0*np.pi/180,0*np.pi/180], 
                                quadrant=4)
        
        # add fillet4 at feature location, orthogonal to fillet1, fillet2
        fillet3=self.add_fillet(radius=radius, 
                                length=lf3, 
                                position=[self.l12+self.l23+self.t3+radius, lf3/2, z2+radius], 
                                orientation=[90*np.pi/180.0,0*np.pi/180,0*np.pi/180], 
                                quadrant=3)
       
        lf5=np.asarray([self.w3, self.w2]).min()-radius
        # trim the points in the interection of plate 1 and plate 3 that are not part of the fillets
        trim_points=[[x3-radius, y2-self.t2-radius, self.t1+lf5+radius],
                     [x3-radius, y2-self.t2-radius, self.t1-self.fudge[2]+radius],
                     [x3-radius, y2-self.t2, self.t1+lf5+radius],
                     [x3-radius, y2-self.t2, self.t1-self.fudge[2]+radius],
                     [x3+self.t3+radius, y2-self.t2-radius, self.t1+lf5+radius],
                     [x3+self.t3+radius, y2-self.t2-radius, self.t1-self.fudge[2]+radius],
                     [x3+self.t3+radius, y2-self.t2, self.t1+lf5+radius],
                     [x3+self.t3+radius, y2-self.t2, self.t1-self.fudge[2]+radius]]
        
        self.trim_part(trim_points, show_box=True)
        
        # add fillet5 at feature location, vertical and orthogonal to fillet3
        fillet5=self.add_fillet(radius=radius, 
                                length=lf5, 
                                position=[self.l12+self.l23-radius,y2-self.t2-radius, z2+self.w3/2+radius/2], 
                                orientation=[0*np.pi/180.0,0*np.pi/180,0*np.pi/180], 
                                quadrant=1, show_box=True)
        
        # add fillet6 at feature location, vertical and orthogonal to fillet4
        fillet6=self.add_fillet(radius=radius, 
                                length=np.asarray([self.w3, self.w2]).min()-radius, 
                                position=[self.l12+self.l23+self.t3+radius,y2-self.t2-radius, z2+self.w3/2+radius/2], 
                                orientation=[0*np.pi/180.0,0*np.pi/180,0*np.pi/180], 
                                quadrant=2, show_box=True)
        
        # trim the points under the corners, only one box needed 
        boxl=lf3
        trim_points=[[x3-radius, y2-self.t2-radius, self.t1+radius],
                     [x3-radius, y2-self.t2-radius, self.t1-self.fudge[2]],
                     [x3-radius, y2-self.t2, self.t1+radius],
                     [x3-radius, y2-self.t2, self.t1-self.fudge[2]],
                     [x3+self.t3+radius, y2-self.t2-radius, self.t1+radius],
                     [x3+self.t3+radius, y2-self.t2-radius, self.t1-self.fudge[2]],
                     [x3+self.t3+radius, y2-self.t2, self.t1+radius],
                     [x3+self.t3+radius, y2-self.t2, self.t1-self.fudge[2]]]
        
        self.trim_part(trim_points)

 
        # add corner1 at feature location
        corner1=self.add_corner(radius=radius, 
                                position=[self.l12+self.l23-radius, y2-self.t2-radius, z2+radius], 
                                orientation=[0,0,0], 
                                octant=5) # eigth octants in a sphere
        
        # add corner2 at feature location
        corner2=self.add_corner(radius=radius, 
                                position=[self.l12+self.l23+self.t3+radius, y2-self.t2-radius, z2+radius], 
                                orientation=[0,0,0], 
                                octant=6)

        # trim the points in the interection of plate2 and plate1 that are not part of the fillets
        boxl=self.l2
        boxw=self.t2-self.fudge[0]
        trim_points=[[x2+self.fudge[0]/2, y2-boxw, self.t1+boxw],
                     [x2+self.fudge[0]/2, y2-self.fudge[0]/2, self.t1+boxw],
                     [x2+self.fudge[0]/2, y2-self.fudge[0]/2, self.t1-self.fudge[2]],
                     [x2+self.fudge[0]/2, y2-boxw, self.t1-self.fudge[2]],
                     [x2+boxl, y2-boxw, self.t1+boxw],
                     [x2+boxl, y2-self.fudge[0]/2, self.t1+boxw],
                     [x2+boxl, y2-self.fudge[0]/2, self.t1-self.fudge[2]],
                     [x2+boxl, y2-boxw, self.t1-self.fudge[2]]]

        self.trim_part(trim_points)

        # trim the points off the top of plate2 
        boxl=self.l2
        boxw=self.t2-self.fudge[0]
        boxh=2*self.fudge[2]
        trim_points=[[x2+self.fudge[0]/2, y2-boxw, self.t1+boxh+self.w2],
                     [x2+self.fudge[0]/2, y2-self.fudge[0]/2, self.t1+boxh+self.w2],
                     [x2+self.fudge[0]/2, y2-self.fudge[0]/2, self.t1-self.fudge[2]+self.w2],
                     [x2+self.fudge[0]/2, y2-boxw, self.t1-self.fudge[2]+self.w2],
                     [x2+boxl, y2-boxw, self.t1+boxh+self.w2],
                     [x2+boxl, y2-self.fudge[0]/2, self.t1+boxh+self.w2],
                     [x2+boxl, y2-self.fudge[0]/2, self.t1-self.fudge[2]+self.w2],
                     [x2+boxl, y2-boxw, self.t1-self.fudge[2]+self.w2]]

        self.trim_part(trim_points)

        # trim the points in the interection of plate3 and plate1 that are not part of the fillets
        boxl=self.t3-self.fudge[0]
        boxw=self.l3
        boxh=boxl
        trim_points=[[x2+self.l23, y2-boxw-self.t2+self.fudge[0]/2, self.t1+boxh],
                     [x2+self.l23+self.fudge[0]/2, y2-self.fudge[0]/2, self.t1+boxh],
                     [x2+self.l23+self.fudge[0]/2, y2-self.fudge[0]/2, self.t1-self.fudge[2]],
                     [x2+self.l23, y2-boxw-self.t2+self.fudge[0]/2, self.t1-self.fudge[2]],
                     [x2+boxl+self.l23, y2-boxw-self.t2+self.fudge[0]/2, self.t1+boxh],
                     [x2+boxl+self.l23+self.fudge[0]/2, y2-self.fudge[0]/2, self.t1+boxh],
                     [x2+boxl+self.l23+self.fudge[0]/2, y2-self.fudge[0]/2, self.t1-self.fudge[2]],
                     [x2+boxl+self.l23, y2-boxw-self.t2+self.fudge[0]/2, self.t1-self.fudge[2]]]

        self.trim_part(trim_points)
        
        # trim points off of the top of plate 3
        boxh=2*self.fudge[2]
        trim_points=[[x2+self.l23, y2-boxw-self.t2+self.fudge[0]/2, self.t1+boxh+self.w3],
                     [x2+self.l23+self.fudge[0]/2, y2-self.fudge[0]/2, self.t1+boxh+self.w3],
                     [x2+self.l23+self.fudge[0]/2, y2-self.fudge[0]/2, self.t1-self.fudge[2]+self.w3],
                     [x2+self.l23, y2-boxw-self.t2+self.fudge[0]/2, self.t1-self.fudge[2]+self.w3],
                     [x2+boxl+self.l23, y2-boxw-self.t2+self.fudge[0]/2, self.t1+boxh+self.w3],
                     [x2+boxl+self.l23+self.fudge[0]/2, y2-self.fudge[0]/2, self.t1+boxh+self.w3],
                     [x2+boxl+self.l23+self.fudge[0]/2, y2-self.fudge[0]/2, self.t1-self.fudge[2]+self.w3],
                     [x2+boxl+self.l23, y2-boxw-self.t2+self.fudge[0]/2, self.t1-self.fudge[2]+self.w3]]

        self.trim_part(trim_points)
        # trim the points in the interection of plate2 and plate3 that are not part of the fillets
        boxl=self.t3-self.fudge[0]
        boxw=self.t2-self.fudge[1]
        boxh=self.w3
        trim_points=[[x2+self.l23+self.fudge[0]/2,        y2-boxw-self.fudge[1],   self.t1+boxh],
                     [x2+self.l23+self.fudge[0]/2,        y2-self.fudge[1],        self.t1+boxh],
                     [x2+self.l23+self.fudge[0]/2,        y2-self.fudge[1],        self.t1],
                     [x2+self.l23+self.fudge[0]/2,        y2-boxw-self.fudge[1],   self.t1],
                     [x2+boxl+self.l23-self.fudge[0]/2,   y2-boxw-self.fudge[1],   self.t1+boxh],
                     [x2+boxl+self.l23-self.fudge[0]/2,   y2-self.fudge[1],        self.t1+boxh],
                     [x2+boxl+self.l23-self.fudge[0]/2,   y2-self.fudge[1],        self.t1],
                     [x2+boxl+self.l23-self.fudge[0]/2,   y2-boxw-self.fudge[1],   self.t1]]

        self.trim_part(trim_points)

        # trim the points off the external vertical edge of plate 3
        boxw=2*self.fudge[1]
        trim_points=[[x2+self.l23+self.fudge[0]/2,        y2-self.t2-boxw-self.fudge[1]-self.l3,   self.t1+boxh],
                     [x2+self.l23+self.fudge[0]/2,        y2-self.t2+self.fudge[1]-self.l3,        self.t1+boxh],
                     [x2+self.l23+self.fudge[0]/2,        y2-self.t2+self.fudge[1]-self.l3,        self.t1],
                     [x2+self.l23+self.fudge[0]/2,        y2-self.t2-boxw-self.fudge[1]-self.l3,   self.t1],
                     [x2+boxl+self.l23-self.fudge[0]/2,   y2-self.t2-boxw-self.fudge[1]-self.l3,   self.t1+boxh],
                     [x2+boxl+self.l23-self.fudge[0]/2,   y2-self.t2+self.fudge[1]-self.l3,        self.t1+boxh],
                     [x2+boxl+self.l23-self.fudge[0]/2,   y2-self.t2+self.fudge[1]-self.l3,        self.t1],
                     [x2+boxl+self.l23-self.fudge[0]/2,   y2-self.t2-boxw-self.fudge[1]-self.l3,   self.t1]]

        self.trim_part(trim_points)
        
        # trim the points off the first external vertical edge of plate 2
        boxl=2*self.fudge[0]
        boxw=self.t2-self.fudge[1]
        boxh=self.w2
        trim_points=[[x2-self.fudge[0],        y2-boxw,   self.t1+boxh],
                     [x2-self.fudge[0],        y2-self.fudge[1],        self.t1+boxh],
                     [x2-self.fudge[0],        y2-self.fudge[1],        self.t1],
                     [x2-self.fudge[0],        y2-boxw,   self.t1],
                     [x2+boxl+self.fudge[0],   y2-boxw,   self.t1+boxh],
                     [x2+boxl+self.fudge[0],   y2-self.fudge[1],        self.t1+boxh],
                     [x2+boxl+self.fudge[0],   y2-self.fudge[1],        self.t1],
                     [x2+boxl+self.fudge[0],   y2-boxw,   self.t1]]

        self.trim_part(trim_points)
       
        # trim the points off the second external vertical edge of plate 2
        boxl=2*self.fudge[0]
        boxw=self.t2-self.fudge[1]
        boxh=self.w2
        trim_points=[[x2+self.l2-self.fudge[0]/2,        y2-boxw,   self.t1+boxh],
                     [x2+self.l2-self.fudge[0]/2,        y2-self.fudge[1],        self.t1+boxh],
                     [x2+self.l2-self.fudge[0]/2,        y2-self.fudge[1],        self.t1],
                     [x2+self.l2-self.fudge[0]/2,        y2-boxw,   self.t1],
                     [x2+self.l2+boxl+self.fudge[0],   y2-boxw,   self.t1+boxh],
                     [x2+self.l2+boxl+self.fudge[0],   y2-self.fudge[1],        self.t1+boxh],
                     [x2+self.l2+boxl+self.fudge[0],   y2-self.fudge[1],        self.t1],
                     [x2+self.l2+boxl+self.fudge[0],   y2-boxw,   self.t1]]

        self.trim_part(trim_points)
        # trim the points off the bottom of plate1
        trim_points=[[0, 0, 0],
                     [self.l1+self.fudge[0], 0, 0],
                     [self.l1+self.fudge[0], self.w1+self.fudge[1], 0],
                     [0, self.w1+self.fudge[1], 0],
                     [0, 0, self.t1-self.fudge[2]],
                     [self.l1+self.fudge[0], 0, self.t1-self.fudge[2]],
                     [self.l1+self.fudge[0], self.w1+self.fudge[1], self.t1-self.fudge[2]],
                     [0, self.w1+self.fudge[1], self.t1-self.fudge[2]]]

        #self.trim_part(trim_points)
        
        # trim the points off the back of plate2, this was a temp hack to force similarity to real scans 
        boxw=0.5*self.w2
        trim_points=[[self.l12, self.w2, self.t1],
                    [self.l12+self.l2, self.w2, self.t1],
                    [self.l12+self.l2, self.w2, self.t1+self.w2],
                    [self.l12, self.w2, self.t1+self.w2],
                    [self.l12, self.w12+self.fudge[1], self.t1],
                    [self.l12+self.l2, self.w12+self.fudge[1], self.t1],
                    [self.l12+self.l2, self.w12+self.fudge[1], self.t1+self.w2],
                    [self.l12, self.w12+self.fudge[1], self.t1+self.w2]]

        #self.trim_part(trim_points)
        
   #     # add the table
   #     dx_max=self.l0*.10
   #     dx=np.random.rand(1,3)*dx_max-dx_max/2
   #     theta_max=90*np.pi/180.0
   #     tmp=np.random.rand(1,1)*theta_max-theta_max/2
   #     dtheta=tmp[0][0] # fix this soon
   #     #print(dx)       
   #     #print(dtheta)
   #     # add a plate for the table , do this before adding features
   #     plate0, plate0_lines, plane01, plane02 = self.add_plate(size=(self.l0,self.w0,self.t0), 
   #                                                             position=(-self.l0/2+dx[0][0],-self.w0/2+dx[0][1],-self.t0), 
   #                                                             orientation=(0,0,dtheta) )
 
   #     #trim points off the bottom of the table plate0
   #     boxl=self.l0*5 # times 5 is a hack to cover entire space below rotated table... fix this but it should not cost points
   #     boxw=self.w0*5
   #     trim_points=[[-boxl/2-self.fudge[0]+dx[0][0], -boxw/2-self.fudge[1]+dx[0][1], -self.fudge[2]],
   #                  [boxl/2+self.fudge[0]+dx[0][0], -boxw/2-self.fudge[1]+dx[0][1], -self.fudge[2]],
   #                  [boxl/2+self.fudge[0]+dx[0][0],  boxw/2+self.fudge[1]+dx[0][1], -self.fudge[2]],
   #                  [-boxl/2-self.fudge[0]+dx[0][0], boxw/2+self.fudge[1]+dx[0][1], -self.fudge[2]],
   #                  [-boxl/2-self.fudge[0]+dx[0][0], -boxw/2-self.fudge[1]+dx[0][1], -self.t0-self.fudge[2]],
   #                  [boxl/2+self.fudge[0]+dx[0][0], -boxw/2-self.fudge[1]+dx[0][1], -self.t0-self.fudge[2]],
   #                  [boxl/2+self.fudge[0]+dx[0][0], boxw/2+self.fudge[1]+dx[0][1], -self.t0-self.fudge[2]],
   #                  [-boxl/2-self.fudge[0]+dx[0][0], boxw/2+self.fudge[1]+dx[0][1], -self.t0-self.fudge[2]]]
   #    # ct=np.cos(orientation[0])
   #    # st=np.sin(orientation[0])
   #    # Rx=[[1, 0, 0], 
   #    #     [0, ct, -st],
   #    #     [0, st, ct]]
   #    # ct=np.cos(orientation[1])
   #    # st=np.sin(orientation[1])
   #    # Ry=[[ct, 0, st],
   #    #     [0,  1, 0], 
   #    #     [-st, 0, ct]]
   #     ct=np.cos(dtheta)
   #     st=np.sin(dtheta)
   #     Rz=[[ct, -st, 0],
   #         [st, ct, 0], 
   #         [0,  0,  1]]
   #     #Rxy=np.matmul(Rx,Ry)
   #     #Rxyz=np.matmul(Rxy,Rz)
   #     #print(trim_points) # this rotated trim is not working for some reason, fix this soon
   #     trim_points_rot=np.matmul(trim_points,Rz) 
   #     self.trim_part(trim_points, show_box=True)
   #     #print(trim_points_rot)       

   #     #trim the shadow beneath the part from the table
   #     trim_points=[[self.fudge[0],self.fudge[1], self.t1/2],
   #                  [self.l1-self.fudge[0], self.fudge[1], self.t1/2],
   #                  [self.l1-self.fudge[0], self.w1+self.fudge[1], self.t1/2],
   #                  [self.fudge[0], self.w1+self.fudge[1], self.t1/2],
   #                  [self.fudge[0],self.fudge[1],-self.t0-self.fudge[2]],
   #                  [self.l1-self.fudge[0], self.fudge[1], -self.t0-self.fudge[2]],
   #                  [self.l1-self.fudge[0], self.w1+self.fudge[1], -self.t0-self.fudge[2]],
   #                  [self.fudge[0], self.w1+self.fudge[1], -self.t0-self.fudge[2]] ]
   #     
   #     self.trim_part(trim_points, show_box=True)
        
        # dimensions of feature bounding box based on weld geometry, REMOVE THIS SOON, NO MORE LEGS AND ROOTS
        #legy=np.asarray([self.t1, self.t2]).min()/2 # calculate leg from smallest thickness plate (ask SC about this)
        legy=self.w2*self.edge_fraction 
        rooty=legy/40  # calculate root from leg (must past surface to ensure catching points when sampling)  
        legz=legy      # use a square cross section   
        rootz=rooty   
        legx=legy         
        rootx=rooty 

        corner_radius=self.fillet_radius # corners have same radius as fillets for now, this avoids overlapping boxes
        corner_length=corner_radius+corner_radius*self.corner_edge_fraction 
        fillet_radius=self.fillet_radius
        
        # feature 1 is an inside fillet contained by a single rectangular bounding box 
        # it lies in the xdirection on the intersection of the two plates
        lf1=np.asarray([self.l2, self.l1-x2]).min()-self.fudge[0]
        wf1=fillet_radius+fillet_radius*self.fillet_edge_fraction
        hf1=wf1 # fillet features have square cross section      
        f1_bbox, f1_cloud, f1_indices = self.sample_part( size=(lf1, wf1, hf1), 
                                                          position=(x2+self.fudge[0]/2,y2-self.fudge[1], z2-rooty), 
                                                          orientation=(0,0,0) )
        f1_orientation=(0,0,0) # votenet show_oriented_boxes() assumes f1 orientation is (0,0,0) for now
        self.add_feature(f1_bbox, f1_cloud, 'inside_fillet', 
                         position=position, 
                         orientation=orientation,
                         feature_orientation=f1_orientation) 
        
        
        # feature 2 is another inside fillet contained by a single rectangular bounding box 
        lf2=self.l23-corner_length
        wf2=wf1
        hf2=wf1
        f2_bbox, f2_cloud, f2_indices = self.sample_part( size=(lf2, wf2, hf2), 
                                                          position=(x2+self.fudge[0],
                                                                    y2-self.t2-wf2+self.fudge[1], 
                                                                    z2-self.fudge[2]), 
                                                          orientation=(0,0,0) )
        f2_orientation=(0,0,-np.pi) # fillets on back side of plate 2 rotated by 180 deg
        self.add_feature(f2_bbox, f2_cloud, 'inside_fillet', position=position, 
                                                             orientation=orientation,
                                                             feature_orientation=f2_orientation)

        # feature 3 is the continuation of feature 2 on the other side of plate 3 
        lf3=np.asarray([self.l2, self.l2-(self.l23+self.l2-self.l1)]).min()-self.l23-self.t3-corner_length
        wf3=wf1
        hf3=wf1
        f3_bbox, f3_cloud, f3_indices = self.sample_part( size=(lf3, wf3, hf3), 
                                                          position=(x2+self.l23+self.t3+corner_length,
                                                                    y2-self.t2-wf3+self.fudge[1], 
                                                                    z2-self.fudge[2]), 
                                                          orientation=(0,0,0) )
        f3_orientation=(0,0,-np.pi)                                 
        self.add_feature(f3_bbox, f3_cloud, 'inside_fillet', position=position, 
                                                             orientation=orientation,
                                                             feature_orientation=f3_orientation)
                                                                    
        # feature 4 is another fillet that is orthogonal to the others
        #lf4=np.asarray([self.l3, self.w1-self.w12-self.t2]).min()-wf3
        lf4=np.asarray([self.l3, self.w12]).min()-wf3
        wf4=wf1                                                    
        hf4=hf1                                                    
        f4_bbox, f4_cloud, f4_indices = self.sample_part( size=(lf4, wf4, hf4), 
                                                          position=(x2+self.l23+self.fudge[0],
                                                                    y2-lf4-self.t2-wf3, 
                                                                    z2-self.fudge[2]), 
                                                          orientation=(0,0,np.pi/2))
        f4_orientation=(0,0,-np.pi/2)
        self.add_feature(f4_bbox, f4_cloud, 'inside_fillet', position=position, 
                                                             orientation=orientation,
                                                             feature_orientation=f4_orientation,
                                                             orientation_flag='yaxis' )
        
        # feature 5 is another fillet that is orthogonal to the others just like feature 4 on the other side of plate 3
        lf5=lf4
        wf5=wf1
        hf5=hf1
        f5_bbox, f5_cloud, f5_indices = self.sample_part( size=(lf5, wf5, hf5), 
                                                          position=(x2+self.l23+self.t3+wf5-self.fudge[0],
                                                                    y2-lf4-self.t2-wf3, 
                                                                    z2-self.fudge[2]), 
                                                          orientation=(0,0,np.pi/2))
        f5_orientation=(0,0,np.pi/2) 
        self.add_feature(f5_bbox, f5_cloud, 'inside_fillet', position=position, 
                                                             orientation=orientation,
                                                             feature_orientation=f5_orientation,
                                                             orientation_flag='yaxis' )
        
        # feature 6 is a vertical fillet made from plate 2 and 3
        lf6=np.asarray([self.w3, self.w2]).min()-hf3
        wf6=wf1
        hf6=hf1
        f6_bbox, f6_cloud, f6_indices = self.sample_part( size=(lf6, wf6, hf6), 
                                                          position=(x2+self.l23+self.t3+wf5-self.fudge[0],
                                                                    y2-self.t2-wf6+self.fudge[1], 
                                                                    z2-self.fudge[2]+hf1), 
                                                          orientation=(0,-np.pi/2,0) )
        f6_orientation=(0,np.pi/2,np.pi) 
        self.add_feature(f6_bbox, f6_cloud, 'inside_fillet', position=position, 
                                                             orientation=orientation,
                                                             feature_orientation=f6_orientation,
                                                             orientation_flag='zaxis')
                                                                                               
        # feature 7 is the other vertical fillet made from plate 2 and 3                       
        lf7=np.asarray([self.w3, self.w2]).min()-hf3                                           
        wf7=wf1                                                                                
        hf7=hf1                                                                                
        f7_bbox, f7_cloud, f7_indices = self.sample_part( size=(lf7, wf7, hf7),                
                                                          position=(x2+self.l23+self.fudge[0], 
                                                                    y2-self.t2-wf7+self.fudge[1], 
                                                                    z2-self.fudge[2]+hf1),     
                                                          orientation=(0,-np.pi/2,0) )         
        f7_orientation=(0,-np.pi/2,np.pi)                                                      
        self.add_feature(f7_bbox, f7_cloud, 'inside_fillet', position=position,                
                                                             orientation=orientation,          
                                                             feature_orientation=f7_orientation,
                                                             orientation_flag='zaxis')
                                                                                               
        # corner 1 - feature 8                                                                 
        corner_radius=self.fillet_radius # corners have same radius as fillets for now, this avoids overlapping boxes
        lf8=corner_radius+corner_radius*self.corner_edge_fraction                             
        f8_bbox, f8_cloud, f8_indices = self.sample_part(size=(lf8,lf8,lf8),                  
                                                         position=(x2+self.l23+self.t3-self.fudge[0], 
                                                                   y2-self.t2-lf8+self.fudge[1], 
                                                                   z2-self.fudge[2]),         
                                                         orientation=(0,0,0))                 
                                                                                              
        f8_orientation=(0,0,np.pi/2) # first corner rotated by 180 degs                       
        self.add_feature(f8_bbox, f8_cloud, 'inside_corner', position=position,               
                                                             orientation=orientation,
                                                             feature_orientation=f8_orientation)
    
        #corner 2 - feature 9
        lf9=lf8 # same size as previous corner 
        f9_bbox, f9_cloud, f9_indices = self.sample_part(size=(lf9,lf9,lf9), 
                                                         position=(x2+self.l23-lf9+self.fudge[0], 
                                                                   y2-self.t2-lf9+self.fudge[1], 
                                                                   z2-self.fudge[2]), 
                                                         orientation=(0,0,0))

        f9_orientation=(0,0,np.pi) # first corner rotated by 180 degs
        self.add_feature(f9_bbox, f9_cloud, 'inside_corner', position=position, 
                                                             orientation=orientation,
                                                             feature_orientation=f9_orientation)
    
        # trimming after the last add_feature breaks the segs, dont this here 

        # after adding all peices and sampling, rotate the part and feature 
        for k,item in enumerate(self.part_items):
            R = item.get_rotation_matrix_from_xyz(orientation) 
            item.rotate(R, center=(0,0,0)) # be sure to use center=(0,0,0)
            item.translate(position)
        
        # rotate the part cloud as well 
        R = self.part_cloud.get_rotation_matrix_from_xyz(orientation) 
        self.part_cloud.rotate(R, center=(0,0,0)) # use center=(0,0,0) here too
        self.part_cloud.translate(position)
        
        self.save_part(filename)

        # after adding all peices, features (and labels) increment the part counter
        self.part_num+=1
        
        return self.part_items  
 

    def create_5plate_partA(self, position=(0,0,0), orientation=(0,0,0), scale=1, filename='use_part_name'):
       
        self.part_type='5plateA' 
        self.part_name='%06d_%s'%(self.part_set_num, self.part_type)
        self.max_plates=5
        
        if filename=='use_part_name':
            filename='scene%s'%(self.part_name)
        
        fudge_x=self.fudge[0]
        fudge_y=self.fudge[1]
        fudge_z=self.fudge[2]
  
        plate1, plate1_lines, plane11, plane12 = self.add_plate(size=(self.l1,self.w1,self.t1), position=(0,0,0), orientation=(0,0,0))
 
        x2=self.l12  
        y2=np.asarray([ self.w12+self.t2, self.w1 ]).min() # limit the y offset to the edge of plate (prevents gap)
        z2=self.t1
        plate2, plate2_lines, plane21, plane22 = self.add_plate(size=(self.l2,self.w2,self.t2), 
                                                                position=(x2,y2,z2), 
                                                                orientation=(np.pi/2,0,0))
              
        
        # feature 1 lies in the xdirection on the intersection of the two plates
        lf1=np.asarray([self.l2, self.l1-self.l12]).min()-self.fudge[0]

        # dimensions of feature bounding box based on weld geometry
        #legy=np.asarray([self.t1, self.t2]).min()/2 # calculate leg from smallest thickness plate (ask SC about this)
        legy=self.w2*self.edge_fraction 
        rooty=legy/40  # calculate root from leg (this is not realistic for welding, must barely past surface to ensure catching points when sampling)  
        legz=legy      # use a square cross section   
        rootz=rooty   
        
        # trim the points in the interection of the two plates that are not part of the fillets
        trim_points=[[x2, y2-self.t2-legy, self.t1+legz],
                     [x2, y2+legy, self.t1+legz],
                     [x2, y2+legy, self.t1-self.fudge[2]],
                     [x2, y2-self.t2-legy, self.t1-self.fudge[2]],
                     [x2+lf1, y2-self.t2-legy, self.t1+legz],
                     [x2+lf1, y2+legy, self.t1+legz],
                     [x2+lf1, y2+legy, self.t1-self.fudge[2]],
                     [x2+lf1, y2-self.t2-legy, self.t1-self.fudge[2]]]

        self.trim_part(trim_points)
        
        # add fillet at feature location 
        fillet1=self.add_fillet(radius=legy/2, 
                                length=3.0, 
                                position=[x2+lf1/2, y2+legy, z2+legz], 
                                orientation=[0,90*np.pi/180,0], 
                                quadrant=4)

        # add second illet at feature location 
        fillet2=self.add_fillet(radius=legy/2, 
                                length=3.0, 
                                position=[x2+lf1/2, 
                                y2-self.t2-legy, z2+legz], 
                                orientation=[0,90*np.pi/180,0], 
                                quadrant=3)

        # delete random regions of points from the cloud
        self.delete_regions(self.part_cloud, num_regions=5, min_radius=0.5, max_radius=0.75)

        # sample and create a pointcloud with a bounding box for feature1
        feat1_bbox, feat1_cloud, feat1_indices = self.sample_part(size=(lf1, legy+rooty, legz+rootz), 
                                                                  position=(x2+self.fudge[0]/2, y2-self.t2-legy, self.t1-rooty), 
                                                                  orientation=(0,0,0) )
        self.add_feature(feat1_bbox, feat1_cloud, 'inside_fillet', orientation) # pass in final part orientation so the label has updated info

    # feature 2 disable, shadow in use
    #    # feature 2 is the same but on the opposite side
    #    # sample and create a pointcloud with a bounding box for feature 2
    #    feat2_bbox, feat2_cloud, feat2_indices = self.sample_part(size=(lf1, legy+rooty, legz+rootz), 
    #                                                              position=(x2+self.fudge[0]/2, y2, self.t1-rooty), 
    #                                                              orientation=(0,0,0) )
    #    self.add_feature(feat2_bbox, feat2_cloud, 'inside_fillet', orientation)

        # after labeling add noise to the pointcloud
        self.part_cloud = self.add_noise(self.part_cloud, sigma=0.05)   
        
        # after adding all peices and sampling, rotate the part and feature 
        for k,item in enumerate(self.part_items):
            item.translate(position)
            R = item.get_rotation_matrix_from_xyz(orientation) 
            item.rotate(R, center=(0,0,0))
        
        R = self.part_cloud.get_rotation_matrix_from_xyz(orientation) 
        self.part_cloud.rotate(R, center=(0,0,0))
          
        self.save_part(filename)     
  
        # after adding all peices, features (and labels) increment the part counter
        self.part_num+=1
      
        return self.part_items


    # function to load a part from a recorded pointcloud from sensor
    def load_sensor_trainingpart(self, position=(0,0,0), orientation=(0,0,0), filename='sensor_part'):

        #part=TrainingPart()
 
        print('creating trainingpart from recorded sensor data')
        print(filename) 
        # first load pcd files, start with prepared scans (no cropping, scaling needed)
        fdir=os.path.join(os.environ[params.data_dir],params.sensor_data_dir)
     
        fpath=os.path.join(fdir,filename)

        if fpath.endswith('.ply'):
              mesh = o3d.io.read_triangle_mesh(fpath)
              print(f"Pointcloud loaded pointcloud from: {fpath}")
              # resample mesh to generate pointcloud        
              cloud=o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, config.sample_points)

        elif fpath.endswith('.pcd'):
            cloud = o3d.io.read_point_cloud(fpath)
            print(f"Pointcloud loaded pointcloud from: {fpath}")

        else:
            print('unknown input file type')

        self.part_type='3plateA' 
        self.part_name='%06d_%s'%(self.part_set_num, self.part_type)

        filename='scene%s'%(self.part_name)
        self.part_type='3plateA' 

        self.part_cloud=combine_pointclouds(self.part_cloud, cloud) # copy the cloud from file to the part_cloud
        self.part_segments.append(cloud)

        for i,point in enumerate(cloud.points):
            self.seg_indices.append(0) # zero is for unannotated points (non feature points)

        
        num_features=0   
        adding_features=True
        #for i in range(num_features):
        while adding_features: 
            # add a feature contained by a single rectangular bounding box selected by the user 
            print('adding feature number: ', i)   
            class_index=input('Enter the class of the feature as an integer, leave blank to complete.')
            if len(class_index)>0:
                print('Feature will be labeled as: ', params.feature_classes[int(class_index)])
                findices=self.pick_feature_points(cloud)
                # use axis aligned boxes for now
                fpoints=[]
                print(findices)
                for idx in findices:
                    fpoints.append(cloud.points[idx])
                print(np.asarray(fpoints))
                # put the max and min xyz values from the clicked feature points into array 
                fmin=[np.asarray(fpoints)[:,0].min(), np.asarray(fpoints)[:,1].min(), np.asarray(fpoints)[:,2].min()]
                fmax=[np.asarray(fpoints)[:,0].max(), np.asarray(fpoints)[:,1].max(), np.asarray(fpoints)[:,2].max()]
                fbox=o3d.geometry.AxisAlignedBoundingBox(min_bound=fmin, max_bound=fmax )
                fsize=fbox.get_extent()
                fcen=fbox.get_center()    
                print('min_bounds:', fmin)
                print('max_bounds:', fmax)

                f1_bbox, f1_cloud, f1_indices = self.sample_part( size=fsize,
                                                                  position=fmin,
                                                                  orientation=orientation )
                f1_orientation=(0,0,0) # votenet show_oriented_boxes() assumes f1 orientation is (0,0,0) for now
                self.add_feature(f1_bbox, f1_cloud, params.feature_classes[int(class_index)],
                                 position=position,
                                 orientation=orientation,
                                 feature_orientation=f1_orientation)
                num_features+=1
            else:
                print('adding features complete')
                adding_features=False                     
         
        self.save_part(filename)     
  
        # after adding all peices, features (and labels) increment the part counter
        self.part_num+=1
   
 
    # function to get points from graphics object using the mouse
    def pick_feature_points(self, pcd):

        print("")
        print("1) Pick points using [shift + left click]")
        print("   Press [shift + right click] to undo point picking")
        print("2) After picking points, press 'Q' to close the window")

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        print("")
        
        return vis.get_picked_points()        


    def show_grid(self):

        # show a 1in grid for reference
        grey=[.8,.8,.8] # colors are defined globally, move these
        black=[0,0,0]
        blue=[0,0,1] 
        # xlines parallel to the xaxis
        dx=1 
        dy=1
        xmax=10
        ymax=10
        num_xlines=2*xmax/dx
        num_ylines=2*ymax/dy
        len_xlines=2*xmax
        len_ylines=2*ymax
        
        pointsA=[[int(-len_xlines/2),i*dx,0] for i in range(int(-num_xlines/2),int(num_xlines/2))]
        pointsB=[[int(len_xlines/2),i*dx,0] for i in range(int(-num_xlines/2),int(num_xlines/2))]
        points=pointsA+pointsB
        
        lines = [[0+i, num_xlines+i] for i in range(len(pointsA))]
        colors = [grey for i in range(len(lines))]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors) 
        self.part_items.append(line_set)
        
        axis_points = [[int(-len_xlines/2),0,0],[int(len_xlines/2),0,0]]
        axis_set=o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(axis_points),
            lines=o3d.utility.Vector2iVector([[0,1]])
        )
        axis_set.colors = o3d.utility.Vector3dVector([black]) 
        self.part_items.append(axis_set)
        
        # ylines parallel to the yaxis
        pointsA=[[i*dy,int(-len_ylines/2),0] for i in range(int(-num_ylines/2),int(num_ylines/2))]
        pointsB=[[i*dy,int(len_ylines/2),0]for i in range(int(-num_ylines/2),int(num_ylines/2))]
        points=pointsA+pointsB

        lines = [[0+i, num_ylines+i] for i in range(len(pointsA))]
        colors = [grey for i in range(len(lines))]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )        
        line_set.colors = o3d.utility.Vector3dVector(colors)
        self.part_items.append(line_set)
        
        axis_points = [[0,int(-len_ylines/2),0],[0,int(len_ylines/2),0]]
        axis_set=o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(axis_points),
            lines=o3d.utility.Vector2iVector([[0,1]])
        )
        axis_set.colors = o3d.utility.Vector3dVector([black])
        self.part_items.append(axis_set)


    # function to show part colorized by segmentation indices 
    def show_segments(self):

        for i, seg in enumerate(self.seg_indices):
            #self.part_cloud.colors.append(self.hex_to_rgb(colormap[seg]))
            self.part_cloud.colors.append(hex_to_rgb(colormap[seg]))
        self.display_items.append(self.part_cloud)  
    
        # showing all the points as spheres is too much graphics load
        # i wish there was a different way to change the size of the points, maybe we should update
        # sspheres=[]
        # sphere=o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
        # for point in self.part_cloud.points:
        #     
        #   ssphere=copy.deepcopy(sphere).translate(point)
        #   ssphere.paint_uniform_color(self.hex_to_rgb(IBM_COLORS['blue40']))
        #   sspheres.append(ssphere)
                
        #o3d.visualization.draw_geometries( sspheres )
        #self.display_items+=sspheres
    

    def show_features(self):
        
        fspheres=[]
        sphere=o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
        for idx, cloud in enumerate(self.feature_clouds):
            for point in cloud.points:
                
              fsphere=copy.deepcopy(sphere).translate(point)
              #fsphere.paint_uniform_color(self.hex_to_rgb(IBM_COLORS['magenta40']))
              fsphere.paint_uniform_color(class_colors[self.instance_groups[idx]['label']])
              fspheres.append(fsphere)
                    
            self.display_items+=fspheres
            
            print('instance group:', self.instance_groups[idx])


    def show_part(self, features_as_spheres=False):
        # create a coordinate frame to display as the origin
        origin_base = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.origin=copy.deepcopy(origin_base).scale(1.0, center=(0,0,0))
 
        self.display_items=self.part_items
        self.display_items.append(self.origin) 
        self.show_segments() 
        self.show_grid()
        if features_as_spheres:
            self.show_features() # show the features as spheres (nice but slow)

        o3d.visualization.draw_geometries(self.display_items, mesh_show_back_face=True)

       
   # def hex_to_rgb(self,hexcode):
   #     
   #     h=hexcode.lstrip('#') # remove the # from the color code
   #     rgb=tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4)) # convert to rgb, (from SO)

   #     return rgb


    def trim_part(self, trim_points, rotation=np.identity(3), show_box=False):
        # remove the points from the intersection because they are internal
        rm_points=o3d.utility.Vector3dVector(trim_points)

       # rm_bbox=o3d.geometry.AxisAlignedBoundingBox().create_from_points(rm_points)
       # rm_bbox_o=o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(rm_bbox) 
        rm_bbox_o=o3d.geometry.OrientedBoundingBox.create_from_points(rm_points)        
       # rm_bbox_min=rm_bbox_o.get_minimal_oriented_bounding_box() 
        
        rm_bbox_o.rotate(rotation) 
        rm_indices=rm_bbox_o.get_point_indices_within_bounding_box(self.part_cloud.points)

        # wipe out the points but do not remove elements (dont pop!) to avoid index shift
        for index in rm_indices:
            np.asarray(self.part_cloud.points)[index]=[None]
            self.seg_indices[index]=None   # also remove indices from segs list

        # get the Non-None elements indices (might be faster with nested if actually)
        res = [i for i in range(len(self.seg_indices)) if self.seg_indices[i] != None]    
        
        # create new lists for points and seg indices without the nones
        tmp_points=o3d.utility.Vector3dVector()    
        tmp_indices=[]
        for index in res:   
            tmp_points.append(np.asarray(self.part_cloud.points)[index])
            tmp_indices.append(self.seg_indices[index])
       
        # copy the tmps to the originals
        self.part_cloud.clear()
        self.part_cloud.points=o3d.utility.Vector3dVector(tmp_points) 
        self.seg_indices.clear()  
        self.seg_indices=tmp_indices 
        
        if show_box:
            rm_bbox_o.color=hex_to_rgb(IBM_COLORS['red60']) 
            self.part_items.append(rm_bbox_o)
            #self.part_items.append(rm_bbox_min)

    def add_noise(self, cloud, mu=0, sigma=0.1, max_noise=1 ):

        N=len(cloud.points)
        noise=np.random.normal(mu,sigma,[N,3])*max_noise
        noisy_cloud=o3d.geometry.PointCloud()
        noisy_cloud.points=o3d.utility.Vector3dVector(cloud.points+noise)

        return noisy_cloud


    def delete_points(self, cloud, delete_percent=0.2):

        print('num points before delete:', len(cloud.points), len(self.seg_indices))

        shuffled_points=np.asarray(copy.deepcopy(cloud.points))
        np.random.shuffle(shuffled_points)

        N=int(len(shuffled_points)*(1-delete_percent))

        tmp_points=shuffled_points[:N]
        seg_indices=np.zeros(N, int).tolist()
        
        # copy the tmps to the originals
        self.part_cloud.clear()
        self.part_cloud.points=o3d.utility.Vector3dVector(tmp_points) 
        self.seg_indices.clear()  
        self.seg_indices=seg_indices 
        print('num points after delete:', N, len(self.part_cloud.points), len(self.seg_indices))


    # this function may be unused
    def delete_bins(self, cloud, bin_area=0.5, delete_percent=0.2):
        # note this did not work because it relied on the points being ordered, but that is not the case
        num_points=len(cloud.points)
        points_per_bin=self.point_density*bin_area
        #num_bins=int(num_points/points_per_bin)
        num_bins=math.ceil(num_points/points_per_bin)

        num_deleted=int(num_bins*(delete_percent))

        bin_mask=np.ones(num_bins-num_deleted, int).tolist()
        bin_mask.extend(np.zeros(num_deleted,int).tolist())
        random.shuffle(bin_mask)

        #print('bin_mask length:%d'%len(bin_mask))
        #print('bin_mask:', bin_mask)
        #print('point_density:%d, num_points:%d'%(self.point_density, num_points)) 
        #print('points_per_bin:%d, num_bins:%d'%(points_per_bin, num_bins))

        new_points=o3d.utility.Vector3dVector()    
        new_indices=[]
        #jdx=0

        for idx,point in enumerate(cloud.points):

            bin_num=int(idx/points_per_bin)
            print('bin_num: %i'%bin_num)

            if not bin_mask[bin_num]:
                print('point in bin_num: %d will not be added to new_points'%bin_num)
            else:
                print('point in bin_num: %d will be added to new_points'%bin_num)    
                new_points.append(cloud.points[idx])
                new_indices.append(self.seg_indices[idx])
               #jdx=jdx+1

        # copy the tmps to the originals
        self.part_cloud.clear()
        self.part_cloud.points=o3d.utility.Vector3dVector(new_points) 
        self.seg_indices.clear()  
        self.seg_indices=new_indices       

        print('new_points length: %d'%len(new_points))


    def delete_regions( self, cloud, num_regions, min_radius=0.5, max_radius=1 ):
            
        num_points=len(cloud.points)
        center_indices=np.random.randint(0,num_points,num_regions)
        region_radii=np.random.random(num_regions)*max_radius+min_radius

        new_points=o3d.utility.Vector3dVector()    
        new_indices=[]

        for idx,point in enumerate(cloud.points):
            delete_point=False # assume each point will be kept

            for jdx,index in enumerate(center_indices):
        
                center=cloud.points[index]
                radius=region_radii[jdx]

                #print('radius:', radius)
                #print('point_distance:',  self.point_distance(point, center) )

                if self.point_distance(point, center) < radius:
                    delete_point=True
                    #print('point is inside delete region and will be deleted ')

                #else:
                    #print('point is outside delete region')

            if not delete_point:

                new_points.append(cloud.points[idx])
                new_indices.append(self.seg_indices[idx])
                

        # copy the tmps to the originals
        self.part_cloud.clear()
        self.part_cloud.points=o3d.utility.Vector3dVector(new_points) 
        self.seg_indices.clear()  
        self.seg_indices=new_indices       

            
    def point_distance(self, point1, point2):

        p1=np.asarray(point1)
        p2=np.asarray(point2)
        distance=((p2[0]-p1[0])**2+(p2[1]-p1[1])**2+(p2[2]-p1[2])**2)**(1/2) 
        return distance        


# function to combine pointclouds into a single pointcloud through concantenation
def combine_pointclouds(cloud1, cloud2, verbose=False):

    cloud12 = o3d.geometry.PointCloud()
    cloud12_points=np.concatenate( (np.asarray(cloud1.points), np.asarray(cloud2.points)), axis=0 )
    cloud12.points=o3d.utility.Vector3dVector(cloud12_points)

    if verbose:
        print("combined cloud points:", type(cloud12.points), np.asarray(cloud12.points).shape )
    
    return cloud12


if __name__ == "__main__":
    # get user params from command line
    args = parser.parse_args()
   
    print('trainingpart.py - parts and functions for trainingset part generation')
   
    part=TrainingPart() 
    part.labels_dir='.'    


    for i,name in enumerate(part.param_indices.keys()):
        part.param_indices[name]=params.param_indices[name][0]        
    part.update_params(verbose=True)
    
    part.create_1plate_partA(position=(1,1,1), 
                             orientation=(0,0,0), 
                             filename=args.filename)
