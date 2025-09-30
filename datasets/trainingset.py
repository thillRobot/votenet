# trainingset.py - replaces generate_dataset.py, part definition functions in trainingpart.py 
# thillRobot - March 19, 2025
# this program generates parametric shapes with labels for CustomFeatures dataset
# note: seg fault issues may be solved intalling with the following process:
#    ``` conda install numpy
#        pip install open3d ``` 
# tested with Python 3.11 and Open3d 0.18.0 
       
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
import itertools

import params # custom params
from trainingpart import TrainingPart

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
parser.add_argument('--orientation', help="orientation for single part creation", nargs='+', type=float, required=False)
parser.add_argument('--sensor-data', help="use real sensor data, skip sythentic dataset creation ", action="store_true")

#custom color rgb codes, these are not really needed, use hex_to_rgb(IBM_COLORS['red40']) for example
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

# Check on potential issue with color cycling, (features should always get color but sometimes they get black)
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

colormap=[IBM_COLORS['black']] # make a color map of the IBM colors in the order shown above          
for color in IBM_COLORS.keys(): 
    colormap.append(IBM_COLORS[color])

class TrainingSet:
    def __init__(self, part_type='part'):
        
        # create a coordinate frame to display at the origin
        origin_base = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.origin=copy.deepcopy(origin_base).scale(1.0, center=(0,0,0))
        
        self.part_cloud=o3d.geometry.PointCloud() # complete part pointcloud 
        #self.feature_cloud=o3d.geometry.PointCloud() # feature cloud for display only, unused      
        self.feature_clouds=[] 
        # dictionary for parameter value indicies, does this belong in params.py ?
         
        self.param_indices={
                            'l0':0, 'l1':0, 'l2':0, 'l3':0, 
                            'w0':0, 'w1':0, 'w2':0, 'w3':0, 
                            't0':0, 't1':0, 't2':0, 't3':0, 
                            'l01':0, 'l12':0, 'l23':0, 'w01':0, 'w12':0, 
                            'point_density':0, 'fillet_radius':0, 'corner_radius':0
                           }                            
        
        self.part_type=part_type # this may be unused
        # define the names of the different types of parts (not feature classes), should be in params.py?
        self.part_types=['1plateA', '1plateB', '1plateC', '2plateA', '3plateA', '5plateA']       
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

        print('Adding subdirectories for the labels files')
        # make a directory for the labels files
        if not os.path.exists(self.labels_dir):
            os.makedirs(self.labels_dir)
        # and for the points files
        if not os.path.exists(self.points_dir):
            os.makedirs(self.points_dir)
        # and for the pcd files
        if not os.path.exists(self.pcds_dir):
            os.makedirs(self.pcds_dir)
       
        print('Adding subdirectories for each part type files')
        # and subdirectory for each part type
        for i,category in enumerate(self.part_types):
            directory=os.path.join(self.labels_dir, category) # in the labels dir
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            directory=os.path.join(self.points_dir, category) # and in the points dir (full parts go here)
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            directory=os.path.join(self.pcds_dir, category) # and in the pcds dir (full parts go here)
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        print('Adding subdirectories for each feature_class')
        # add subdirectory for each feature class        
        for i,category in enumerate(self.feature_classes):
            directory=os.path.join(self.labels_dir, category) # and in the points dir  (this might be unused for now)
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            directory=os.path.join(self.points_dir, category) # and in the points dir (clean segments only go here)
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            directory=os.path.join(self.pcds_dir, category) # and in the pcds dir (clean segments only go here)
            if not os.path.exists(directory):
                os.makedirs(directory)

        self.save_feature_files = False # options to save the feature points as separate pcds, uses extra space       
            
        #self.update_params()
        self.part_num=0  # tracks number parts in the dataset (a single object)
        self.part_set_num=0 # part number in the set of a single part type
        self.clear_part()
        

    def update_params(self, verbose=False):

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
        self.l1=params.l1[self.param_indices['l1']]
        self.l2=params.l2[self.param_indices['l2']]
        self.l3=params.l3[self.param_indices['l3']]

        self.w0=params.w0[self.param_indices['w0']]
        self.w1=params.w1[self.param_indices['w1']]
        self.w2=params.w2[self.param_indices['w2']]
        self.w3=params.w3[self.param_indices['w3']]

        self.t0=params.t0[self.param_indices['t0']]
        self.t1=params.t1[self.param_indices['t1']]
        self.t2=params.t2[self.param_indices['t2']]
        self.t3=params.t3[self.param_indices['t3']]

        self.l01=params.l01[self.param_indices['l01']]
        self.l12=params.l12[self.param_indices['l12']]
        self.l23=params.l23[self.param_indices['l23']]
        self.w01=params.w01[self.param_indices['w01']]
        self.w12=params.w12[self.param_indices['w12']] 
       
        self.point_density=params.point_density[self.param_indices['point_density']]
        self.fillet_radius=params.fillet_radius[self.param_indices['fillet_radius']]       
        self.corner_radius=params.fillet_radius[self.param_indices['corner_radius']]       
         
        self.fudge=params.fudge

        #self.point_density=params.density # number of points per ares (unit^2)
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
        
        #self.part_items=[self.origin] # separate meshes and pointcloud objects, starts with an origin only
        self.part_items=[] # separate meshes and pointcloud objects, starts with an origin only
        #self.display_items=[self.origin]
        self.part_cloud.clear() # clear the complete part pointcloud
        self.part_segments=[]
        self.feature_segments=[]
        self.seg_indices=[]
        self.instance_groups.clear()
        self.feature_clouds=[]

    
  #  def show_grid(self):

  #      # show a 1in grid for reference
  #      grey=[.8,.8,.8] # colors are defined globally, move these
  #      black=[0,0,0]
  #      blue=[0,0,1] 
  #      # xlines parallel to the xaxis
  #      dx=1 
  #      dy=1
  #      xmax=10
  #      ymax=10
  #      num_xlines=2*xmax/dx
  #      num_ylines=2*ymax/dy
  #      len_xlines=2*xmax
  #      len_ylines=2*ymax
  #      
  #      pointsA=[[int(-len_xlines/2),i*dx,0] for i in range(int(-num_xlines/2),int(num_xlines/2))]
  #      pointsB=[[int(len_xlines/2),i*dx,0] for i in range(int(-num_xlines/2),int(num_xlines/2))]
  #      points=pointsA+pointsB
  #      
  #      lines = [[0+i, num_xlines+i] for i in range(len(pointsA))]
  #      colors = [grey for i in range(len(lines))]
  #      line_set = o3d.geometry.LineSet(
  #          points=o3d.utility.Vector3dVector(points),
  #          lines=o3d.utility.Vector2iVector(lines),
  #      )
  #      line_set.colors = o3d.utility.Vector3dVector(colors) 
  #      self.part_items.append(line_set)
  #      
  #      axis_points = [[int(-len_xlines/2),0,0],[int(len_xlines/2),0,0]]
  #      axis_set=o3d.geometry.LineSet(
  #          points=o3d.utility.Vector3dVector(axis_points),
  #          lines=o3d.utility.Vector2iVector([[0,1]])
  #      )
  #      axis_set.colors = o3d.utility.Vector3dVector([black]) 
  #      self.part_items.append(axis_set)
  #      
  #      # ylines parallel to the yaxis
  #      pointsA=[[i*dy,int(-len_ylines/2),0] for i in range(int(-num_ylines/2),int(num_ylines/2))]
  #      pointsB=[[i*dy,int(len_ylines/2),0]for i in range(int(-num_ylines/2),int(num_ylines/2))]
  #      points=pointsA+pointsB

  #      lines = [[0+i, num_ylines+i] for i in range(len(pointsA))]
  #      colors = [grey for i in range(len(lines))]
  #      line_set = o3d.geometry.LineSet(
  #          points=o3d.utility.Vector3dVector(points),
  #          lines=o3d.utility.Vector2iVector(lines),
  #      )        
  #      line_set.colors = o3d.utility.Vector3dVector(colors)
  #      self.part_items.append(line_set)
  #      
  #      axis_points = [[0,int(-len_ylines/2),0],[0,int(len_ylines/2),0]]
  #      axis_set=o3d.geometry.LineSet(
  #          points=o3d.utility.Vector3dVector(axis_points),
  #          lines=o3d.utility.Vector2iVector([[0,1]])
  #      )
  #      axis_set.colors = o3d.utility.Vector3dVector([black])
  #      self.part_items.append(axis_set)


    # function to show part colorized by segmentation indices 
  #  def show_segments(self):

  #      for i, seg in enumerate(self.seg_indices):
  #          self.part_cloud.colors.append(self.hex_to_rgb(colormap[seg]))
  #      
  #      self.display_items.append(self.part_cloud)  
  #  
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
    
   # def show_features(self):
   #     
   #     fspheres=[]
   #     sphere=o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
   #     for cloud in self.feature_clouds:
   #         for point in cloud.points:
   #             
   #           fsphere=copy.deepcopy(sphere).translate(point)
   #           fsphere.paint_uniform_color(self.hex_to_rgb(IBM_COLORS['magenta40']))
   #           fspheres.append(fsphere)
   #                 
   #         self.display_items+=fspheres

   # def show_part(self):
   #     
   #     self.display_items=self.part_items
   #     self.display_items.append(self.origin) 
   #     self.show_segments() 
   #     self.show_grid()
   #     #self.show_features() # show the features as spheres (nice but slow)

   #     o3d.visualization.draw_geometries(self.display_items, mesh_show_back_face=True)

       
    def hex_to_rgb(self,hexcode):
        
        h=hexcode.lstrip('#') # remove the # from the color code
        rgb=tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4)) # convert to rgb, (from SO)

        return rgb


    def create_dataset(self, show_set=False, random_position='none', random_orientation='none', simple_corners=False):
        
        dx=5.0 # options for the entire set (display only)
        dy=5.0  
        dz=15.0
        dxx=10
        dyy=10
        dzz=10
        droll=50.0
        dyaw=50.0
        parts_trans=[] 
         
        param_names=list(params.param_indices.keys())
        param_indices=list(params.param_indices.values())
        # use itertools to find all combinations of parameters (replaces huge nest of loops)
        param_combos=list(itertools.product(*param_indices))
        part_points=[]   
     
        num_part_types=len(self.part_types)
        display_interval=10 # perecent complete intervals to print progress 
        percent_complete=0 
        
        max_parts=(len(param_combos))
        total_parts=max_parts*len(params.part_types)
        rng=np.random.default_rng()
        part_set_num=0
        print('Generating %i total parts'%total_parts)
       
        # generate parts for each combination of the parameter set 
        for indices in param_combos:
            
            for i,idx in enumerate(indices):
                self.param_indices[param_names[i]]=idx        
            #self.update_params(verbose=True)
            #print(self.param_indices)

            # generate random signs for offline data augmentation 
            xyzsigns=[]
            rpysigns=[]
            for k in range(3):
                if rng.random()>0.5:
                    xyzsigns.append(1)
                else:
                    xyzsigns.append(-1)
            
                if rng.random()>0.5:
                    rpysigns.append(1)
                else:
                    rpysigns.append(-1)
  
            # generate random translations
            if random_position=='xyz':
                posx,posy,posz=[rng.random()*params.max_x*xyzsigns[0], 
                                rng.random()*params.max_y*xyzsigns[1], 
                                rng.random()*params.max_z*xyzsigns[2]]       
            elif random_position=='xy':
                posx,posy,posz=[rng.random()*params.max_x*xyzsigns[0], 
                                rng.random()*params.max_y*xyzsigns[1], 
                                0]       
            elif random_position=='x':
                posx,posy,posz=[rng.random()*params.max_x*xyzsigns[0], 
                                0, 
                                0]       
            elif random_position=='none':
                posx,posy,posz=[0, 0, 0]                                                        
            
            # generate random rotations
            if random_orientation=='xyz':
                roll, pitch, yaw=[rng.random()*params.max_roll*rpysigns[0], 
                                  rng.random()*params.max_pitch*rpysigns[1], 
                                  rng.random()*params.max_yaw*rpysigns[2]] 
            elif random_orientation=='xy':
                roll, pitch, yaw=[rng.random()*params.max_roll*rpysigns[0], 
                                  rng.random()*params.max_pitch*rpysigns[1], 
                                  0] 
            elif random_orientation=='x':
                roll, pitch, yaw=[rng.random()*params.max_roll*rpysigns[0], 
                                   0, 
                                  0] 
            elif random_orientation=='y':
                roll, pitch, yaw=[0, 
                                  rng.random()*params.max_pitch*rpysigns[1], 
                                  0] 
            elif random_orientation=='z':
                roll, pitch, yaw=[0, 
                                  0, 
                                  rng.random()*params.max_yaw*rpysigns[2]]
            elif random_orientation=='none':
                roll, pitch, yaw=[0, 0, 0]                                                 
            
               
            for part_type in params.part_types:
                part=TrainingPart() 
                part.update_params(self.param_indices)

                part.part_set_num=part_set_num
                part_func=getattr(part, 'create_'+part_type)      
                part_func(position=(posx,posy,posz), 
                          orientation=(roll,pitch,yaw))
                
                self.all_parts.append('%s'%part.part_filename)
                part_points.append(len(part.part_cloud.points))       
                
                if args.show_each:
                    part.show_part(features_as_spheres=False)
                    part.clear_part()
       
            part_set_num+=1 
            
            if not (part_set_num % int(max_parts/display_interval)):
                percent_complete+=display_interval
                print('Generated %d of %d total parts'%(part_set_num*len(params.part_types), 
                                                          max_parts*len(params.part_types)))
                print(' Maximum number of points:', np.asarray(part_points).max())
                print(' Minimum number of points:', np.asarray(part_points).min())
                print(' Average number of points:', int(np.asarray(part_points).mean()))
          
        # add global origin for angle reference    
        parts_trans.append(self.origin)    
        # generate the splits (train, val, test) files
        self.prepare_splits(params.train_frac, params.val_frac)

        # show all mesh geometries in new window
        if show_set:
            o3d.visualization.draw_geometries(parts_trans, mesh_show_back_face=False)

    # function to create a dataset from real sensor data, replaces synthetic dataset
    def prepare_sensor_data(self, save_output=True, show_cloud=False):
        
        print('preparing files for dataset from recorded sensor data')
        # load pcd files, transform, scale, and crop 
        fdir=os.path.join(os.environ[params.data_dir],params.sensor_data_dir)
        sdata_dir=os.path.join(fdir,'prepared_data')
        
        if not os.path.exists(sdata_dir):
            mkdir(sdata_dir)

        # get this list of files to use
        from os import listdir
        from os.path import isfile, join
        filelist = [f for f in listdir(fdir) if isfile(join(fdir, f))]
        print(filelist)

        part_set_num=0
        for fname in filelist: 
            print(fname)
            fpath=os.path.join(fdir,fname)
 
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

            # use picked points to choose bounding box and ...
            findices=self.pick_feature_points(cloud)
            # use axis aligned boxes for now
            fpoints=[]
            print(findices)
            for idx in findices:
                fpoints.append(cloud.points[idx])
    
            #rotate the pointcloud
            #R=cloud.get_rotation_matrix_from_xyz((-90*np.pi/180,0,0))
            R=cloud.get_rotation_matrix_from_xyz((50*np.pi/180,5*np.pi/180,0))
            #R=cloud.get_rotation_matrix_from_xyz((0*np.pi/180,0,0))
            cloud_rot=copy.deepcopy(cloud)
            cloud_rot.rotate(R, center=(0,0,0))
            #translate the pointcloud
            cloud_trans=cloud_rot.translate((20, -15, 23))
            #cloud_trans=cloud_rot.translate((6, 30, -30))
            #cloud_trans=cloud_rot.translate((12, 75, 25))

            # create a filename for the copy
            #fname=str(config.input_file).split('/')[-1]
            #fname=str(config.input_file).split('/')[1]

               
            fname=fname.split('.')[0]+'_transformed.pcd'

            new_fpath=os.path.join(sdata_dir,fname)

            # save a copy of the file	to the output directory
            #o3d.io.write_triangle_mesh(new_fpath, cloud)
            if save_output:
                o3d.io.write_point_cloud(new_fpath, cloud_trans) 
                print(f"Mesh saved to file: {new_fpath}")

            if show_cloud:    
                # show the pointcloud in a new window
                origin_base = o3d.geometry.TriangleMesh.create_coordinate_frame()
                origin=copy.deepcopy(origin_base).scale(25, center=(0,0,0))
                show_pcd([origin, cloud_trans])

  #  # function to create a dataset from real sensor data, replaces synthetic dataset
    def load_sensor_trainingset(self, position=(0,0,0), orientation=(0,0,0)):
        
        print('creating dataset from recorded sensor data')
        # first load pcd files, start with prepared scans (no cropping, scaling needed)
        fdir=os.path.join(os.environ[params.data_dir],params.sensor_data_dir)

        # get this list of files to use
        from os import listdir
        from os.path import isfile, join
        filelist = [f for f in listdir(fdir) if isfile(join(fdir, f))]
        print(filelist)

        part_points=[]
        part_set_num=0
        for fname in filelist: 

            # load a new part from each file in the list
            part=TrainingPart()
            part.part_set_num=part_set_num
            part.load_sensor_trainingpart(filename=fname)
                
            part.show_part(features_as_spheres=True)

            self.all_parts.append('%s'%part.part_filename)
            part_points.append(len(part.part_cloud.points))       
            
            part_set_num+=1
        
        train_frac=0
        val_frac=1
        # generate the splits (train, val, test) files
        self.prepare_splits(train_frac, val_frac, test=True)
    

    # function to shuffle and split parts names for training, validation, and testing
    def prepare_splits(self, train_frac, val_frac, shuffle=True, test=False):
        print('preparing split files') 
        #print('len(self.all_parts)', len(self.all_parts) )
        shuffled_parts=self.all_parts.copy()
        if shuffle:
            random.shuffle(shuffled_parts)
  
        for i,part in enumerate(shuffled_parts):
            
            if i<train_frac*len(shuffled_parts):
              split='train'
            elif i<(train_frac+val_frac)*len(shuffled_parts):
              split='val'
            else:
              split='test'

            if test: # put all the files in the _test.txt file
                split='test'
                
            splits_file='CustomFeatures/custom_%s.txt'%split
            #print('writing splits file:', os.path.join(self.data_dir, splits_file))     
            with open(os.path.join(self.data_dir, splits_file), 'a', encoding="utf-8") as f:
                f.write(part+'\n')
                f.close

            if test:
                # you also need blank _train.txt and _val.txt for VoteNet
                split='train'
                splits_file='CustomFeatures/custom_%s.txt'%split
                with open(os.path.join(self.data_dir, splits_file), 'a', encoding="utf-8") as f:
                    f.close
                split='val'
                splits_file='CustomFeatures/custom_%s.txt'%split
                with open(os.path.join(self.data_dir, splits_file), 'a', encoding="utf-8") as f:
                    f.close

        #print('splits file saved as:', os.path.join(self.data_dir, splits_file))     

    # function to save a compressed archive of the dataset and the files required to generate dataset    
    def compress_dataset(self):
        print('compressing dataset as: {}'.format(self.archive_file))
        zip_dirs=['pcds','points','labels'] # directories to add to archive 
        split_files=['custom_train.txt', 'custom_val.txt', 'custom_test.txt'] # the split files
        pkg_files=['generate_dataset.py', 'params.py']   # scripts required to generate dataset                    
        dataset_dir=os.path.join(self.data_dir,'CustomFeatures')
        archive_path=os.path.join(dataset_dir,self.archive_file) 
        with zipfile.ZipFile(archive_path,'w', zipfile.ZIP_DEFLATED) as zipf:
            # nested os.walk is clunky, but it accomplishes selective folder compression, (FIX THIS THIS SOON)
            # walk through and save all pcds, points, and labels
            for root, dirs, files in os.walk(dataset_dir):
                if os.path.relpath(root,dataset_dir) in zip_dirs:
                    for rt, dr, fl in os.walk(root):
                        for file in fl:
                            relpath=os.path.relpath(os.path.dirname(os.path.join(rt, file)), dataset_dir)
                            zipf.write(os.path.join(rt,file),os.path.join(relpath,file))
            # also save the split files
            for file in split_files:
              zipf.write(os.path.join(dataset_dir,file),file)
            
            # also save this script and the params.py file for records,  this uses a single walk, can we do this above?
            for root, dirs, files in os.walk(os.environ['PWD']):
                pkg_dir=os.path.relpath(root,os.environ['PWD']).split('/')[-1]
                if pkg_dir=='custom_features':
                    for file in files: 
                        if file in pkg_files:
                           zipf.write(os.path.join(root,file),os.path.join(pkg_dir,file))
        
        # show the file size to confirm completion
        archive_bytes=os.stat(archive_path).st_size
        print(f'archive file size: {archive_bytes/1e3:.2f} MB ({archive_bytes/1e9:.2f} GB)')
        # this function is overly complicated, but fortunately is only has to run once at the end :)


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
 
    # instantiate a training dataset object 
    trainingset=TrainingSet() 
      
    if args.single_part: # create a single part for testing
        for part_type in params.part_types:
            print('Creating single part, skipping dataset generation')
            part=TrainingPart() 
            for i,name in enumerate(trainingset.param_indices.keys()): # use the first index in each list
                trainingset.param_indices[name]=params.param_indices[name][0]        
            part.update_params(trainingset.param_indices, verbose=True)
                
            part_func=getattr(part, 'create_'+part_type)     
     
            part_func(position=(1,1,1), 
                      orientation=(args.orientation[0]*np.pi/180,
                                   args.orientation[1]*np.pi/180,
                                   args.orientation[2]*np.pi/180),
                      filename=args.filename)

            part.show_part()
            part.clear_part()

    elif(args.sensor_data): # create a dataset from recorded sensor data
      
 
        #trainingset.prepare_sensor_data()        
        trainingset.load_sensor_trainingset()
        
    else: # create a dataset of synthetic parts
        print('Beginning dataset generation')
        
        trainingset.create_dataset(show_set=args.show_set, 
                                   random_position=params.random_position, 
                                   random_orientation=params.random_orientation, 
                                   simple_corners=params.simple_corners) 
        
        if args.no_archive:     # this is counter intuitive to me (--quiet mode behavior)
            trainingset.compress_dataset()
        else:      
            print('Skipping compress_dataset()')
