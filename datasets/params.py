import numpy as np

data_dir='DATA_DIR' # set data directory with env var DATA_DIR
labels_dir='CustomFeatures/labels'
points_dir='CustomFeatures/points'
pcds_dir='CustomFeatures/pcds'
sensor_data_dir='CustomFeatures/sensor_data/realsense_images'

archive_file='2plateA_3plateA_3456parts_0position_0orientation.zip'

part_types=['2plate_partA', '3plate_partA']

feature_classes=['inside_fillet', 
                 'outside_fillet', 
                 'inside_corner', 
                 'outside_corner',
                 'inside_outside_corner']

# parameters for part generation: 
# l1, l2, l3, w1, w2, w3, t1, t2, t3, (th12, th13, of12), roll, pitch, yaw


  #  # 512 parts (single type)
  #  param_indices={
  #                'l0':[0],
  #                'l1':[8,16],
  #                'l2':[8,16],
  #                'l3':[4],
  #                'w0':[0],
  #                'w1':[6,12], 
  #                'w2':[6], 
  #                'w3':[3],
  #                't0':[0],
  #                't1':[3,4], 
  #                't2':[3,4], 
  #                't3':[3,4],
  #                'l01':[0],
  #                'l12':[4],
  #                'l23':[4],
  #                'w01':[14],
  #                'w12':[8,12],
  #                'point_density':[2,3],
  #                'fillet_radius':[1,2],
  #                'corner_radius':[0]} # equals fillet radius 

# 512 parts (single type)
param_indices={
              'l0':[1],
              'l1':[8,12,16],
              'l2':[8,12,16],
              'l3':[4],
              'w0':[1],
              'w1':[6,12], 
              'w2':[6], 
              'w3':[3],
              't0':[0],
              't1':[3,4], 
              't2':[3,4], 
              't3':[3,4],
              'l01':[0],
              'l12':[4],
              'l23':[4],
              'w01':[14],
              'w12':[8,12],
              'point_density':[2,3],
              'fillet_radius':[1,2],
              'corner_radius':[0]} # equals fillet radius 
#  # 1536 parts
   # param_indices={
   #               'l0':[0],
   #               'l1':[8,10,12,14],
   #               'l2':[8,10],
   #               'l3':[4],
   #               'w0':[0],
   #               'w1':[6,8], 
   #               'w2':[4], 
   #               'w3':[2],
   #               't0':[0],
   #               't1':[10,14], 
   #               't2':[3,4], 
   #               't3':[3,4],
   #               'l01':[0],
   #               'l12':[4],
   #               'l23':[4],
   #               'w01':[14],
   #               'w12':[8,12],
   #               'point_density':[2,3,4],
   #               'fillet_radius':[1,2],
   #               'corner_radius':[0]} # equals fillet radius 

# 1536 parts
#    param_indices={
#                  'l0':[0],
#                  'l1':[8,10,12,14],
#                  'l2':[8,10],
#                  'l3':[4],
#                  'w0':[0],
#                  'w1':[6,8,10], 
#                  'w2':[4], 
#                  'w3':[2],
#                  't0':[0],
#                  't1':[10, 11, 12, 14], 
#                  't2':[3,4], 
#                  't3':[3,4],
#                  'l01':[0],
#                  'l12':[4],
#                  'l23':[4],
#                  'w01':[14],
#                  'w12':[8,12],
#                  'point_density':[2,3,4],
#                  'fillet_radius':[1,2],
#                  'corner_radius':[0]} # equals fillet radius 

# 1536, 3072 parts
#  param_indices={
#                'l0':[0],
#                'l1':[6,8],
#                'l2':[6,8],
#                'l3':[5],
#                'w0':[0],
#                'w1':[6,8], 
#                'w2':[4], 
#                'w3':[2],
#                't0':[0],
#                't1':[10,14], 
#                't2':[3,4], 
#                't3':[3,4],
#                'l01':[13],
#                'l12':[4],
#                'l23':[4],
#                'w01':[0],
#                'w12':[4,8],
#                'point_density':[1,2,3,4],
#                'fillet_radius':[0,1,2],
#               'corner_radius':[0]} # equals fillet radius 

# 1728 parts
#  param_indices={
#                 'l0':[0],
#                 'l1':[18,19,20],
#                 'l2':[18,19,20],
#                 'l3':[5],
#                 'w0':[0],
#                 'w1':[18,19], 
#                 'w2':[4,6], 
#                 'w3':[2],
#                 't0':[0],
#                 't1':[3], 
#                 't2':[3,4,5], 
#                 't3':[3,4],
#                 'l01':[13],
#                 'l12':[4,5],
#                 'l23':[4],
#                 'w01':[0],
#                 'w12':[13],
#                 'point_density':[1,2],
#                 'fillet_radius':[1], # [0,1,2]
#                 'corner_radius':[0]} # equals fillet radius 

# 4374 parts
#  param_indices={
#                 'l0':[0],
#                 'l1':[18,19,20],
#                 'l2':[18,19,20],
#                 'l3':[5],
#                 'w0':[0],
#                 'w1':[16,18,20], 
#                 'w2':[4,6,8], 
#                 'w3':[2],
#                 't0':[0],
#                 't1':[3], 
#                 't2':[3], 
#                 't3':[3],
#                 'l01':[13],
#                 'l12':[4,5,6],
#                 'l23':[4],
#                 'w01':[0],
#                 'w12':[8,10,14],
#                 'point_density':[1,2],
#                 'fillet_radius':[1,2,3], # [0,1,2]
#                 'corner_radius':[0]} # equals fillet radius 
 
# 7776 parts
 # param_indices={
 #               'l0':[0],
 #               'l1':[18,19,20],
 #               'l2':[18,19,20],
 #               'l3':[5],
 #               'w0':[0],
 #               'w1':[18,19,20], 
 #               'w2':[4,5,7], 
 #               'w3':[2],
 #               't0':[0],
 #               't1':[3,4,5,6], 
 #               't2':[3,4,5,6], 
 #               't3':[4],
 #               'l01':[13],
 #               'l12':[4,5,6],
 #               'l23':[4],
 #               'w01':[13],
 #               'w12':[13,14],
 #               }

# lists of parameter values
l0=[12.0, 18.0, 24.0, 30, 36.0, 48.0]
l1=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15] # length of plate 1
l2=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15] # length of plate 2
l3=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15] # length of plate 3
n_l=len(l1)       

w0=[12, 18.0, 24, 30, 36, 48]
w1=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15] # width of plate 1
w2=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15] # width of plate 2
w3=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15] # width of plate 3
n_w=len(w1)       
      
t0=[1.5]
t1=[0.125, 0.1875, 0.25, 0.3125, 0.375, 0.5, 0.625, 0.875, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] # thickness of plate 1
t2=[0.125, 0.1875, 0.25, 0.3125, 0.375, 0.5, 0.625, 0.875, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] # thickness of plate 2 
t3=[0.125, 0.1875, 0.25, 0.3125, 0.375, 0.5, 0.625, 0.875, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] # thickness of plate 3
n_t=len(t1)       
           
l01=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  #offset between plates, along the length axis of plate 0,1
l12=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  #offset between plates, along the width axis of plate 1 
l23=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  #offset between plates, along the length axios of plate 1,2
w01=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  #offset between plates, along the width axis of plate 0 
w12=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  #offset between plates, along the width axis of plate 1 
n_lw=len(w12)       

# these are not currently used, clean this up soon 
#th12=(90.0, 0.0, 0.0) # rotations between plate1 and plate2, about the xyz axes
#th13=(90.0, 0.0, 0.0) # angles between plate1 and plate3, about the xyz axes
#n_th=len(th12)       

# offline data augmentation (in the dataset)
random_position='none' # options: ['xyz', 'xy', 'x', 'none']
max_x=6.0
max_y=6.0
max_z=0.0 
random_orientation='none' # options: ['xyz', 'xy', 'z', 'none'] 
max_roll=0
max_pitch=0
max_yaw=0

fudge=(0.001, 0.001, 0.001)

point_density=[50, 100, 200, 400, 800] # number of points per area (unit^2)
# self.max_noise=0.8  # artificial noise in part clouds
max_noise=0.25
sigma_noise=0.025
mu_noise=0

num_holes=0 # 5         # artificial voids (circular holes) in part clouds
min_hole_radius=0.25
max_hole_radius=1.0

# fillet params
fillet_radius_fraction=0.75
fillet_radius=[0.125, 0.25, 0.5, 0.75, 1.0] 
fillet_edge_fraction=0.5 # flat length of feature is fillet_radius*fillet_edge_fraction
simple_corners=True

# corner params
corner_radius=[0.25, 0.5, 0.75, 1.0]
corner_edge_fraction=0.5
corner_fraction=0.10    
edge_fraction=0.50

# define the training, validation, and test splits 
train_frac=0.7
val_frac=0.2


# params for senor data dataset











