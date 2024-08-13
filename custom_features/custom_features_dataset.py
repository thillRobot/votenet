# coding: utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Dataset for object bounding box regression.
An axis aligned bounding box is parameterized by (cx,cy,cz) and (dx,dy,dz)
where (cx,cy,cz) is the center point of the box, dx is the x-axis length of the box.
"""
import os
import sys
import numpy as np
from torch.utils.data import Dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
from model_util_custom import CustomDatasetConfig
from model_util_custom import rotate_aligned_boxes
from model_util_custom import rotate_oriented_boxes
from model_util_custom import show_oriented_boxes


DC = CustomDatasetConfig()
MAX_NUM_OBJ = 64
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8]) # where do these magic numbers come from?

class CustomFeaturesDataset(Dataset):
       
    def __init__(self, split_set='train', num_points=20000,
        use_color=False, use_height=False, augment=False):

        self.data_path = os.path.join(BASE_DIR, 'CustomFeatures/data')
        all_scan_names = list(set([os.path.basename(x)[0:19] \
            for x in os.listdir(self.data_path) if x.startswith('scene')]))
        #print('all_scan_names:', all_scan_names)

        if split_set=='all':            
            self.scan_names = all_scan_names
        elif split_set in ['train', 'val', 'test']:
            split_filenames = os.path.join(ROOT_DIR, 'custom_features/CustomFeatures',
                'custom_{}.txt'.format(split_set))
            #print('split_filenames:', split_filenames)    

            with open(split_filenames, 'r') as f:
                self.scan_names = f.read().splitlines()  
            #print('scan_names:', self.scan_names)            

            # remove unavailiable scans
            num_scans = len(self.scan_names)
            self.scan_names = [sname for sname in self.scan_names \
                if sname in all_scan_names]
            print('kept {} scans out of {}'.format(len(self.scan_names), num_scans))
            num_scans = len(self.scan_names)
        else:
            print('illegal split name')
            return
        
        self.num_points = num_points
        self.use_color = use_color        
        self.use_height = use_height
        self.augment = augment
       
    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_clouds: (N,3+C)
            center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
            sem_cls_label: (MAX_NUM_OBJ,) semantic class index
            angle_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
            angle_residual_label: (MAX_NUM_OBJ,)
            size_classe_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
            size_residual_label: (MAX_NUM_OBJ,3)
            box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
            point_votes: (N,3) with votes XYZ
            point_votes_mask: (N,) with 0/1 with 1 indicating the point is in one of the object's OBB.
            scan_idx: int scan index in scan_names list
            pcl_color: unused
        """
        
        scan_name = self.scan_names[idx]        
        mesh_vertices = np.load(os.path.join(self.data_path, scan_name)+'_vert.npy')
        instance_labels = np.load(os.path.join(self.data_path, scan_name)+'_ins_label.npy')
        semantic_labels = np.load(os.path.join(self.data_path, scan_name)+'_sem_label.npy')
        instance_bboxes = np.load(os.path.join(self.data_path, scan_name)+'_bbox.npy')

        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3] # do not use color for now
            pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6] 
            point_cloud[:,3:] = (point_cloud[:,3:]-MEAN_COLOR_RGB)/256.0
        
        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) 
            
        # ------------------------------- LABELS ------------------------------        
        target_bboxes = np.zeros((MAX_NUM_OBJ, 10))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ)) 

        semantic_classes = np.zeros((MAX_NUM_OBJ,))

        xangle_classes = np.zeros((MAX_NUM_OBJ,))
        xangle_residuals = np.zeros((MAX_NUM_OBJ,))
        yangle_classes = np.zeros((MAX_NUM_OBJ,))
        yangle_residuals = np.zeros((MAX_NUM_OBJ,))
        zangle_classes = np.zeros((MAX_NUM_OBJ,))
        zangle_residuals = np.zeros((MAX_NUM_OBJ,))
        #angle_classes = np.zeros((MAX_NUM_OBJ,))
        #angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        
        point_cloud, choices = pc_util.random_sampling(point_cloud,
            self.num_points, return_choices=True)        
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        
        pcl_color = pcl_color[choices]
        
        target_bboxes_mask[0:instance_bboxes.shape[0]] = 1
        target_bboxes[0:instance_bboxes.shape[0],:] = instance_bboxes[:,0:10]
        
        
        # ------------------------------- DATA AUGMENTATION ------------------------------        
        augment_flip=False
        augment_scale=True
        augment_rotate=True
        augment_translate=True

        if self.augment and augment_flip:
            #mirror about the YZ plane    
            if np.random.random() > 0.5:
               point_cloud[:,0] = -1 * point_cloud[:,0]
               target_bboxes[:,0] = -1 * target_bboxes[:,0]                
               target_bboxes[:,6] += np.pi # add half rotation to x angle, check on this soon

            #mirror about the XZ plane
            if np.random.random() > 0.5:        
               point_cloud[:,1] = -1 * point_cloud[:,1]
               target_bboxes[:,1] = -1 * target_bboxes[:,1]                                
               target_bboxes[:,7] += np.pi # add half rotation to y angle
        
        if self.augment and augment_rotate:
            # Rotate about X-axis 
            # rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            # rot_angle = (np.random.random()*2*np.pi) # random angle from 0 to 360 deg
            # rot_mat = pc_util.rotx(rot_angle)
            # point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            # target_bboxes = rotate_aligned_boxes(target_bboxes, rot_mat)
            # Rotation about Y-axis 
            # rot_angle = (np.random.random()*2*np.pi) 
            # rot_mat = pc_util.roty(rot_angle)
            # point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            # target_bboxes = rotate_aligned_boxes(target_bboxes, rot_mat)
            
            #show_oriented_boxes(target_bboxes, point_cloud)

            #Rotate about Z-axis 
            if np.random.random()>0.5:
               dgamma = (np.random.random()*90*np.pi/180)
            else:    
               dgamma = -(np.random.random()*90*np.pi/180)
            Rz = pc_util.rotz(dgamma)
            
            #point_cloud[:,0:3], mat = pc_util.rotate_point_cloud(point_cloud[:,0:3],rot_mat) # this rotates about cloud center
            #point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(Rz))         # this rotates about the origin
               
            tmp = np.matmul(Rz, np.transpose(point_cloud[:,0:3]))
            point_cloud[:,0:3]=np.transpose(tmp)

            target_bboxes = rotate_oriented_boxes(target_bboxes, [0, 0, dgamma])  # this also rotates about the origin
            #target_bboxes = rotate_aligned_boxes(target_bboxes, rot_mat) # was used by scannet, no rotations

            #show_oriented_boxes(target_bboxes, point_cloud)

        if self.augment and augment_scale:        
            # note this scaling without resampling breaks the assumption of uniform point density
            scale_ratio = np.random.random()*1.0+0.5   # 0.5x to 1.5x scaling
            scale_ratio = np.expand_dims(np.tile(scale_ratio,3),0) # convert to be multiplied by list directly

            point_cloud[:,0:3]=point_cloud[:,0:3]*scale_ratio     # scale the xyz components of the feature pointcloud
            target_bboxes[:,0:3]=target_bboxes[:,0:3]*scale_ratio  # scale the center point of the bounding box
            target_bboxes[:,3:6]=target_bboxes[:,3:6]*scale_ratio  # scale the xyz sizes of the bounding box

        if self.augment and augment_translate:  
            #Translate on the XY plane
            table_size=36
            if np.random.random()>0.5:
               delx=np.random.random()*table_size/2
            else:    
               delx=-np.random.random()*table_size/2

            if np.random.random()>0.5:
               dely=np.random.random()*table_size/2
            else:    
               dely=-np.random.random()*table_size/2

            point_cloud[:,0:3]=point_cloud[:,0:3]+[delx, dely, 0] # move the points
            target_bboxes[:,0:3]=target_bboxes[:,0:3]+[delx, dely, 0] # move the box centers         

        # compute votes *AFTER* augmentation
        # Note: since there's no map between bbox instance labels and
        # pc instance_labels (it had been filtered 
        # in the data preparation step) we'll compute the instance bbox
        # from the points sharing the same instance label. 
        point_votes = np.zeros([self.num_points, 3])
        point_votes_mask = np.zeros(self.num_points)
        for i_instance in np.unique(instance_labels):            
            # find all points belong to that instance
            ind = np.where(instance_labels == i_instance)[0]
            # find the semantic label            
            if semantic_labels[ind[0]] in DC.classids:
                x = point_cloud[ind,:3]
                center = 0.5*(x.min(0) + x.max(0))
                point_votes[ind, :] = center - x
                point_votes_mask[ind] = 1.0
        point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical 
        
        class_ind = [np.where(DC.classids == x)[0][0] for x in instance_bboxes[:,-1]]   
        # NOTE: set size class as semantic class. Consider use size2class.
        size_classes[0:instance_bboxes.shape[0]] = class_ind
        size_residuals[0:instance_bboxes.shape[0], :] = \
            target_bboxes[0:instance_bboxes.shape[0], 3:6] - DC.mean_size_arr[class_ind,:]

        # compute size and heading classes after data augmentation (from sunrgb_detection_dataset.py)
        for i in range(target_bboxes.shape[0]):
            bbox = target_bboxes[i]
            #semantic_classes[i] = bbox[9]
            #box3d_center = bbox[0:3]
            xangle_class, xangle_residual = DC.angle2class(bbox[6]) # 
            yangle_class, yangle_residual = DC.angle2class(bbox[7]) #
            zangle_class, zangle_residual = DC.angle2class(-bbox[8]) # negative beacuse mention in 'tips' document ? 

            # NOTE: The mean size stored in size2class is of full length of box edges,
            # while in sunrgbd_data.py data dumping we dumped *half* length l,w,h.. so have to time it by 2 here 
            #box3d_size = bbox[3:6]*2
            #size_class, size_residual = DC.size2class(box3d_size, DC.class2type[semantic_class])
            #box3d_centers[i,:] = box3d_center
            xangle_classes[i] = xangle_class
            xangle_residuals[i] = xangle_residual
            yangle_classes[i] = yangle_class
            yangle_residuals[i] = yangle_residual
            zangle_classes[i] = zangle_class
            zangle_residuals[i] = zangle_residual
            #size_classes[i] = size_class
            #size_residuals[i] = size_residual
            #box3d_sizes[i,:] = box3d_size    
            
        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['center_label'] = target_bboxes.astype(np.float32)[:,0:3]
        ret_dict['xheading_class_label'] = xangle_classes.astype(np.int64)
        ret_dict['xheading_residual_label'] = xangle_residuals.astype(np.float32)
        ret_dict['yheading_class_label'] = yangle_classes.astype(np.int64)
        ret_dict['yheading_residual_label'] = yangle_residuals.astype(np.float32)
        ret_dict['zheading_class_label'] = zangle_classes.astype(np.int64)
        ret_dict['zheading_residual_label'] = zangle_residuals.astype(np.float32)
        #ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        #ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))                                
        target_bboxes_semcls[0:instance_bboxes.shape[0]] = \
            [DC.id2class[x] for x in instance_bboxes[:,-1][0:instance_bboxes.shape[0]]]                
        #target_bboxes_semcls=semantic_classes
        
        #print('semantic_classes:', semantic_classes)
        #print('target_bboxes_semcls:', target_bboxes_semcls)

        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)
        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['pcl_color'] = pcl_color
        return ret_dict
        
############# Visualizaion ########

def viz_votes(pc, point_votes, point_votes_mask, name=''):
    """ Visualize point votes and point votes mask labels
    pc: (N,3 or 6), point_votes: (N,9), point_votes_mask: (N,)
    """
    inds = (point_votes_mask==1)
    pc_obj = pc[inds,0:3]
    pc_obj_voted1 = pc_obj + point_votes[inds,0:3]    
    pc_util.write_ply(pc_obj, 'pc_obj{}.ply'.format(name))
    pc_util.write_ply(pc_obj_voted1, 'pc_obj_voted1{}.ply'.format(name))
    
def viz_obb(pc, label, mask, angle_classes, angle_residuals,
    size_classes, size_residuals, name=''):
    """ Visualize oriented bounding box ground truth
    pc: (N,3)
    label: (K,3)  K == MAX_NUM_OBJ
    mask: (K,)
    angle_classes: (K,)
    angle_residuals: (K,)
    size_classes: (K,)
    size_residuals: (K,3)
    """
    oriented_boxes = []
    K = label.shape[0]
    for i in range(K):
        if mask[i] == 0: continue
        obb = np.zeros(7)
        obb[0:3] = label[i,0:3]
        heading_angle = 0 # hard code to 0
        box_size = DC.mean_size_arr[size_classes[i], :] + size_residuals[i, :]
        obb[3:6] = box_size
        obb[6] = -1 * heading_angle
        print(obb)        
        oriented_boxes.append(obb)
    pc_util.write_oriented_bbox(oriented_boxes, 'gt_obbs{}.ply'.format(name))
    pc_util.write_ply(label[mask==1,:], 'gt_centroids{}.ply'.format(name))

    
if __name__=='__main__': 
    dset = CustomFeaturesDataset(use_height=True, num_points=40000)
    for i_example in range(4):
        example = dset.__getitem__(1)
        pc_util.write_ply(example['point_clouds'], 'pc_{}.ply'.format(i_example))    
        viz_votes(example['point_clouds'], example['vote_label'],
            example['vote_label_mask'],name=i_example)    
        viz_obb(pc=example['point_clouds'], label=example['center_label'],
            mask=example['box_label_mask'],
            angle_classes=None, angle_residuals=None,
            size_classes=example['size_class_label'], size_residuals=example['size_residual_label'],
            name=i_example)
