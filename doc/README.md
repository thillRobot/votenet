# Notes for VoteNet with CustomFeatures Dataset (FeatureNet)
# Tristan Hill, Summer 2024, Spring 2025

# generate models with labeled features
# currently this is outside this directory ../datasets/custom_features but it should be move in?
# it is in machine_vision instead because that is private, not ready for release yet

```
cd ../datasets/custom_features
python generate_dataset.py 
```
# this should generate the following directories of labeled data

  - labels/
  - pcds/
  - points/

# there are also three text files to define the split (train, validate, test)

  - custom_train.txt
  - custom_test.txt
  - custom_val.txt

# prepare dataset to be used for training 
  
```
cd votenet/custom_features
python batch_load_custom_data.py
```

# this will generate the following folder of data

  - data/

# use the clear script to save time loading new datsets
# this should delete all the files associated with a dataset for starting fresh
```
cd custom_features
python clear_custom_data.py
```
# run the training script to train a model from scratch or using a saved checkpoint
```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset custom --log_dir log_custom

CUDA_VISIBLE_DEVICES=0 python train.py --dataset custom --log_dir custom_features/CustomFeatures/log --batch_size 20 --eval_interval 10 --overwrite --learning_rate=.001

CUDA_VISIBLE_DEVICES=0 python train.py --dataset custom --log_dir custom_features/CustomFeatures/log --batch_size 24 --eval_interval 10 --overwrite --learning_rate=.001

CUDA_VISIBLE_DEVICES=0 python train.py --dataset custom --log_dir custom_features/CustomFeatures/log --batch_size 24 --max_epoch 200 --eval_interval 10 --num_point 300000 --ap_iou_thresh 0.25


# run the demo with the default weights and input
```
python demo.py --dataset custom
```

# specify a checkpoint file to use different weights and a separate input image 
# use the current training (log/checkpoint.tar) or a saved training (ckpt/foo.tar)

# test images from the training set or other synthetic images
``` 
python demo.py --dataset custom --checkpoint_path custom_features/CustomFeatures/log/checkpoint.tar --input_dir custom_features/CustomFeatures/pcds/2plateA --input_file scene000027_2plateA.pcd

python demo.py --dataset custom --checkpoint_path custom_features/CustomFeatures/ckpt/checkpoint_7776parts_norotation_epoch52.tar --input_dir custom_features/CustomFeatures/pcds/2plateA --input_file scene000027_2plateA.pcd

python demo.py --dataset custom --checkpoint_path custom_features/CustomFeatures/ckpt/checkpoint_7776parts_norotation_epoch52.tar --input_dir custom_features/CustomFeatures/demo_files --input_file 2plate_part.pcd
```

# test images from real sensor data (some tripod scans, some robot scans)
```
python demo.py --dataset custom --checkpoint_path custom_features/CustomFeatures/ckpt/checkpoint_7776parts_norotation_epoch52.tar --input_dir custom_features/CustomFeatures/demo_files/single_scans --input_file 3plate_partA_fig1_scaled40_transformed_cropped.pcd

python demo.py --dataset custom --checkpoint_path custom_features/CustomFeatures/log/checkpoint.tar --input_dir custom_features/CustomFeatures/demo_files/single_scans --input_file 3plate_partA_fig1_scaled40_transformed_cropped.pcd

python demo.py --dataset custom --checkpoint_path custom_features/CustomFeatures/log/checkpoint.tar --input_dir custom_features/CustomFeatures/demo_files/single_scans --input_file 3plate_partA_fig2_scaled40_transformed_cropped.pcd

python demo.py --dataset custom --checkpoint_path custom_features/CustomFeatures/ckpt/checkpoint_7776parts_norotation_epoch52.tar--input_dir custom_features/CustomFeatures/demo_files/single_scans --input_file application_fig1_scaled40_transformed_cropped.pcd 

python demo.py --dataset custom --checkpoint_path custom_features/CustomFeatures/log/checkpoint.tar --input_dir custom_features/CustomFeatures/demo_files/single_scans --input_file application_fig1_scaled40_transformed_cropped.pcd 

python demo.py --dataset custom --checkpoint_path custom_features/CustomFeatures/log/checkpoint.tar --input_dir custom_features/CustomFeatures/demo_files/single_scans --input_file application_fig2_scaled40_transformed_cropped.pcd --num_point 100000

python demo.py --dataset custom --checkpoint_path custom_features/CustomFeatures/log/checkpoint.tar --input_dir custom_features/CustomFeatures/demo_files --input_file part1_x5_y6_theta45_13_inliers_scaled15.pcd


# run the eval routine 
```
python eval.py --dataset custom --checkpoint_path custom_features/CustomFeatures/log/checkpoint.tar --dump_dir custom_features/CustomFeatures/eval_results/

python eval.py --dataset custom --checkpoint_path demo_files/pretrained_votenet_on_custom_features.tar --dump_dir demo_files/custom_results/
```



# meeting notes 6/19

 - is there a better camera ?

 - would transfer learning work with the camera transform uncertainty ?

 - feature based approach is used in the industry - said sam'

 - it must work on sim data first... then to real data

 - it does not need to handle everything to be commercially viable, pos comment from sam

 - round geometry might be better 

 - next step, build it out to make more robust, more features, more test parts 

 - update: 3-4 weeks, SC likes 3wks


# things to do
 
 - clean up generate datset code 

 - improve documentation, this README
 
 - get data ready for IDETC2025 
