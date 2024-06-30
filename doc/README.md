

python batch_load_custom_data.py


CUDA_VISIBLE_DEVICES=0 python train.py --dataset custom --log_dir log_custom


CUDA_VISIBLE_DEVICES=0 python train.py --dataset custom --log_dir log_custom --batch_size 24 --max_epoch 100


CUDA_VISIBLE_DEVICES=0 python train.py --dataset custom --log_dir custom_features/CustomFeatures/log --batch_size 24 --max_epoch 500 --batch_interval 10



# run the demo with the default weights and input
python demo.py --dataset custom

# specify a checkpoint file to use different weights and a separate input image 
python demo.py --dataset custom --checkpoint_path custom_features/CustomFeatures/ckpt/checkpoint.tar --input_dir custom_features/CustomFeatures/demo_files --input_file scene2059_3plate.pcd

python demo.py --dataset custom --checkpoint_path custom_features/CustomFeatures/ckpt/epoch147.tar --input_dir custom_features/CustomFeatures/demo_files --input_file 2plate_part.pcd

python demo.py --dataset custom --checkpoint_path custom_features/CustomFeatures/log/checkpoint.tar --input_dir custom_features/CustomFeatures/demo_files --input_file 2plate_part.pcd

python demo.py --dataset custom --checkpoint_path custom_features/CustomFeatures/log/checkpoint.tar --input_dir custom_features/CustomFeatures/pcds --input_file 3plate/scene2173_3plate.pcd


python demo.py --dataset custom --checkpoint_path custom_features/CustomFeatures/log/checkpoint.tar --input_dir custom_features/CustomFeatures/pcds --input_file 3plate/scene0433_3plate.pcd

# run the eval routine 
python eval.py --dataset custom --checkpoint_path custom_features/CustomFeatures/log/checkpoint.tar --dump_dir custom_features/CustomFeatures/eval_results/

python eval.py --dataset custom --checkpoint_path demo_files/pretrained_votenet_on_custom_features.tar --dump_dir demo_files/custom_results/




meetin notes 6/19

 - is there a better camera ?

 - would transfer learning work with the camera transform uncertainty ?

 - feature based approach is used in the industry - said sam'

 - it must work on sim data first... then to real data

 - it does not need to handle everything to be commercially viable, pos comment from sam

 - round geometry might be better 

 - next step, build it out to make more robust, more features, more test parts 

 - update: 3-4 weeks, SC likes 3wks


 