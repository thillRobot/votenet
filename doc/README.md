

python batch_load_custom_data.py


CUDA_VISIBLE_DEVICES=0 python train.py --dataset custom --log_dir log_custom


CUDA_VISIBLE_DEVICES=0 python train.py --dataset custom --log_dir log_custom --batch_size 24 --max_epoch 100


CUDA_VISIBLE_DEVICES=0 python train.py --dataset custom --log_dir /home/votenet_ws/shared/log_custom --batch_size 24 --max_epoch 40



# run the demo with the default weights and input
python demo.py --dataset custom

# specify a checkpoint file to use different weights and a separate input image 
python demo.py --dataset custom --checkpoint_path custom_features/CustomFeatures/ckpt/checkpoint.tar --input_dir custom_features/CustomFeatures/demo_files --input_file scene2059_3plate.pcd

python demo.py --dataset custom --checkpoint_path custom_features/CustomFeatures/ckpt/checkpoint.tar --input_dir custom_features/CustomFeatures/demo_files --input_file 2plate_part.pcd

python eval.py --dataset custom --checkpoint_path demo_files/pretrained_votenet_on_custom_features.tar --dump_dir demo_files/custom_results/
