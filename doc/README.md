

python batch_load_custom_data.py


CUDA_VISIBLE_DEVICES=0 python train.py --dataset custom --log_dir log_custom


CUDA_VISIBLE_DEVICES=0 python train.py --dataset custom --log_dir log_custom --batch_size 24 --max_epoch 100


CUDA_VISIBLE_DEVICES=0 python train.py --dataset custom --log_dir /home/votenet_ws/shared/log_custom --batch_size 24 --max_epoch 40


python demo.py --dataset custom --checkpoint_path custom_features_epoch120.tar --input_file scene2059_3plate.pcd


python eval.py --dataset custom --checkpoint_path demo_files/pretrained_votenet_on_custom_features.tar --dump_dir demo_files/custom_results/
