# python test.py --dataroot ./datasets/horse2zebra --name h2z_SB --checkpoints_dir ./checkpoints --mode SB --eval --phase test --num_test 120 --gpu_ids 0
dataset_mode="celeba"
dataroot="/trinity/home/daniil.selikhanovych/data/img_align_celeba_all/img_align_celeba"
crop_size=128
path_to_celeba_attr="/trinity/home/daniil.selikhanovych/NeuralOptimalTransport/datasets/list_attr_celeba.txt"
exp_name="celeba_male_to_female_128"
python test_celeba.py --dataset_mode ${dataset_mode} --dataroot ${dataroot} --name ${exp_name} --crop_size ${crop_size} --checkpoints_dir ./checkpoints --path_to_celeba_attr ${path_to_celeba_attr} \
--mode sb --eval --phase test