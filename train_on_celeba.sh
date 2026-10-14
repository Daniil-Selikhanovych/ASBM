dataset_mode="celeba"
dataroot="/trinity/home/daniil.selikhanovych/data/img_align_celeba_all/img_align_celeba"
crop_size=128
path_to_celeba_attr="/trinity/home/daniil.selikhanovych/NeuralOptimalTransport/datasets/list_attr_celeba.txt"
exp_name="celeba_male_to_female_128"
python train_on_celeba.py --dataset_mode ${dataset_mode} --dataroot ${dataroot} --name ${exp_name} --crop_size ${crop_size} --path_to_celeba_attr ${path_to_celeba_attr} \
--mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --gpu_ids 0 --save_by_iter