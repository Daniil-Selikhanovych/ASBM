exp_name="swiss_roll_backward_exp_eps_0.03"
epsilon=0.03
dataset_name='swiss_roll_to_gaussian'
CUDA_VISIBLE_DEVICES=0 python train_asbm_swissroll.py --exp_name ${exp_name} --epsilon ${epsilon} \
--dataset_name ${dataset_name}