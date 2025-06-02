cd ../..
exp_name="swiss_roll_forward_exp_eps_0.1"
epsilon=0.1
CUDA_VISIBLE_DEVICES=0 python train_asbm_swissroll.py --exp_name ${exp_name} --epsilon ${epsilon}