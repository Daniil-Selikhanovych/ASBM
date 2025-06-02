cd ../..
exp_name="swiss_roll_forward_exp"
epsilon=1.0
CUDA_VISIBLE_DEVICES=0 python test_swiss_roll.py --exp_name ${exp_name} --epsilon ${epsilon}