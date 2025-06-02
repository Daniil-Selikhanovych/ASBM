cd ../..
exp_name="swiss_roll_imf_eps_0.1"
epsilon=0.1
dataset_name="swiss_roll_swiss_roll_to_gaussian"
root="dd_gan_imf"
content_name="content_fw_imf_num_iter_19_20000.pth"
CUDA_VISIBLE_DEVICES=0 python test_swiss_roll.py --exp_name ${exp_name} --epsilon ${epsilon} \
--dataset_name ${dataset_name} --root ${root} --content_name ${content_name}