cd ../..
exp_name="swiss_roll_imf_eps_0.3"
epsilon=0.1
exp_forward="swiss_roll_forward_exp_eps_0.3"
exp_backward="swiss_roll_backward_exp_eps_0.3"
exp_forward_model="content_200000.pth"
exp_backward_model="content_200000.pth"
dataset_forward="swiss_roll"
dataset_backward="swiss_roll_to_gaussian"
save_ckpt_every=2
save_content_every=10000
inner_imf_mark_proj_iters=20001
imf_iters=20
CUDA_VISIBLE_DEVICES=0 python train_asbm_imf_swissroll.py --exp ${exp_name} --save_content --epsilon ${epsilon} \
--save_ckpt_every ${save_ckpt_every} --save_content_every ${save_content_every} \
--exp_forward ${exp_forward} --exp_backward ${exp_backward} --dataset_forward ${dataset_forward} \
--dataset_backward ${dataset_backward} --inner_imf_mark_proj_iters ${inner_imf_mark_proj_iters} \
--imf_iters ${imf_iters} --exp_forward_model ${exp_forward_model} --exp_backward_model ${exp_backward_model}