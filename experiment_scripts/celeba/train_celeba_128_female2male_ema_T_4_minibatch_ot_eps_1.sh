cd ../..
T=4
eps=1
dataset="celeba_female_to_male"
exp_name="asbm_unpaired_celeba_female2male_ema_bigger_nets_T_4_minibatch_ot_eps_1_celeba_128"
master_port="6087"
posterior="brownian_bridge"
data_root="img_align_celeba"
image_size=128
num_channels_dae=64
ema_decay=0.999
batch_size=32
save_ckpt_every=2
save_content_every=4
num_epochs=150
CUDA_VISIBLE_DEVICES=0 python train_asbm.py --dataset ${dataset} --num_timesteps ${T} --exp ${exp_name} \
--num_channels 3 --num_channels_dae ${num_channels_dae} --num_res_blocks 2 --batch_size ${batch_size} --num_epoch ${num_epochs} \
--ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 \
--lazy_reg 15 --num_process_per_node 1 --ch_mult 1 1 2 2 4 4 --save_content --posterior ${posterior} --epsilon ${eps} \
--master_port ${master_port} --data_root ${data_root} --image_size ${image_size} --use_ema \
--ema_decay ${ema_decay} --save_ckpt_every ${save_ckpt_every} --save_content_every ${save_content_every} \
--use_minibatch_ot --num_process_per_node 1