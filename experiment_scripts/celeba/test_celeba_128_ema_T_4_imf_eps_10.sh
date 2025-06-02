cd ../..
T=4
eps=1
dataset="celeba_celeba_female_to_male"
exp_name="ddgan_unpaired_celeba_ema_bigger_nets_T_4_imf_eps_10_celeba_128_ema_sampling"
posterior="brownian_bridge"
data_root="/trinity/home/daniil.selikhanovych/data/img_align_celeba"
image_size=128
num_channels_dae=64
ema_decay=0.999
CUDA_VISIBLE_DEVICES=0 python test_asbm_imf.py --dataset ${dataset} --num_timesteps ${T} --exp ${exp_name} \
--num_channels 3 --num_channels_dae ${num_channels_dae} --num_res_blocks 2 --batch_size 100 --nz 100 \
--z_emb_dim 256 --n_mlp 4 --embedding_type positional \
--ch_mult 1 1 2 2 4 4 --posterior ${posterior} --epsilon ${eps} --paired --data_root ${data_root} \
--image_size ${image_size}  --use_ema --ema_decay ${ema_decay}