cd ../..
current_dim=
current_eps=

current_T=32
d_opt_steps=3

bw_ckpt='path_to_bw_first_D_IMF_iter_ckpt'
fw_ckpt='path_to_fw_first_D_IMF_iter_ckpt'

inner_imf_mark_proj_iters=50000

python train_ASBM_SB_bench.py --epsilon ${current_eps} --dim ${current_dim} --fw_ckpt ${fw_ckpt} --bw_ckpt ${bw_ckpt} --inner_imf_mark_proj_iters ${inner_imf_mark_proj_iters} --imf_iters 10 --eval_freq 5000 --num_timesteps ${current_T} --D_opt_steps ${d_opt_steps}
