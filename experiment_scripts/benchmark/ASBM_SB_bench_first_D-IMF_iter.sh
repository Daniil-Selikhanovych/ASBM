cd ../..
current_dim=
current_eps=

python train_SB_bench_ABM.py --D_opt_steps 3 --plan 'ind' --dim ${current_dim} --epsilon ${current_eps} --fb 'f' --num_timesteps 32 --num_iterations 100000 --eval_freq 2500
python train_SB_bench_ABM.py --D_opt_steps 3 --plan 'ind' --dim ${current_dim} --epsilon ${current_eps} --fb 'b' --num_timesteps 32 --num_iterations 100000 --eval_freq 2500
