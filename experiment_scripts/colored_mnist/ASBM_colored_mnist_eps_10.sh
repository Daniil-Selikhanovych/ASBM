cd ../..
current_T=4
current_eps=10
current_innter_proj_iters=25000
current_proj_iters=50000
cur_mini_batch_OT=1
cur_ema_decay=0.999

papermill MNIST_ASBM_IMF.ipynb -p T $current_T -p eps $current_eps -p markovian_proj_iters $current_proj_iters -p inner_imf_mark_proj_iters $current_innter_proj_iters -p ema_decay $cur_ema_decay -p mini_batch_OT $cur_mini_batch_OT  MNIST_ASBM_IMF_log.ipynb 
