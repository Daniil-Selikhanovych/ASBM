import wandb
import numpy as np
import torch

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image

def fig2data (fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def fig2img ( fig ):
    buf = fig2data( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )
    
def pca_plot(x_0_gt, x_1_gt, x_1_pred, n_plot=512, save_name='plot_pca_samples.png', wandb_save_postfix='', is_wandb=True):

    x_0_gt, x_1_gt, x_1_pred = x_0_gt.cpu(), x_1_gt.cpu(), x_1_pred.cpu()
    fig,axes = plt.subplots(1, 3,figsize=(12,4),squeeze=True,sharex=True,sharey=True)
    pca = PCA(n_components=2).fit(x_1_gt)
    
    x_0_gt_pca = pca.transform(x_0_gt[:n_plot])
    x_1_gt_pca = pca.transform(x_1_gt[:n_plot])
    x_1_pred_pca = pca.transform(x_1_pred[:n_plot])
    
    axes[0].scatter(x_0_gt_pca[:,0], x_0_gt_pca[:,1], c="g", edgecolor = 'black',
                    label = r'$x\sim P_0(x)$', s =30)
    axes[1].scatter(x_1_gt_pca[:,0], x_1_gt_pca[:,1], c="orange", edgecolor = 'black',
                    label = r'$x\sim P_1(x)$', s =30)
    axes[2].scatter(x_1_pred_pca[:,0], x_1_pred_pca[:,1], c="yellow", edgecolor = 'black',
                    label = r'$x\sim T(x)$', s =30)
    
    for i in range(3):
        axes[i].grid()
        axes[i].set_xlim([-5, 5])
        axes[i].set_ylim([-5, 5])
        axes[i].legend()
    
    fig.tight_layout(pad=0.5)
    im = fig2img(fig)
    im.save(save_name)
    
    # можно заменить на другое логгирование
    if is_wandb:
        wandb.log({f'PCA samples {wandb_save_postfix}' : [wandb.Image(fig2img(fig))]})


from eot_benchmark.gaussian_mixture_benchmark import get_test_input_samples
from eot_benchmark.metrics import calculate_cond_bw
from tqdm import tqdm

def compute_condBWUVP(sample_fn, dim, eps, n_samples=1000, device='cpu'):
    test_samples = get_test_input_samples(dim=dim, device=device)
    
    model_input = test_samples.reshape(test_samples.shape[0], 1, -1).repeat(1, n_samples, 1)
    predictions = []

    with torch.no_grad():
        for test_samples_repeated in tqdm(model_input):
            predictions.append(sample_fn(test_samples_repeated).cpu())

    predictions = torch.stack(predictions, dim=0)

    # calculate cond_bw new
    
    if eps >= 1:
        eps=int(eps)

    print(test_samples.shape, predictions.shape)    
    
    cond_bw = calculate_cond_bw(test_samples, predictions, eps=eps, dim=dim)
    
    return cond_bw
