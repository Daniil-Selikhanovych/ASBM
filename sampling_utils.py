import torch


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out

def q_sample(coeff, x_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
      noise = torch.randn_like(x_start)
      
    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise
    
    return x_t


def sample_posterior(coefficients, x_0, x_t, t):
    
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped
    
  
    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        
        noise = torch.randn_like(x_t)
        
        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:,None,None,None] * torch.exp(0.5 * log_var) * noise
            
    sample_x_pos = p_sample(x_0, x_t, t)
    
    return sample_x_pos

def q_sample_pairs(coeff, x_start, t):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = extract(coeff.a_s, t+1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t+1, x_start.shape) * noise
    
    return x_t, x_t_plus_one

def q_sample_supervised(pos_coeff, x_start, t, x_end, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
      noise = torch.randn_like(x_start)

    # T = len(coeff.a_s_cum)
    T = len(pos_coeff.a_s_cum)

    x_t = x_end
    for t_current in reversed(list(range(t[0], T))):
        t_tensor = torch.full((x_t.size(0),), t_current, dtype=torch.int64).to(x_t.device)
        x_t = sample_posterior(pos_coeff, x_start, x_t, t_tensor)
    
    return x_t


def q_sample_supervised_pairs(pos_coeff, x_start, t, x_end):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
#     noise = torch.randn_like(x_start)
    T = pos_coeff.posterior_mean_coef1.shape[0]

    x_t_plus_one = x_end
    t_current = T

    while t_current != t[0]:
        t_tensor = torch.full((x_end.size(0),), t_current-1, dtype=torch.int64).to(x_end.device)
        x_t_plus_one = sample_posterior(pos_coeff, x_start, x_t_plus_one, t_tensor)
        t_current -= 1

    t_tensor = torch.full((x_end.size(0),), t_current, dtype=torch.int64).to(x_end.device)
    x_t = sample_posterior(pos_coeff, x_start, x_t_plus_one, t_tensor)
    
    return x_t, x_t_plus_one


def q_sample_supervised_pairs_brownian(pos_coeff, x_start, t, x_end):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    num_steps = pos_coeff.posterior_mean_coef1.shape[0]
    t_plus_one_tensor = ((t+1)/num_steps)[:, None, None, None]

    x_t_plus_one = t_plus_one_tensor*x_end + (1.0 - t_plus_one_tensor)*x_start + torch.sqrt(pos_coeff.epsilon*t_plus_one_tensor*(1-t_plus_one_tensor))*noise
    
    x_t = sample_posterior(pos_coeff, x_start, x_t_plus_one, t)
    
    return x_t, x_t_plus_one


def q_sample_supervised_trajectory(pos_coeff, x_start, x_end):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
#     noise = torch.randn_like(x_start)
    trajectory = [x_end]
    T = pos_coeff.posterior_mean_coef1.shape[0]

    x_t_plus_one = x_end
    t_current = T

    while t_current != 0:
        t_tensor = torch.full((x_end.size(0),), t_current-1, dtype=torch.int64).to(x_end.device)
        x_t_plus_one = sample_posterior(pos_coeff, x_start, x_t_plus_one, t_tensor)
        t_current -= 1
        trajectory.append(x_t_plus_one)

    t_tensor = torch.full((x_end.size(0),), t_current, dtype=torch.int64).to(x_end.device)
    x_t = sample_posterior(pos_coeff, x_start, x_t_plus_one, t_tensor)
    trajectory.append(x_t)
    
    return trajectory


def sample_from_model(coefficients, generator, n_time, x_init, opt, return_trajectory=False):
    x = x_init
    trajectory = [x]
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
          
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(x, t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()
            trajectory.append(x)

    if return_trajectory:
        return x, trajectory
    
    return x


def sample_from_generator(coefficients, generator, n_time, x_init, opt, append_first=False):
    x = x_init
    if append_first:
        trajectory = [x]
    else:
        trajectory = []
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
          
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(x, t_time, latent_z)
            trajectory.append(x_0.detach())
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()

    trajectory = torch.stack(trajectory)

    return trajectory