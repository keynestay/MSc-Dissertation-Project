import numpy as np
from scipy.stats import norm
import tensorflow as tf

#GLM HELPER FNS
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Growth
def growth_fn_glm(z1, z, m_par):
    mean_size_next_year = (m_par["growth_int"] + m_par["growth_slope"] * z)
    growth_sd = m_par["growth_noise"]
    probability_density = norm.pdf(z1, loc=mean_size_next_year, scale=growth_sd)
    return probability_density

# Survival
def survival_fn_glm(z, m_par):
    linear_predictor = (m_par["surv_int"] + m_par["surv_slope"] * z)
    survival_probability = sigmoid(linear_predictor)
    return survival_probability

# Flowering
def flowering_fn_glm(z, m_par):
    linear_predictor = (m_par["repr_int"] + m_par["repr_slope"] * z)
    flowering_probability = sigmoid(linear_predictor)
    return flowering_probability

# Seed Production [Poisson mean function]
def seed_fn_glm(z, m_par):
    linear_predictor = m_par["seed_int"] + m_par["seed_slope"] * z
    expected_seed_number = np.exp(linear_predictor)
    return expected_seed_number

# Probability density of recruit size
def new_recruits_fn_glm(z1, m_par):
    recruit_mean_size = m_par["rcsz.int"]
    recruit_standard_deviation = m_par["rcsz.sd"]
    recruit_density = norm.pdf(z1, loc=recruit_mean_size, scale=recruit_standard_deviation)
    return recruit_density

def survival_kernel_glm(z1, z, m_par, funcs):
    flow = funcs['repr']
    surv = funcs['surv']
    grow = funcs['grow']
    return (
        (1 - flow(z, m_par))
        * surv(z, m_par)
        * grow(z1, z, m_par)
    )

def fecundity_kernel_glm(z1, z, m_par, funcs):
    flow = funcs['repr']
    seed = funcs['seed']
    new_recruits = funcs['new_recruits_fn']
    return (
        flow(z, m_par)
        * seed(z, m_par)
        * m_par["p.r"]
        * new_recruits(z1, m_par)
    )

def mk_K_glm(n_mesh_points, m_par, lower_size, upper_size, vital_rate_functions, correction):
    mesh_width = (upper_size - lower_size) / n_mesh_points # Width of each mesh interval
    mesh_points = (lower_size + (np.arange(n_mesh_points) + 0.5)* mesh_width) # Midpoints of each interval
    z1_grid, z_grid = np.meshgrid(mesh_points, mesh_points,indexing="ij")
    # P is a (n_mesh_points x n_mesh_points) matrix of transition probabilities
    # F is a (n_mesh_points x n_mesh_points) matrix of fecundity probabilities
    P = (mesh_width * survival_kernel_glm(z1_grid, z_grid, m_par, vital_rate_functions))
    F = (mesh_width * fecundity_kernel_glm(z1_grid, z_grid, m_par, vital_rate_functions))

    # Constant correction according to Metcalf et al.(2013)
    if correction:
        target = (1 - vital_rate_functions['repr'](mesh_points, m_par)) * vital_rate_functions['surv'](mesh_points, m_par) 
        col_sums = P.sum(axis=0)
        print(col_sums)
        print(target)
        correction_factor = np.divide(target, col_sums, out=np.ones_like(col_sums), where=col_sums>0)
        P *= correction_factor
    K = P + F
    return {
        "K": K,
        "P": P,
        "F": F,
        "mesh_points": mesh_points
    }

# GP HELPER FNS

def predict_y_mean(model, z):
    z = np.asarray(z, dtype=np.float64)
    if z.size == 0:
        return np.empty(0)

    X = tf.convert_to_tensor(z.reshape(-1,1), dtype=gp.default_float())
    mean, _ = model.predict_y(X)

    return mean.numpy().ravel()
def predict_f_mean(model, z):
    z = np.asarray(z, dtype=np.float64)

    if z.size == 0:
        return np.empty(0)

    X = tf.convert_to_tensor(z.reshape(-1, 1), dtype=gp.default_float())
    mean, _ = model.predict_f(X)

    return mean.numpy().ravel()
def sample_f(model, z):
    z = np.asarray(z, dtype=np.float64)

    if z.size == 0:
        return np.empty(0)

    X = tf.convert_to_tensor(z.reshape(-1, 1), dtype=gp.default_float())
    sample = model.predict_f_samples(Xnew=X, num_samples=1)

    return sample.numpy().ravel()

def survival_kernel_gp(z_next, z_current, params, funcs):
    p_repr = np.clip(predict_y_mean(funcs['repr'],z_current), 0.0, 1.0)
    p_surv = np.clip(predict_y_mean(funcs['surv'], z_current), 0.0, 1.0)
    grow_sd = np.sqrt(float(np.asarray(params['growth_noise_variance'])))
    return (
        (1.0 - p_repr) * p_surv * norm.pdf(z_next, loc=predict_f_mean(funcs['growth'], z_current), scale=grow_sd,)
    )
def fecundity_kernel_gp(z_next, z_current, params, funcs):
    p_repr = np.clip(predict_y_mean(funcs['repr'], z_current), 0.0, 1.0)
    expected_seeds = np.clip(predict_y_mean(funcs['seed'], z_current), 0.0, None)
    recruit_size_density = norm.pdf(z_next, loc=params['recruit_mean'], scale=params['recruit_sd'],)

    return (p_repr * expected_seeds * params['prob_recruit_est'] * recruit_size_density)

def make_gp_ipm(n_mesh_points, lower_size, upper_size, params, funcs, correction):
    mesh_width = (upper_size - lower_size) / n_mesh_points
    mesh_points = (lower_size + (np.arange(n_mesh_points) + 0.5) * mesh_width)
    z_next_grid = mesh_points[:, None]
    z_current_grid = mesh_points[None, :]
    
    P = mesh_width * survival_kernel_gp(z_next_grid, z_current_grid, params, funcs)

    # Constant correction according to Metcalf et al.(2013)
    if correction:
        p_repr = np.clip(predict_y_mean(funcs["repr"], mesh_points), 0.0, 1.0)
        p_surv = np.clip(predict_y_mean(funcs["surv"], mesh_points), 0.0, 1.0)
        target = (1.0 - p_repr) * p_surv
        col_sums = P.sum(axis=0)

        P *= np.divide(target, col_sums, out=np.ones_like(col_sums), where=col_sums > 0,)   

    F = mesh_width * fecundity_kernel_gp(z_next_grid, z_current_grid, params, funcs)
    K = P + F

    return {
        "K": K,
        "P": P,
        "F": F,
        "mesh_points": mesh_points,
        "mesh_width": mesh_width,
        "lower_size": lower_size,
        "upper_size": upper_size,
    }

