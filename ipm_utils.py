import numpy as np
from scipy.stats import norm

def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Growth
def growth_fn_glm(z1, z, m_par):
    mean_size_next_year = (m_par["grow.int"] + m_par["grow.z"] * z)
    growth_sd = m_par["grow.sd"]
    probability_density = norm.pdf(z1, loc=mean_size_next_year, scale=growth_sd)
    return probability_density

# Survival
def survival_fn_glm(z, m_par):
    linear_predictor = (m_par["surv.int"] + m_par["surv.z"] * z)
    survival_probability = sigmoid(linear_predictor)
    return survival_probability

# Flowering
def flowering_fn_glm(z, m_par):
    linear_predictor = (m_par["flow.int"] + m_par["flow.z"] * z)
    flowering_probability = sigmoid(linear_predictor)
    return flowering_probability

# Seed Production [Poisson mean function]
def seed_fn_glm(z, m_par):
    linear_predictor = m_par["seed.int"] + m_par["seed.z"] * z
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

def mk_K(n_mesh_points, m_par, lower_size, upper_size, vital_rate_functions):
    mesh_width = (upper_size - lower_size) / n_mesh_points # Width of each mesh interval
    mesh_points = (lower_size + (np.arange(n_mesh_points) + 0.5)* mesh_width) # Midpoints of each interval
    z1_grid, z_grid = np.meshgrid(mesh_points, mesh_points,indexing="ij")
    # P is a (n_mesh_points x n_mesh_points) matrix of transition probabilities
    # F is a (n_mesh_points x n_mesh_points) matrix of fecundity probabilities
    P = (mesh_width * survival_kernel_glm(z1_grid, z_grid, m_par, vital_rate_functions))
    F = (mesh_width * fecundity_kernel_glm(z1_grid, z_grid, m_par, vital_rate_functions))
    K = P + F

    return {
        "K": K,
        "P": P,
        "F": F,
        "mesh_points": mesh_points
    }