import numpy as np
import pandas as pd
from scipy.linalg import eig
from scipy.special import expit
from scipy.stats import norm


def sigmoid(x):
    return expit(x)


def survival_fn_glm(z, m_par):
    return sigmoid(m_par["surv_int"] + m_par["surv_slope"] * z)


def reproduction_fn_glm(z, m_par):
    return sigmoid(m_par["repr_int"] + m_par["repr_slope"] * z)


def recruitment_fn_glm(m_par):
    return float(sigmoid(m_par["recr_int"]))


def growth_fn_glm(z1, z, m_par):
    mean = m_par["growth_int"] + m_par["growth_slope"] * z
    return norm.pdf(z1, loc=mean, scale=m_par["growth_noise"])


def recruit_size_fn_glm(z1, z, m_par):
    mean = m_par["rcsz_int"] + m_par["rcsz_slope"] * z
    return norm.pdf(z1, loc=mean, scale=m_par["rcsz_noise"])


def survival_kernel_glm(z1, z, m_par, funcs=None):
    funcs = funcs or {}
    surv = funcs.get("surv", survival_fn_glm)
    grow = funcs.get("growth", growth_fn_glm)
    return surv(z, m_par) * grow(z1, z, m_par)


def fecundity_kernel_glm(z1, z, m_par, funcs=None):
    funcs = funcs or {}
    surv = funcs.get("surv", survival_fn_glm)
    repr_fn = funcs.get("repr", reproduction_fn_glm)
    recruit_size = funcs.get("rcsz", recruit_size_fn_glm)
    female_prob = m_par.get("female_prob", 0.5)
    return surv(z, m_par) * repr_fn(z, m_par) * female_prob * recruitment_fn_glm(m_par) * recruit_size(z1, z, m_par)


def _correct_columns(matrix, target):
    column_sums = matrix.sum(axis=0)
    matrix *= np.divide(target, column_sums, out=np.ones_like(column_sums), where=column_sums > 0)


def mk_K_glm(n_mesh_points, m_par, lower_size, upper_size, vital_rate_functions=None, correction=True):
    h = (upper_size - lower_size) / n_mesh_points
    mesh = lower_size + (np.arange(n_mesh_points) + 0.5) * h
    z1, z = np.meshgrid(mesh, mesh, indexing="ij")
    P = h * survival_kernel_glm(z1, z, m_par, vital_rate_functions)
    F = h * fecundity_kernel_glm(z1, z, m_par, vital_rate_functions)
    if correction:
        survival_target = survival_fn_glm(mesh, m_par)
        fecundity_target = survival_target * reproduction_fn_glm(mesh, m_par) * m_par.get("female_prob", 0.5) * recruitment_fn_glm(m_par)
        _correct_columns(P, survival_target)
        _correct_columns(F, fecundity_target)
    return {"K": P + F, "P": P, "F": F, "mesh_points": mesh, "mesh_width": h}


def simulate_sheep_ibm(m_par, n_years, init_pop_size, rng, max_population=5000, retain="pooled"):
    initial_mean = m_par["rcsz_int"] + m_par["rcsz_slope"] * 3.2
    z = rng.normal(initial_mean, m_par["rcsz_noise"], init_pop_size)
    frames = []
    final_frame = None
    pop_size = []
    mean_size = []
    mean_reproductive_size = []
    year = 1
    while year < n_years and 0 < len(z) < max_population:
        n = len(z)
        surv = rng.binomial(1, survival_fn_glm(z, m_par))
        survived = surv == 1
        z1 = np.full(n, np.nan)
        z1[survived] = rng.normal(m_par["growth_int"] + m_par["growth_slope"] * z[survived], m_par["growth_noise"])
        repr_ = np.full(n, np.nan)
        repr_[survived] = rng.binomial(1, reproduction_fn_glm(z[survived], m_par))
        reproduced = survived & (repr_ == 1)
        sex = np.full(n, np.nan)
        sex[reproduced] = rng.binomial(1, m_par.get("female_prob", 0.5), reproduced.sum())
        female = reproduced & (sex == 1)
        recr = np.full(n, np.nan)
        recr[female] = rng.binomial(1, recruitment_fn_glm(m_par), female.sum())
        recruited = female & (recr == 1)
        rcsz = np.full(n, np.nan)
        rcsz[recruited] = rng.normal(m_par["rcsz_int"] + m_par["rcsz_slope"] * z[recruited], m_par["rcsz_noise"])
        final_frame = pd.DataFrame({"z": z, "Surv": surv, "z1": z1, "Repr": repr_, "Sex": sex, "Recr": recr, "Rcsz": rcsz, "yr": year})
        if retain == "pooled":
            frames.append(final_frame)
        pop_size.append(n)
        mean_size.append(z.mean())
        mean_reproductive_size.append(z[reproduced].mean() if reproduced.any() else np.nan)
        z = np.concatenate([rcsz[recruited], z1[survived]])
        year += 1
    if final_frame is None:
        raise RuntimeError("The IBM completed no simulation years")
    data = pd.concat(frames, ignore_index=True) if retain == "pooled" else final_frame.reset_index(drop=True)
    return {"data": data, "pop_size": np.asarray(pop_size), "mean_size": np.asarray(mean_size), "mean_reproductive_size": np.asarray(mean_reproductive_size), "last_year": year - 1, "next_population_size": len(z)}


def predict_y_mean(model, z):
    import gpflow
    import tensorflow as tf
    z = np.asarray(z, dtype=np.float64)
    if z.size == 0:
        return np.empty(0)
    x = tf.convert_to_tensor(z.reshape(-1, 1), dtype=gpflow.default_float())
    mean, _ = model.predict_y(x)
    return mean.numpy().ravel()


def predict_f_mean(model, z):
    import gpflow
    import tensorflow as tf
    z = np.asarray(z, dtype=np.float64)
    if z.size == 0:
        return np.empty(0)
    x = tf.convert_to_tensor(z.reshape(-1, 1), dtype=gpflow.default_float())
    mean, _ = model.predict_f(x)
    return mean.numpy().ravel()


def sample_f(model, z):
    import gpflow
    import tensorflow as tf
    z = np.asarray(z, dtype=np.float64)
    if z.size == 0:
        return np.empty(0)
    x = tf.convert_to_tensor(z.reshape(-1, 1), dtype=gpflow.default_float())
    return model.predict_f_samples(x, num_samples=1).numpy().ravel()


def predict_f_mean_and_var(model, z):
    import gpflow
    import tensorflow as tf
    z = np.asarray(z, dtype=np.float64)
    if z.size == 0:
        return np.empty(0), np.empty(0)
    x = tf.convert_to_tensor(z.reshape(-1, 1), dtype=gpflow.default_float())
    mean, var = model.predict_f(x, full_cov=False)
    return mean.numpy().ravel(), np.maximum(var.numpy().ravel(), 0.0)


def survival_kernel_gp(z_next, z_current, params, funcs):
    p_surv = np.clip(predict_y_mean(funcs["surv"], z_current), 0.0, 1.0)
    growth_mean, growth_var = predict_f_mean_and_var(funcs["growth"], z_current)
    growth_sd = np.sqrt(growth_var + float(params["growth_noise"]) ** 2)
    return p_surv * norm.pdf(z_next, loc=growth_mean, scale=growth_sd)


def fecundity_kernel_gp(z_next, z_current, params, funcs):
    p_surv = np.clip(predict_y_mean(funcs["surv"], z_current), 0.0, 1.0)
    p_repr = np.clip(predict_y_mean(funcs["repr"], z_current), 0.0, 1.0)
    recruit_mean, recruit_var = predict_f_mean_and_var(funcs["rcsz"], z_current)
    recruit_sd = np.sqrt(recruit_var + float(params["rcsz_noise"]) ** 2)
    recruit_density = norm.pdf(z_next, loc=recruit_mean, scale=recruit_sd)
    p_recr = float(sigmoid(params["recr_int"]))
    return p_surv * p_repr * params.get("female_prob", 0.5) * p_recr * recruit_density


def make_gp_ipm(n_mesh_points, lower_size, upper_size, params, funcs, correction=True):
    h = (upper_size - lower_size) / n_mesh_points
    mesh = lower_size + (np.arange(n_mesh_points) + 0.5) * h
    z_next = mesh[:, None]
    z_current = mesh[None, :]
    P = h * survival_kernel_gp(z_next, z_current, params, funcs)
    F = h * fecundity_kernel_gp(z_next, z_current, params, funcs)
    if correction:
        p_surv = np.clip(predict_y_mean(funcs["surv"], mesh), 0.0, 1.0)
        p_repr = np.clip(predict_y_mean(funcs["repr"], mesh), 0.0, 1.0)
        fecundity_target = p_surv * p_repr * params.get("female_prob", 0.5) * float(sigmoid(params["recr_int"]))
        _correct_columns(P, p_surv)
        _correct_columns(F, fecundity_target)
    return {"K": P + F, "P": P, "F": F, "mesh_points": mesh, "mesh_width": h, "lower_size": lower_size, "upper_size": upper_size}


def calculate_ipm_metrics(K, pb):
    eigenvalues, left_vectors, right_vectors = eig(K, left=True, right=True)
    dominant = np.argmax(eigenvalues.real)
    lambda_ = eigenvalues[dominant].real
    omega = right_vectors[:, dominant].real
    omega *= -1 if omega.sum() < 0 else 1
    omega /= omega.sum()
    gamma = left_vectors[:, dominant].real
    gamma *= -1 if gamma[0] < 0 else 1
    gamma /= gamma[0]
    omega_rep = pb * omega
    omega_rep /= omega_rep.sum()
    return {"lambda": lambda_, "omega": omega, "omega_rep": omega_rep, "gamma": gamma}
