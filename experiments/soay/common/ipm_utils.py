import numpy as np
import pandas as pd
from scipy.linalg import eig
from scipy.special import expit
from scipy.stats import norm


def sigmoid(x):
    return expit(x)

def survival_fn_glm(z, m_par, age=None, use_age=False):
    linear_predictor = m_par["surv_int"] + m_par["surv_slope"] * z
    if use_age:
        linear_predictor += m_par["surv_age"] * age
    return sigmoid(linear_predictor)


def reproduction_fn_glm(z, m_par, age=None, use_age=False):
    linear_predictor = m_par["repr_int"] + m_par["repr_slope"] * z
    if use_age:
        linear_predictor += m_par["repr_age"] * age
    return sigmoid(linear_predictor)


def recruitment_fn_glm(m_par, age=None, use_age=False):
    linear_predictor = m_par["recr_int"]
    if use_age:
        linear_predictor += m_par["recr_age"] * age
    return sigmoid(linear_predictor)


def growth_fn_glm(z1, z, m_par, age=None, use_age=False):
    mean = m_par["growth_int"] + m_par["growth_slope"] * z
    if use_age:
        mean += m_par["growth_age"] * age
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
    # Applies 'constant' style correction for eviction by rescaling each column 
    # of the matrix so that it sums to target
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


def mk_K_glm_agesize(n_mesh_points, 
                     m_par, 
                     lower_size, 
                     upper_size, 
                     max_age, 
                     vital_rate_functions=None, 
                     correction=True, 
                     use_age=False, 
                     min_reproductive_age=1):
    funcs = vital_rate_functions or {}
    surv = funcs.get("surv", survival_fn_glm)
    growth = funcs.get("growth", growth_fn_glm)
    repr_fn = funcs.get("repr", reproduction_fn_glm)
    recruit = funcs.get("recr", recruitment_fn_glm)
    recruit_size = funcs.get("rcsz", recruit_size_fn_glm)

    mesh_width = (upper_size - lower_size) / n_mesh_points
    mesh = lower_size + (np.arange(n_mesh_points) + 0.5) * mesh_width
    z1, z = np.meshgrid(mesh, mesh, indexing="ij")

    n_ages = max_age + 2
    n_states = n_ages * n_mesh_points

    P = np.zeros((n_states, n_states))
    F = np.zeros((n_states, n_states))

    age_zero_rows = slice(0, n_mesh_points) 

    for age in range(n_ages):
        # Select columns the matrix this age-group can contribute to. Most of the rows will be 0.
        source_columns = slice(age * n_mesh_points, (age + 1) * n_mesh_points) 

        survival_probability = surv(z, m_par, age, use_age)
        growth_density = growth(z1, z, m_par, age, use_age)
        P_block = mesh_width * survival_probability * growth_density

        if correction:
            survival_target = surv(mesh, m_par, age, use_age)
            _correct_columns(P_block, survival_target)

        next_age = age + 1 if age < n_ages - 1 else age
        #Growth to age+1
        survival_rows = slice(next_age * n_mesh_points, (next_age + 1) * n_mesh_points) 
        P[survival_rows, source_columns] = P_block

        if age >= min_reproductive_age: 
            reproduction_probability = repr_fn(z, m_par, age, use_age)
            recruitment_probability = recruit(m_par, age, use_age)
            recruit_size_density = recruit_size(z1, z, m_par)

            F_block = (
                mesh_width
                * survival_probability
                * reproduction_probability
                * m_par.get("female_prob", 0.5)
                * recruitment_probability
                * recruit_size_density
            )

            if correction:
                fecundity_target = (
                    surv(mesh, m_par, age, use_age)
                    * repr_fn(mesh, m_par, age, use_age)
                    * m_par.get("female_prob", 0.5)
                    * recruit(m_par, age, use_age)
                )
                _correct_columns(F_block, fecundity_target)
            #Reproduce and contribute to age 0 population
            F[age_zero_rows, source_columns] = F_block

    return {
        "K": P + F,
        "P": P,
        "F": F,
        "mesh_points": mesh,
        "mesh_width": mesh_width,
        "n_ages": n_ages,
        "max_age": max_age,
    }

def simulate_sheep_ibm(m_par, 
                       n_years, 
                       init_pop_size, 
                       rng, 
                       max_population=5000, 
                       use_age = False, 
                       min_reproductive_age=1,
                       max_age = None,
                       retain="pooled"):
    '''
    Simulates the original dataset using IBM, with GLM parameterized vital rates.
    GLM parameters are taken from Ellner et al. 2016
    Size only IPM: https://github.com/ipmbook/first-edition/tree/master/Rcode/c2
    Age-size IPM:  https://github.com/ipmbook/first-edition/tree/master/Rcode/c6
    '''

    initial_mean = m_par["rcsz_int"] + m_par["rcsz_slope"] * 3.2 
    z = rng.normal(initial_mean, m_par["rcsz_noise"], init_pop_size)
    age = np.zeros(init_pop_size, dtype=int)
    frames = []
    final_frame = None
    pop_size = []
    mean_size = []
    mean_reproductive_size = []
    year = 1
    while year < n_years and 0 < len(z) < max_population:
        n = len(z)
        surv = rng.binomial(1, survival_fn_glm(z, m_par, age, use_age))
        survived = surv == 1

        z1 = np.full(n, np.nan)
        growth_mean = m_par["growth_int"] + m_par["growth_slope"] * z[survived]
        if use_age:
            growth_mean += m_par["growth_age"] * age[survived]
        z1[survived] = rng.normal(growth_mean, m_par["growth_noise"])

        repr_ = np.full(n, np.nan)
        repr_[survived] = 0

        eligible_to_reproduce = survived & (age >= min_reproductive_age)

        repr_[eligible_to_reproduce] = rng.binomial(
            1,
            reproduction_fn_glm(
                z[eligible_to_reproduce],
                m_par,
                age[eligible_to_reproduce],
                use_age,
            ),
        )
        reproduced = eligible_to_reproduce & (repr_ == 1)

        sex = np.full(n, np.nan)
        sex[reproduced] = rng.binomial(1, m_par.get("female_prob", 0.5), reproduced.sum())

        female = reproduced & (sex == 1)
        recr = np.full(n, np.nan)
        recr[female] = rng.binomial(1, recruitment_fn_glm(m_par, age[female], use_age), female.sum())

        recruited = female & (recr == 1)
        rcsz = np.full(n, np.nan)
        rcsz[recruited] = rng.normal(m_par["rcsz_int"] + m_par["rcsz_slope"] * z[recruited], m_par["rcsz_noise"])

        final_frame = pd.DataFrame({"z": z, 
                                    "age": age, 
                                    "Surv": surv, 
                                    "z1": z1, 
                                    "Repr": repr_, 
                                    "Sex": sex,
                                    "Recr": recr, 
                                    "Rcsz": rcsz, 
                                    "yr": year})
        if retain == "pooled":
            frames.append(final_frame)
        pop_size.append(n)
        mean_size.append(z.mean())
        mean_reproductive_size.append(z[reproduced].mean() if reproduced.any() else np.nan)
        z = np.concatenate([rcsz[recruited], z1[survived]])
        survivor_age = age[survived] + 1
        if max_age is not None:
            survivor_age = np.minimum(survivor_age, max_age + 1)
        age = np.concatenate([np.zeros(recruited.sum(), dtype=int), survivor_age])
        year += 1
    data = pd.concat(frames, ignore_index=True) if retain == "pooled" else final_frame.reset_index(drop=True)
    return {"data": data, 
            "pop_size": np.asarray(pop_size), 
            "mean_size": np.asarray(mean_size), 
            "mean_reproductive_size": np.asarray(mean_reproductive_size), 
            "last_year": year - 1, 
            "next_population_size": len(z)}


def prepare_gp_inputs(X):
    import gpflow
    import tensorflow as tf

    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 0:
        X = X.reshape(1, 1)
    elif X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim != 2:
        raise ValueError("GP inputs must be a scalar, vector, or two-dimensional matrix")

    return tf.convert_to_tensor(X, dtype=gpflow.default_float())


def predict_y_mean(model, X):
    if np.asarray(X).size == 0:
        return np.empty(0)
    X = prepare_gp_inputs(X)
    mean, _ = model.predict_y(X)
    return mean.numpy().ravel()


def predict_f_mean(model, X):
    if np.asarray(X).size == 0:
        return np.empty(0)
    X = prepare_gp_inputs(X)
    mean, _ = model.predict_f(X)
    return mean.numpy().ravel()


def sample_f(model, X):
    if np.asarray(X).size == 0:
        return np.empty(0)
    X = prepare_gp_inputs(X)
    return model.predict_f_samples(X, num_samples=1).numpy().ravel()


def predict_f_mean_and_var(model, X):
    if np.asarray(X).size == 0:
        return np.empty(0), np.empty(0)
    X = prepare_gp_inputs(X)
    mean, variance = model.predict_f(X, full_cov=False)
    return mean.numpy().ravel(), np.maximum(variance.numpy().ravel(), 0)

def survival_kernel_gp_agesize(z_next, z_current, age, params, funcs):
    '''
    Denoting M as the number of mesh points:
    z_next: (M,)
    z_current: (M,)
    age: scalar
    '''
    age_size_inputs = np.column_stack([z_current, np.full(len(z_current), age),])                   #(M,2)
    survival_probability = np.clip(predict_y_mean(funcs["surv"], age_size_inputs), 0, 1,)           #(M,)
    growth_mean, growth_variance = predict_f_mean_and_var(funcs["growth"],age_size_inputs,)         #(M,) (M,)
    growth_sd = np.sqrt(growth_variance + float(params["growth_noise"]) ** 2)
    growth_density = norm.pdf(z_next[:, None], loc=growth_mean[None, :], scale=growth_sd[None, :],) #(M,M)
    return survival_probability[None, :] * growth_density                                           #(M,M)

def fecundity_kernel_gp_agesize(z_next, z_current, age, params, funcs):
    '''
    Denoting M as the number of mesh points:
    z_next: (M,)
    z_current: (M,)
    age: scalar
    '''
    age_size_inputs = np.column_stack([z_current, np.full(len(z_current), age),])                   #(M,2)
    survival_probability = np.clip(predict_y_mean(funcs["surv"], age_size_inputs), 0, 1,)           #(M,)  
    reproduction_probability = np.clip(predict_y_mean(funcs["repr"], age_size_inputs), 0, 1,)       #(M,)
    recruitment_probability = np.clip(predict_y_mean(funcs["recr"], np.array([[age]]))[0], 0, 1,)   #(M,)
    recruit_mean, recruit_variance = predict_f_mean_and_var(funcs["rcsz"], z_current,)              #(M,), (M,)
    recruit_sd = np.sqrt(recruit_variance + float(params["rcsz_noise"]) ** 2)                   
    recruit_density = norm.pdf(z_next[:, None], loc=recruit_mean[None, :], scale=recruit_sd[None, :],) #(M,M)

    return (
        survival_probability[None, :]
        * reproduction_probability[None, :]
        * params.get("female_prob", 0.5)
        * recruitment_probability
        * recruit_density
    )

def make_gp_ipm_agesize(n_mesh_points, lower_size, upper_size, max_age, params, funcs, correction=True, min_reproductive_age=1):
    mesh_width = (upper_size - lower_size) / n_mesh_points
    mesh = lower_size + (np.arange(n_mesh_points) + 0.5) * mesh_width
    n_ages = max_age + 2
    n_states = n_ages * n_mesh_points
    P = np.zeros((n_states, n_states))
    F = np.zeros((n_states, n_states))
    reproduction_probability = np.zeros(n_states)
    age_zero_rows = slice(0, n_mesh_points)

    for age in range(n_ages):
        # Select columns the matrix this age-group can contribute to. Most of the rows will be 0.
        source_columns = slice(age * n_mesh_points, (age + 1) * n_mesh_points)
        age_size_inputs = np.column_stack([mesh, np.full(n_mesh_points, age)])
        p_surv = np.clip(predict_y_mean(funcs["surv"], age_size_inputs), 0.0, 1.0)
        P_block = mesh_width * survival_kernel_gp_agesize(mesh, mesh, age, params, funcs)

        if correction:
            _correct_columns(P_block, p_surv)

        next_age = age + 1 if age < n_ages - 1 else age
        survival_rows = slice(next_age * n_mesh_points, (next_age + 1) * n_mesh_points)
        #Growth to age+1
        P[survival_rows, source_columns] = P_block

        if age >= min_reproductive_age:
            p_repr = np.clip(predict_y_mean(funcs["repr"], age_size_inputs), 0.0, 1.0)
            p_recr = float(np.clip(predict_y_mean(funcs["recr"], np.array([[age]]))[0], 0.0, 1.0))
            F_block = mesh_width * fecundity_kernel_gp_agesize(mesh, mesh, age, params, funcs)

            if correction:
                fecundity_target = p_surv * p_repr * params.get("female_prob", 0.5) * p_recr
                _correct_columns(F_block, fecundity_target)

            #Reproduce and contribute to age 0 population
            F[age_zero_rows, source_columns] = F_block
            reproduction_probability[source_columns] = p_repr

    return {
        "K": P + F,
        "P": P,
        "F": F,
        "mesh_points": mesh,
        "mesh_width": mesh_width,
        "n_ages": n_ages,
        "max_age": max_age,
        "lower_size": lower_size,
        "upper_size": upper_size,
        "reproduction_probability": reproduction_probability,
    }


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
    z_current = mesh
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
    #pb is the probability of reproduction, evaluated at meshpoints
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
