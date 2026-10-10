"""ABC-GP-IPM sampler with diagnostics for any number of vital rates."""

import os
import sys
import time

import gpflow
import numpy as np
from ABC_GP_IPM.abc_gp_ipm import ABC_GP_IPM


class TrackedABC_GP_IPM_AgeSize(ABC_GP_IPM):
    """Track ABC proposals and use a dynamic vital-rate neighbour rule."""

    def _record_stage(self, diagnostics):
        previous = self.stage_diagnostics[-1] if self.stage_diagnostics else None
        diagnostics["cumulative_proposed"] = diagnostics["proposed"] + (previous["cumulative_proposed"] if previous else 0)
        diagnostics["cumulative_accepted"] = diagnostics["accepted"] + (previous["cumulative_accepted"] if previous else 0)
        diagnostics["cumulative_rejected"] = diagnostics["rejected"] + (previous["cumulative_rejected"] if previous else 0)
        self.stage_diagnostics.append(diagnostics)

    @staticmethod
    def _stable_mad(values):
        mad = np.nanmedian(np.abs(values - np.nanmedian(values, axis=0)), axis=0)
        return np.where(np.isfinite(mad) & (mad > 0), mad, 1.0)

    def _infinite_result(self, index):
        summaries = [np.full(len(self.name_ss), np.inf) for _ in self.POPUdata_dict]
        return index, [summaries]

    def IPM_whole_givenindex(self, index):
        """Simulate one particle and safely represent population extinction."""
        models = self.GPmodel.copy()
        for position, key in enumerate(self.MCMC_samples):
            helper = gpflow.optimizers.SamplingHelper(models[key].log_posterior_density, models[key].trainable_parameters)
            for variable, samples in zip(helper.current_state, self.MCMC_samples[key]):
                variable.assign(samples[index[position]])

        summaries = []
        simulated = self.fun_IBM_cache(dataset=self.ini_popu, models=models, **self.fun_IBM_kwargs)
        if len(simulated) <= 1:
            return self._infinite_result(index)
        compared = simulated if not self.capture else self.fun_capture(data_simu=simulated, time_step=0, **self.fun_capture_kwargs)
        summaries.append(self.fun_ss(exp_dataset=self.POPUdata_dict[0], simu_dataset=compared))

        for time_step in range(1, len(self.POPUdata_dict)):
            population = self.fun_structure(data_simu=simulated, time_step=time_step, **self.fun_structure_kwargs)
            simulated = self.fun_IBM_cache(dataset=population, models=models, **self.fun_IBM_kwargs)
            if len(simulated) <= 1:
                return self._infinite_result(index)
            compared = simulated if not self.capture else self.fun_capture(data_simu=simulated, time_step=time_step, **self.fun_capture_kwargs)
            summaries.append(self.fun_ss(exp_dataset=self.POPUdata_dict[time_step], simu_dataset=compared))
        return index, [summaries]

    def ABC_PMC(
        self,
        quantiles,
        n_particles,
        num_cores,
        details=True,
        smallest_unit=10000,
        file_path2store=os.getcwd() + "/ABC_details",
        algo_continue=False,
        hist_mad=None,
        hist_threshold=None,
        hist_index=None,
        hist_weight=None,
    ):
        os.makedirs(file_path2store, exist_ok=True)
        self.stage_diagnostics = []
        if isinstance(n_particles, int):
            sys.exit("n_particles must be an array or list")
        if len(quantiles) != len(n_particles):
            sys.exit("quantiles and n_particles must have equal lengths")
        history = [hist_mad, hist_threshold, hist_index, hist_weight]
        if algo_continue and any(item is None for item in history):
            sys.exit("All history files are required to continue the algorithm")

        if not algo_continue:
            print("c=0 " + time.strftime("%H:%M:%S", time.localtime()), flush=True)
            particles = np.array(self.random_IPM_whole_para(total_samples=n_particles[0], given_weight=False, num_cores=num_cores), dtype=object)
            particle_summaries = np.array([item for block in particles[:, 1] for item in block])
            particle_indices = np.array([item for item in particles[:, 0]])
            summary_shape = particle_summaries.shape
            particle_summaries = particle_summaries.reshape(summary_shape[0], summary_shape[1] * summary_shape[2])
            self.mad = self._stable_mad(particle_summaries)
            summary_mean = np.nanmean(particle_summaries, axis=0)
            summary_median = np.nanmedian(particle_summaries, axis=0)
            distances = np.sqrt(np.sum((particle_summaries / self.mad) ** 2, axis=1))
            self.threshold = np.nanquantile(distances, quantiles[0])
            self.p_index = particle_indices
            self.weight = np.full(len(particle_indices), 1 / len(particle_indices))
            self._record_stage({
                "stage": 0,
                "target_particles": int(n_particles[0]),
                "batches": 1,
                "proposed": int(len(particle_indices)),
                "accepted": int(len(particle_indices)),
                "rejected": 0,
                "acceptance_rate": 1.0,
                "threshold_used": None,
                "threshold_calculated": float(self.threshold),
                "quantile": float(quantiles[0]),
                "unique_particles": int(np.unique(self.p_index, axis=0).shape[0]),
            })
            print(f"Total: {len(particle_indices)}", flush=True)
            print(f"Left: {len(self.p_index)}", flush=True)
            print(f"Unique: {np.unique(self.p_index, axis=0).shape[0]}\n", flush=True)
            if details:
                np.save(os.path.join(file_path2store, "particles_summary0.npy"), particle_summaries)
                np.save(os.path.join(file_path2store, "p_index_0.npy"), self.p_index)
                np.save(os.path.join(file_path2store, "threshold0.npy"), self.threshold)
                np.save(os.path.join(file_path2store, "weight0.npy"), self.weight)
                np.save(os.path.join(file_path2store, "mad0.npy"), self.mad)
                np.save(os.path.join(file_path2store, "summary_mean0.npy"), summary_mean)
                np.save(os.path.join(file_path2store, "summary_median0.npy"), summary_median)
        else:
            self.p_index = np.load(hist_index)
            self.weight = np.load(hist_weight)
            self.mad = np.load(hist_mad)
            self.threshold = np.load(hist_threshold)

        for stage in range(1, len(quantiles)):
            print(f"c={stage} " + time.strftime("%H:%M:%S", time.localtime()), flush=True)
            threshold_used = float(self.threshold)
            accepted_count = 0
            proposed_count = 0
            batch_count = 0
            accepted_summaries = []
            accepted_indices = []
            all_summaries = []

            while accepted_count < n_particles[stage]:
                particles = np.array(self.random_IPM_whole_para(total_samples=smallest_unit, given_weight=True, num_cores=num_cores), dtype=object)
                particle_summaries = np.array([item for block in particles[:, 1] for item in block])
                particle_indices = np.array([item for item in particles[:, 0]])
                summary_shape = particle_summaries.shape
                particle_summaries = particle_summaries.reshape(summary_shape[0], summary_shape[1] * summary_shape[2])
                distances = np.sqrt(np.sum((particle_summaries / self.mad) ** 2, axis=1))
                accepted = distances <= self.threshold
                accepted_summaries.extend(particle_summaries[accepted])
                accepted_indices.extend(particle_indices[accepted])
                all_summaries.extend(particle_summaries)
                proposed_count += len(particle_indices)
                accepted_count += int(accepted.sum())
                batch_count += 1
                print("  " + time.strftime("%H:%M:%S", time.localtime()) + f" Required: {n_particles[stage]}, Now: {accepted_count}", flush=True)

            all_summaries = np.asarray(all_summaries)
            summary_mean = np.nanmean(all_summaries, axis=0)
            summary_median = np.nanmedian(all_summaries, axis=0)
            self.mad = self._stable_mad(all_summaries)
            accepted_indices = np.asarray(accepted_indices)
            required_matches = accepted_indices.shape[1] - 1
            new_weights = []
            for index in accepted_indices:
                neighbours = np.sum(self.p_index == index, axis=1) >= required_matches
                new_weights.append(1 / np.sum(self.weight[neighbours]))
            self.weight = np.asarray(new_weights) / np.sum(new_weights)
            self.p_index = accepted_indices
            accepted_summaries = np.asarray(accepted_summaries)
            current_distances = np.sqrt(np.sum((accepted_summaries / self.mad) ** 2, axis=1))
            self.threshold = np.nanquantile(current_distances, quantiles[stage])
            retained_count = len(self.p_index)
            self._record_stage({
                "stage": int(stage),
                "target_particles": int(n_particles[stage]),
                "batches": int(batch_count),
                "proposed": int(proposed_count),
                "accepted": int(retained_count),
                "rejected": int(proposed_count - retained_count),
                "acceptance_rate": float(retained_count / proposed_count),
                "threshold_used": threshold_used,
                "threshold_calculated": float(self.threshold),
                "quantile": float(quantiles[stage]),
                "unique_particles": int(np.unique(self.p_index, axis=0).shape[0]),
            })
            print(f"Left: {retained_count}", flush=True)
            print(f"Unique: {np.unique(self.p_index, axis=0).shape[0]}\n", flush=True)
            if details:
                np.save(os.path.join(file_path2store, f"particles_summary{stage}.npy"), accepted_summaries)
                np.save(os.path.join(file_path2store, f"p_index_{stage}.npy"), self.p_index)
                np.save(os.path.join(file_path2store, f"threshold{stage}.npy"), self.threshold)
                np.save(os.path.join(file_path2store, f"weight{stage}.npy"), self.weight)
                np.save(os.path.join(file_path2store, f"mad{stage}.npy"), self.mad)
                np.save(os.path.join(file_path2store, f"summary_mean{stage}.npy"), summary_mean)
                np.save(os.path.join(file_path2store, f"summary_median{stage}.npy"), summary_median)

        return self.p_index, self.threshold
