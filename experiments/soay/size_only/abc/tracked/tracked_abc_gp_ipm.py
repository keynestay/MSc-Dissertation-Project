"""ABC-GP-IPM sampler with per-stage proposal diagnostics."""

import os
import sys
import time

import numpy as np
from ABC_GP_IPM.abc_gp_ipm import ABC_GP_IPM


class TrackedABC_GP_IPM(ABC_GP_IPM):
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
            sys.exit("n_particle should be an array or a list!")
        if len(quantiles) != len(n_particles):
            sys.exit("len(quantiles) should equal to len(n_particles)")
        missing_history = any(
            item is None
            for item in [hist_mad, hist_threshold, hist_index, hist_weight]
        )
        if algo_continue and missing_history:
            sys.exit("Please provide all the history files to continue the algorithm.")

        if not algo_continue:
            print("c=0 " + time.strftime("%H:%M:%S", time.localtime()), flush=True)
            particles = np.array(
                self.random_IPM_whole_para(
                    total_samples=n_particles[0],
                    given_weight=False,
                    num_cores=num_cores,
                ),
                dtype=object,
            )
            particles_summary = np.array(
                [item for sublist in particles[:, 1] for item in sublist]
            )
            particles_index = np.array([item for item in particles[:, 0]])
            del particles
            if details:
                np.save(file_path2store + "/p_index_ini.pkl", particles_index)

            summary_shape = particles_summary.shape
            particles_summary = particles_summary.reshape(
                (summary_shape[0], summary_shape[1] * summary_shape[2])
            )
            self.mad = np.nanmedian(
                np.absolute(
                    particles_summary - np.nanmedian(particles_summary, axis=0)
                ),
                axis=0,
            )
            summary_mean = np.nanmean(particles_summary, axis=0)
            summary_median = np.nanmedian(particles_summary, axis=0)
            distance = np.sqrt(
                np.sum((particles_summary / self.mad) ** 2, axis=1)
            )
            self.threshold = np.nanquantile(distance, q=quantiles[0])
            accepted = np.repeat(True, particles_index.shape[0])
            self.p_index = particles_index[accepted]
            self.weight = np.repeat(1 / np.sum(accepted), np.sum(accepted))
            self.stage_diagnostics.append(
                {
                    "stage": 0,
                    "target_particles": int(n_particles[0]),
                    "batches": 1,
                    "proposed": int(particles_index.shape[0]),
                    "accepted": int(self.p_index.shape[0]),
                    "rejected": 0,
                    "acceptance_rate": 1.0,
                    "threshold_used": None,
                    "threshold_calculated": float(self.threshold),
                    "quantile": float(quantiles[0]),
                    "unique_particles": int(np.unique(self.p_index, axis=0).shape[0]),
                }
            )
            print(f"Total: {particles_index.shape[0]}", flush=True)
            print(f"Left: {self.p_index.shape[0]}", flush=True)
            print(
                f"Unique: {np.unique(self.p_index, axis=0).shape[0]}\n", flush=True
            )
            if details:
                np.save(file_path2store + "/particles_summary0.pkl", particles_summary)
                np.save(file_path2store + "/accepted0.pkl", accepted)
                np.save(file_path2store + "/p_index_0.pkl", self.p_index)
                np.save(file_path2store + "/threshold0.pkl", self.threshold)
                np.save(file_path2store + "/weight0.pkl", self.weight)
                np.save(file_path2store + "/mad0.pkl", self.mad)
                np.save(file_path2store + "/r_s_mean0.pkl", summary_mean)
                np.save(file_path2store + "/r_s_median0.pkl", summary_median)
            del particles_summary, particles_index, summary_shape
        else:
            self.p_index = np.load(hist_index)
            self.weight = np.load(hist_weight)
            self.mad = np.load(hist_mad)
            self.threshold = np.load(hist_threshold)

        for stage in range(1, len(quantiles)):
            print(
                f"c={stage} " + time.strftime("%H:%M:%S", time.localtime()),
                flush=True,
            )
            threshold_used = float(self.threshold)
            accepted_count = 0
            proposed_count = 0
            batch_count = 0
            accepted_summaries = []
            accepted_indices = []
            all_summaries = []

            while accepted_count < n_particles[stage]:
                particles = np.array(
                    self.random_IPM_whole_para(
                        total_samples=smallest_unit,
                        given_weight=True,
                        num_cores=num_cores,
                    ),
                    dtype=object,
                )
                particles_summary = np.array(
                    [item for sublist in particles[:, 1] for item in sublist]
                )
                particles_index = np.array([item for item in particles[:, 0]])
                del particles
                summary_shape = particles_summary.shape
                particles_summary = particles_summary.reshape(
                    (summary_shape[0], summary_shape[1] * summary_shape[2])
                )
                distance = np.sqrt(
                    np.sum((particles_summary / self.mad) ** 2, axis=1)
                )
                accepted = distance <= self.threshold
                accepted_summaries.extend(particles_summary[accepted])
                all_summaries.extend(particles_summary)
                accepted_indices.extend(particles_index[accepted])
                proposed_count += int(particles_index.shape[0])
                accepted_count += int(accepted.sum())
                batch_count += 1
                del particles_summary, particles_index, summary_shape, distance, accepted
                print(
                    "  "
                    + time.strftime("%H:%M:%S", time.localtime())
                    + f" Required: {n_particles[stage]}, Now: {accepted_count}",
                    flush=True,
                )

            all_summaries = np.asarray(all_summaries)
            summary_mean = np.nanmean(all_summaries, axis=0)
            summary_median = np.nanmedian(all_summaries, axis=0)
            self.mad = np.nanmedian(
                np.absolute(all_summaries - np.nanmedian(all_summaries, axis=0)),
                axis=0,
            )
            accepted_indices = np.asarray(accepted_indices)
            new_weights = []
            for index in accepted_indices:
                neighbour = np.sum(self.p_index == index, axis=1) >= 3
                new_weights.append(1 / np.sum(self.weight[neighbour]))
            self.weight = new_weights / np.sum(new_weights)
            self.p_index = accepted_indices
            accepted_summaries = np.asarray(accepted_summaries)
            current_distance = np.sqrt(
                np.sum((accepted_summaries / self.mad) ** 2, axis=1)
            )
            self.threshold = np.nanquantile(current_distance, q=quantiles[stage])
            retained_count = int(self.p_index.shape[0])
            self.stage_diagnostics.append(
                {
                    "stage": int(stage),
                    "target_particles": int(n_particles[stage]),
                    "batches": batch_count,
                    "proposed": proposed_count,
                    "accepted": retained_count,
                    "rejected": proposed_count - retained_count,
                    "acceptance_rate": retained_count / proposed_count,
                    "threshold_used": threshold_used,
                    "threshold_calculated": float(self.threshold),
                    "quantile": float(quantiles[stage]),
                    "unique_particles": int(np.unique(self.p_index, axis=0).shape[0]),
                }
            )
            print(f"Left: {self.p_index.shape[0]}", flush=True)
            print(
                f"Unique: {np.unique(self.p_index, axis=0).shape[0]}\n", flush=True
            )
            if details:
                np.save(
                    file_path2store + f"/particles_summary{stage}.pkl",
                    accepted_summaries,
                )
                np.save(file_path2store + f"/p_index_{stage}.pkl", self.p_index)
                np.save(file_path2store + f"/threshold{stage}.pkl", self.threshold)
                np.save(file_path2store + f"/weight{stage}.pkl", self.weight)
                np.save(file_path2store + f"/mad{stage}.pkl", self.mad)
                np.save(file_path2store + f"/r_s_mean{stage}.pkl", summary_mean)
                np.save(file_path2store + f"/r_s_median{stage}.pkl", summary_median)

        return self.p_index, self.threshold
