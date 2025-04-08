import numpy as np

from ..scheduler import Scheduler
from .utils import preprocess_obs, find_stage


class RoundRobinScheduler(Scheduler):
    def __init__(self, num_executors, dynamic_partition=True):
        self.name = "Fair" if dynamic_partition else "FIFO"
        self.num_executors = num_executors
        self.dynamic_partition = dynamic_partition
        self.env_wrapper_cls = None

    def schedule(self, obs: dict) -> tuple[dict, dict]:
        preprocess_obs(obs)
        num_active_jobs = len(obs["exec_supplies"])

        if self.dynamic_partition:
            executor_cap = self.num_executors / max(1, num_active_jobs)
            executor_cap = int(np.ceil(executor_cap))
        else:
            executor_cap = self.num_executors

        # first, try to find a stage in the same job that is releasing executers
        if obs["source_job_idx"] < num_active_jobs:
            selected_stage_idx = find_stage(obs, obs["source_job_idx"])

            if selected_stage_idx != -1:
                return {
                    "stage_idx": selected_stage_idx,
                    "num_exec": obs["num_committable_execs"],
                }, {}

        # search through jobs by order of arrival.
        for j in range(num_active_jobs):
            if obs["exec_supplies"][j] >= executor_cap or j == obs["source_job_idx"]:
                continue

            selected_stage_idx = find_stage(obs, j)
            if selected_stage_idx == -1:
                continue

            num_exec = min(
                obs["num_committable_execs"], executor_cap - obs["exec_supplies"][j]
            )
            return {"stage_idx": selected_stage_idx, "num_exec": num_exec}, {}

        # didn't find any stages to schedule
        return {"stage_idx": -1, "num_exec": obs["num_committable_execs"]}, {}
