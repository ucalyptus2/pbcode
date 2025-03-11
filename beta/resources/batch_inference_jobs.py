from __future__ import annotations

from typing import TYPE_CHECKING

import requests

from predibase.beta.config import BatchInferenceServerConfig
from predibase.beta.resources.batch_inference_job import BatchInferenceJob
from predibase.resources.dataset import Dataset

if TYPE_CHECKING:
    from predibase import Predibase


class BatchInferenceJobs:
    def __init__(self, client: Predibase):
        self._client = client

    def create(
        self,
        *,
        dataset: str | Dataset,
        server_config: BatchInferenceServerConfig,
        # generation_params: dict,  TODO
        # metadata: dict  TODO
    ) -> BatchInferenceJob:

        if isinstance(dataset, Dataset):
            dataset = dataset.name

        resp = self._client.http_post(
            "/v2/batch-inference",
            json={
                "dataset": dataset,
                "serverConfig": server_config.model_dump(by_alias=True),
            },
        )

        job = BatchInferenceJob.model_validate(resp)

        print(
            f"Successfully requested batch inference over {dataset} using {server_config.base_model} as "
            f"{job.uuid}.\n"
        )

        return job

    def get(self, job: str | BatchInferenceJob) -> BatchInferenceJob:
        if isinstance(job, BatchInferenceJob):
            job = job.uuid

        resp = self._client.http_get(f"/v2/batch-inference/{job}")
        return BatchInferenceJob.model_validate(resp)

    def list(self, limit: int = 10) -> list[BatchInferenceJob]:
        resp = self._client.http_get(f"/v2/batch-inference?limit={limit}")
        return [BatchInferenceJob.model_validate(j) for j in resp["data"]["jobs"]]

    def cancel(self, job: str | BatchInferenceJob):
        if isinstance(job, BatchInferenceJob):
            job = job.uuid

        self._client.http_get(f"/v2/batch-inference/{job}/cancel")

    def download_results(self, job: str | BatchInferenceJob, *, dest: str):
        if isinstance(job, BatchInferenceJob):
            job = job.uuid

        resp = self._client.http_get(f"/v2/batch-inference/{job}/results")

        r = requests.get(resp["url"])
        r.raise_for_status()

        with open(dest, "wb") as f:
            f.write(r.content)
