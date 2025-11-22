from __future__ import annotations

from typing import TYPE_CHECKING

from predibase.beta.resources.batch_inference_jobs import BatchInferenceJobs

if TYPE_CHECKING:
    from predibase import Predibase


class Beta:
    def __init__(self, client: Predibase):
        self.batch_inference = BatchInferenceJobs(client)
