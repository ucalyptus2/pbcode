from __future__ import annotations

from pydantic import BaseModel, Field


class BatchInferenceJob(BaseModel):
    uuid: str
    original_request: dict = Field(validation_alias="originalRequest")
    status: str
    error: str | None = Field(default=None)
    result_path: str | None = Field(default=None, validation_alias="resultPath")
