from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict

from predibase.resources.deployment import Deployment


class BatchInferenceServerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    custom_args: list[str] | None = Field(
        default=None,
        serialization_alias="customArgs",
    )
    hf_token: str | None = Field(
        default=None,
        serialization_alias="hfToken",
    )
    lorax_image_tag: str | None = Field(
        default=None,
        serialization_alias="loraxImageTag",
    )
    base_model: str = Field(
        ...,
        serialization_alias="baseModel",
    )
    quantization: str | None = Field(default=None)

    @classmethod
    def from_deployment(cls, deployment: Deployment) -> BatchInferenceServerConfig:
        return BatchInferenceServerConfig(
            base_model=deployment.model,
            quantization=deployment.quantization,
            custom_args=deployment.config.custom_args,
            hf_token=deployment.config.hf_token,
            lorax_image_tag=deployment.config.lorax_image_tag,
        )
