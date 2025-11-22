from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import AliasPath, BaseModel, Field

from predibase.config import UpdateDeploymentConfig

if TYPE_CHECKING:
    pass


class Deployment(BaseModel):
    name: str
    uuid: str
    description: str | None
    type: str
    status: str
    context_window: int = Field(validation_alias=AliasPath("model", "maxInputLength"))
    accelerator: str = Field(validation_alias=AliasPath("accelerator", "id"))
    quantization: str | None = Field(default=None)
    model: str = Field(validation_alias=AliasPath("model", "name"))
    current_replicas: int = Field(validation_alias="currentReplicas")
    config: UpdateDeploymentConfig = Field(
        validation_alias="config",
    )
    uses_guaranteed_capacity: bool = Field(
        validation_alias="usesGuaranteedCapacity",
    )
