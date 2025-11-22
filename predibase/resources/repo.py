from __future__ import annotations

from datetime import datetime
from typing import List

from pydantic import BaseModel, Field
from typing_extensions import Annotated


class Version(BaseModel):
    tag: int
    created_at: datetime = Field(validation_alias="createdAt")
    created_by: str = Field(validation_alias="createdBy")
    base_model: str = Field(validation_alias="baseModel")
    archived: bool


class Repo(BaseModel):
    uuid: str
    name: str
    description: str | None = Field(default=None)
    all_versions: Annotated[List[Version], Field(default_factory=list, validation_alias="versions", repr=False)]

    @property
    def versions(self, show_archived: bool = False) -> List[Version]:
        return [v for v in self.all_versions if show_archived or not v.archived]
