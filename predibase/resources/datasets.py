from __future__ import annotations

# import json
import mimetypes
import os
import pathlib
import tempfile
from os import PathLike
from typing import TYPE_CHECKING

import pandas as pd
import requests
from google.protobuf.json_format import MessageToJson
from predibase_api.artifacts.v1.artifacts_pb2 import (
    ARTIFACT_TYPE_DATASET,
    PresignedUrlForUploadRequest,
    PresignedUrlForUploadResponse,
    RegisterUploadedFileAsDatasetRequest,
)

from predibase.config import AugmentationConfig
from predibase.resources.dataset import Dataset
from predibase.resources.util import (
    parse_connection_and_dataset_name,
    strip_api_from_gateway_url,
    validate_openai_api_key,
    validate_openai_base_model_support,
)

# from predibase.resource.dataset import Dataset

if TYPE_CHECKING:
    from predibase import Predibase


class Datasets:
    def __init__(self, client: Predibase):
        self._client = client
        self._session = client._session

    def get(self, dataset_ref: str | Dataset) -> Dataset:
        if isinstance(dataset_ref, Dataset):
            dataset_ref = f"{dataset_ref.connection_name}/{dataset_ref.name}"

        connection, name = parse_connection_and_dataset_name(dataset_ref)

        dataset_resp = self._client.http_get(f"/v1/datasets/name/{name}?connectionName={connection}")
        return Dataset.model_validate(dataset_resp)

    def download(self, dataset_ref: str | Dataset, *, dest: str):
        if isinstance(dataset_ref, Dataset):
            dataset_ref = dataset_ref.uuid

        resp = self._client.http_get(f"/v2/datasets/{dataset_ref}/download-link")

        extension = resp.get("fileExtension", "")
        if extension != "" and not dest.endswith(f".{extension}"):
            dest = dest + f".{extension}"

        r = requests.get(resp["url"])
        r.raise_for_status()

        with open(dest, "wb") as f:
            f.write(r.content)

    def from_file(self, file_path: PathLike, *, name: str | None = None) -> Dataset:
        """Connects the specified file as a Predibase Dataset for training.

        # Inputs
        :param file_path: (str) The file path name for the training dataset (columns "prompt" and "completion" are
            expected).
        :param name: (str) Optional name of the dataset (default is None).
        :return: (Dataset) The connected Predibase Dataset object.
        """
        if name is None:
            name = pathlib.Path(file_path).stem

        with open(file_path, "rb") as f:
            file_name = os.path.basename(file_path)
            mime_type = mimetypes.guess_type(file_path)[0] or "text/plain"

            # Get presigned url for upload
            presigned_url_req = MessageToJson(
                PresignedUrlForUploadRequest(
                    mime_type=mime_type,
                    expiry_seconds=3600,
                    artifact_type=ARTIFACT_TYPE_DATASET,
                ),
                preserving_proto_field_name=True,
            )
            resp = PresignedUrlForUploadResponse(
                **self._session.post(
                    "/datasets/get_presigned_url",
                    data=presigned_url_req,
                ),
            )

            # Unpack response data
            presigned_url = resp.url
            object_name = resp.object_name
            required_headers = resp.required_headers

            # Get required headers for presigned url upload
            headers = {"Content-Type": mime_type}
            for k, v in required_headers:
                headers[k] = v

            # Upload file to blob storage with pre-signed url
            requests.put(presigned_url, data=f, headers=headers).raise_for_status()

            # Register uploaded file as dataset
            register_dataset_req = MessageToJson(
                RegisterUploadedFileAsDatasetRequest(
                    dataset_name=name,
                    object_name=object_name,
                    file_name=file_name,
                ),
                preserving_proto_field_name=True,
            )
            resp = self._session.post("/datasets/register_uploaded_file", data=register_dataset_req)

            dataset_id = resp["id"]
            endpoint = f"/datasets/{dataset_id}?withInfo=true"
            resp = self._session.wait_for_dataset(endpoint, until_fully_connected=True)
            return Dataset.model_validate(resp)

    def from_pandas_dataframe(self, df: pd.DataFrame, *, name: str | None = None) -> Dataset:
        """Connects the specified Pandas DataFrame as a Predibase Dataset for training.

        # Inputs
        :param df: (pd.DataFrame) The Pandas DataFrame reference for the training dataset (columns "prompt" and
            "completion" are expected).
        :param name: (str) Optional name of the dataset (default is None).
        :return: (Dataset) The connected Predibase Dataset object.
        """
        # The file "temp_file" will be automatically deleted when context exits.
        with tempfile.NamedTemporaryFile(suffix=".csv") as temp_file:
            # Write Pandas DataFrame to the temporary file.
            df.to_csv(path_or_buf=temp_file.name, index=False)
            return self.from_file(file_path=temp_file.name, name=name)

    def augment(
        self,
        config: dict | AugmentationConfig,
        dataset: Dataset,
        name: str | None = None,
        openai_api_key: str | None = None,
    ) -> Dataset:
        """Augments the specified DataFrame using the specified configuration.

        # Inputs
        :param config: (dict) The configuration for the augmentation.
        :param dataset: (Dataset) The dataset to augment.
        :param name: (str) The name of the generated dataset, must be unique in the connection.
        :param openai_api_key: (str) The OpenAI API key. If not provided, will be read from `OPENAI_API_KEY`.

        # Returns
        :return: (Dataset) The augmented dataset object.
        """
        if isinstance(config, dict):
            config = AugmentationConfig(**config)

        if openai_api_key is None:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if openai_api_key is None:
                raise ValueError(
                    "An OpenAI API key must be provided to augment a dataset. Please pass the key as an "
                    "argument through `openai_api_key` or set the OPENAI_API_KEY environment variable.",
                )

        client = validate_openai_api_key(openai_api_key)
        validate_openai_base_model_support(config.base_model, client)

        augment_response = self._client.http_post(
            f"/v2/datasets/{dataset.uuid}/augment",
            json={
                "name": name,
                "config": dict(config),
                "openaiApiKey": openai_api_key,
            },
        )

        augmented_dataset_id = augment_response["datasetId"]
        base_url = strip_api_from_gateway_url(self._session.url)
        augment_dataset_url = f"{base_url}/data/datasets/{augmented_dataset_id}"
        print(f"Augmentation started! Check for augmentation status at {augment_dataset_url}")

        url = f"/datasets/{augmented_dataset_id}?withInfo=true"
        augmented_dataset = self._session.wait_for_dataset(url, until_fully_connected=True)
        return Dataset.model_validate(augmented_dataset)
