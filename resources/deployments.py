from __future__ import annotations

import concurrent.futures
import json
import os
import queue
import types
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import requests
from lorax import AsyncClient as AsyncLoraxClient
from lorax import Client as LoraxClient

from predibase.config import DeploymentConfig, UpdateDeploymentConfig
from predibase.resources.deployment import Deployment
from predibase.resources.util import camel_case_from_snake_case

if TYPE_CHECKING:
    from predibase import Predibase

# Mapping from key names in the old dictionary format to the corresponding key in DeploymentConfig format (None if we
# want to drop the field)
dict_to_deployment_config_keys = {
    ".quantize": "quantization",
    ".custom_args": "custom_args",
    ".source": None,
    ".num_shards": None,
    "scale_up_threshold": "scale_up_request_threshold",
}


class Deployments:
    def __init__(self, client: Predibase):
        self._client = client
        self._session = client._session

    # TODO: nameoruuid for get deployment endpoint
    def get(self, deployment_ref: str) -> Deployment:
        dep = self._client.http_get(f"/v2/deployments/{deployment_ref}")["data"]
        return Deployment.model_validate(dep)

    def client(
        self,
        deployment_ref: str | Deployment,
        force_bare_client: bool = False,
        _timeout_in_seconds: int = 600,
        _max_session_retries: int = 2,
    ) -> LoraxClient:
        if isinstance(deployment_ref, Deployment):
            deployment_ref = deployment_ref.name

        if "/" in deployment_ref:
            raise ValueError(
                f"Deployment name {deployment_ref} appears to be invalid. Are you providing a Hugging "
                f"Face path by accident? See https://docs.predibase.com/user-guide/inference/models for "
                f"a list of available deployments.",
            )

        url = (
            f"https://{self._session.serving_http_endpoint}/{self._session.tenant}/deployments/v2/llms/"
            f"{deployment_ref}"
        )

        c = LoraxClient(
            base_url=url,
            headers=self._session.get_headers(),
            timeout=_timeout_in_seconds,
            max_session_retries=_max_session_retries,
        )

        if force_bare_client:
            return c

        c._ready = False

        c._generate = c.generate
        c.generate = types.MethodType(_make_generate(self._client, deployment_ref), c)

        c._generate_stream = c.generate_stream
        c.generate_stream = types.MethodType(_make_generate_stream(self._client, deployment_ref), c)

        return c

    def async_client(self, deployment_ref: str | Deployment, _timeout_in_seconds: int = 600) -> AsyncLoraxClient:
        if isinstance(deployment_ref, Deployment):
            deployment_ref = deployment_ref.name

        url = (
            f"https://{self._session.serving_http_endpoint}/{self._session.tenant}/deployments/v2/llms/"
            f"{deployment_ref}"
        )

        return AsyncLoraxClient(
            base_url=url,
            headers=self._session.get_headers(),
            timeout=_timeout_in_seconds,
        )

    def openai_compatible_endpoint(self, deployment_ref: str | Deployment) -> str:
        if isinstance(deployment_ref, Deployment):
            deployment_ref = deployment_ref.name

        return (
            f"https://{self._session.serving_http_endpoint}/{self._session.tenant}/deployments/v2/llms/"
            f"{deployment_ref}/v1"
        )

    def create(
        self,
        name: str,
        *,
        config: dict | DeploymentConfig,
        description: str | None = None,
    ) -> Deployment:

        # If we have the config in the form of a dictionary, then we should
        # transform the keys into camelCase from snakeCase
        if isinstance(config, dict):
            config = self.convert_dict_to_deployment_config(config)

        if isinstance(config, DeploymentConfig):
            config = config.model_dump(mode="json", by_alias=True)

        payload = {
            "name": name,
            "config": config,
            "description": description or "",
        }

        self._client.http_post(
            "/v2/deployments",
            json=payload,
        )

        self._session.get_llm_deployment_events_until_with_logging(
            events_endpoint=f"/llms/{name}/events?detailed=false",
            success_cond=lambda resp: "Ready" in [r.get("eventType", None) for r in resp.get("ComputeEvents", [])],
            error_cond=lambda resp: "Failed" in [r.get("eventType", None) for r in resp.get("ComputeEvents", [])]
            or resp.get("deploymentStatus", None) in ("failed", "deleted", "stopped"),
        )

        return self.get(name)

    @staticmethod
    def convert_dict_to_deployment_config(config):
        new_config = {}
        for key, value in config.items():
            # Some key names have changed in DeploymentConfig, or we want to ignore entirely.
            key = dict_to_deployment_config_keys.get(key, key)
            if key is None:
                continue
            # Expected to be a map in dict format but a list in DeploymentConfig format
            if key == "custom_args":
                new_value = []
                for k, v in value.items():
                    new_value.append(k)
                    if v != "":
                        new_value.append(v)
                value = new_value

            new_config[camel_case_from_snake_case(key)] = value
        print(f"nc: {new_config}")
        return new_config

    def list(self, *, type: str | None = None) -> list[Deployment]:
        endpoint = "/v2/deployments"

        if type is not None:
            type = type.lower()
            if type not in ("serverless", "dedicated", "shared", "private"):
                raise ValueError("Type filter must be one of `shared` or `private`")

            endpoint = f"{endpoint}?type={type}"

        resp = self._client.http_get(endpoint)
        return [Deployment.model_validate(d) for d in resp["data"]]

    def update(
        self,
        deployment_ref: str | Deployment,
        *,
        description: str | None = None,
        config: UpdateDeploymentConfig | None = None,
    ) -> Deployment:
        if isinstance(deployment_ref, Deployment):
            deployment_ref = deployment_ref.name

        payload = {}
        if config is not None:
            payload["config"] = config.model_dump(mode="json", by_alias=True)

        if description is not None:
            payload["description"] = description

        self._client.http_put(
            f"/v2/deployments/{deployment_ref}",
            json=payload,
        )

        self._session.get_llm_deployment_events_until_with_logging(
            events_endpoint=f"/llms/{deployment_ref}/events?detailed=false",
            success_cond=lambda resp: "Ready" in [r.get("eventType", None) for r in resp.get("ComputeEvents", [])],
            error_cond=lambda resp: "Failed" in [r.get("eventType", None) for r in resp.get("ComputeEvents", [])]
            or resp.get("deploymentStatus", None) in ("failed", "deleted", "stopped"),
        )

    def delete(self, deployment_ref: str | Deployment):
        if isinstance(deployment_ref, Deployment):
            deployment_ref = deployment_ref.name

        self._client.http_delete(f"/v2/deployments/{deployment_ref}")

    def get_recommended_config(
        self,
        base_model: str,
        accelerator: str | None = None,
        quantization: str | None = None,
    ) -> DeploymentConfig:
        if base_model is None:
            raise ValueError("Base model must be specified")

        params = {
            "baseModel": base_model,
        }
        if accelerator is not None:
            params["accelerator"] = accelerator
        if quantization is not None:
            params["quantization"] = quantization

        return self._client.http_get("/v2/deployments/recommended-config", params=params)

    def download_request_logs(
        self,
        deployment_ref: str,
        dest: os.PathLike | None = None,
        *,
        adapter_id: str | None = None,
        from_: str | None = None,
        to: str | None = None,
    ):
        if dest is None:
            dest = os.path.join(os.getcwd(), f"{deployment_ref}_logs.zip")

        if not dest.endswith(".zip"):
            dest = f"{dest}.zip"

        if os.path.isdir(dest):
            dest = os.path.join(dest, f"{deployment_ref}_logs.zip")

        if from_ is None:
            from_ = datetime.utcnow().replace(minute=0, second=0, microsecond=0).isoformat()

        if to is None:
            to = datetime.utcnow() + timedelta(hours=1)
            to = to.replace(minute=0, second=0, microsecond=0).isoformat()

        print(f"Downloading logs for deployment {deployment_ref} - {adapter_id} as {dest}...")

        endpoint = self._client.api_gateway + f"/v2/deployments/{deployment_ref}/request-logs?from={from_}&to={to}"
        if adapter_id is not None:
            endpoint += f"&adapterId={adapter_id}"

        with self._client._http.get(endpoint, headers=self._client.default_headers) as r:
            try:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    f.write(r.content)
                print("Done!")
            except requests.HTTPError as e:
                try:
                    response_text_obj = json.loads(r.text)
                    if "error" in response_text_obj:
                        if "message" in response_text_obj["error"]:
                            raise RuntimeError(response_text_obj["error"]["message"]) from e
                    raise e
                except json.JSONDecodeError as je:
                    raise ValueError(f"Encountered unexpected problem while decoding error: {r.text}") from je


def _make_generate(pb: Predibase, deployment_ref: str):
    def _lorax_generate(self, *args, **kwargs):
        if self._ready:
            return self._generate(*args, **kwargs)

        def _generate_thread(q: queue.Queue):
            try:
                q.put_nowait({"type": "generate", "data": self._generate(*args, **kwargs)})
            except Exception as e:
                q.put_nowait({"type": "generate", "exception": e})

        def _status_thread(q: queue.Queue):
            try:
                q.put_nowait({"type": "status", "status": pb.deployments.get(deployment_ref).status})
            except Exception:
                # q.put_nowait({"type": "status", "exception": e})
                pass

        q = queue.Queue()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            pool.submit(_generate_thread, q)
            pool.submit(_status_thread, q)

            while True:
                resp = q.get()
                if resp["type"] == "generate":
                    if "exception" in resp:
                        raise resp["exception"]

                    self._ready = True
                    return resp["data"]

                if resp["type"] == "status":
                    if resp["status"] not in ("ready", "updating"):
                        print(
                            f"Deployment {deployment_ref} is still spinning up. Your prompt may take longer than "
                            f"normal to execute.\n",
                        )
                    else:
                        self._ready = True

    return _lorax_generate


def _make_generate_stream(pb: Predibase, deployment_ref: str):
    def _lorax_generate_stream(self, *args, **kwargs):
        if self._ready:
            return self._generate_stream(*args, **kwargs)

        def _generate_thread(q: queue.Queue):
            try:
                for r in self._generate_stream(*args, **kwargs):
                    q.put_nowait({"type": "generate", "data": r})

                q.put_nowait(None)
            except Exception as e:
                q.put_nowait({"type": "generate", "exception": e})

        def _status_thread(q: queue.Queue):
            try:
                q.put_nowait({"type": "status", "status": pb.deployments.get(deployment_ref).status})
            except Exception as e:
                q.put_nowait({"type": "status", "exception": e})

        q = queue.Queue()
        resp_seen = False
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            pool.submit(_generate_thread, q)
            pool.submit(_status_thread, q)

            while True:
                resp = q.get()

                if resp is None:
                    break

                if resp["type"] == "generate":
                    if "exception" in resp:
                        raise resp["exception"]

                    resp_seen = True
                    self._ready = True
                    yield resp["data"]
                    continue

                if resp["type"] == "status" and not resp_seen:
                    if resp["status"] not in ("ready", "updating"):
                        print(
                            f"Deployment {deployment_ref} is still spinning up. Your prompt may take longer than "
                            f"normal to execute.\n",
                        )
                    else:
                        self._ready = True

    return _lorax_generate_stream
