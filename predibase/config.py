from __future__ import annotations

import ast
import base64
import inspect
from functools import cached_property
from typing import Callable, Literal, TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, PositiveFloat, PositiveInt

if TYPE_CHECKING:
    pass


class SamplingParamsConfig(BaseModel):
    temperature: float | None = Field(default=None)
    top_p: float | None = Field(default=None)
    top_k: int | None = Field(default=None)
    max_tokens: int | None = Field(default=None)


class FinetuningConfig(BaseModel):
    # This is required in most cases, but coded as optional for continued training.
    base_model: str | None = Field(default=None)

    adapter: str | None = Field(default=None)
    task: Literal["sft", "continued_pretraining", "grpo"] | None = Field(default=None)
    epochs: int | None = Field(default=None)
    learning_rate: float | None = Field(default=None)
    rank: int | None = Field(default=None)
    target_modules: list[str] | None = Field(default=None)
    enable_early_stopping: bool | None = Field(default=None)
    apply_chat_template: bool | None = Field(default=None)
    lr_scheduler: dict | None = Field(default=None)
    optimizer: dict | None = Field(default=None)
    lora_alpha: PositiveInt | None = Field(default=None)
    lora_dropout: PositiveFloat | None = Field(default=None)
    warmup_ratio: PositiveFloat | None = Field(default=None)
    effective_batch_size: PositiveInt | None = Field(default=None)


class SFTConfig(FinetuningConfig):
    task: Literal["sft"] = "sft"


class ContinuedPretrainingConfig(FinetuningConfig):
    task: Literal["continued_pretraining"] = "continued_pretraining"


class GRPOConfig(FinetuningConfig):
    task: Literal["grpo"] = "grpo"
    beta: float | None = Field(default=None)
    num_generations: int | None = Field(default=None)
    sampling_params: SamplingParamsConfig | None = Field(default=None)
    reward_fns: RewardFunctionsConfig | None = Field(default=None, serialization_alias="rewardFns")

    @field_validator("reward_fns", mode="before")
    @classmethod
    def convert_callable_list(cls, val: Any) -> Any:
        if not isinstance(val, list):
            return val

        return RewardFunctionsConfig(functions=val)


class RewardFunctionsConfig(BaseModel, validate_assignment=True):
    runtime: RewardFunctionsRuntimeConfig | None = Field(default=None)
    functions: dict[str, RewardFunction] = Field(default_factory=dict)

    @field_validator("functions", mode="before")
    @classmethod
    def convert_callable_list(cls, val: Any) -> Any:
        if not isinstance(val, list):
            return val

        return {f.__name__: RewardFunction.from_callable(f) for f in val}


class RewardFunctionsRuntimeConfig(BaseModel):
    packages: list[str] | None = Field(default=None)


class RewardFunction(BaseModel):
    encoded_fn: str = Field(serialization_alias="encodedFn")

    @classmethod
    def from_callable(cls, fn: Callable) -> RewardFunction:
        fn_name = getattr(fn, "__name__", None)
        if not fn_name:
            raise ValueError(f"Input is not a function or is missing a name")

        if not inspect.isfunction(fn):
            raise ValueError(f"Expected input to be a function")

        params = inspect.signature(fn).parameters
        if len(params) != 3:
            raise ValueError(f"unexpected number of arguments for input - expected 3, "
                             f"found {len(params)}")

        fn_source = inspect.getsource(fn)
        return RewardFunction(encoded_fn=base64.b64encode(fn_source.encode("utf-8")).decode("utf-8"))

    @cached_property
    def source(self):
        return base64.b64decode(self.encoded_fn).decode("utf-8")

    @cached_property
    def function(self):
        ast_root = ast.parse(self.source)

        # Check that the input is a single function definition
        if len(ast_root.body) != 1 or not isinstance(ast_root.body[0], ast.FunctionDef):
            raise ValueError("Expected encoded data to be a Python function")

        # Render the function locally
        exec(compile(ast_root, filename="<string>", mode="exec"))

        return locals()[ast_root.body[0].name]


class UpdateDeploymentConfig(BaseModel):
    # Note: necessary because this particular model and its children are used to both deserialize and serialize data.
    # Other models in this SDK are used exclusively for deserializing data to a Python object that is never (for now)
    # reserialized back to the API for a different request.
    model_config = ConfigDict(populate_by_name=True)

    custom_args: list[str] | None = Field(
        default=None,
        validation_alias="customArgs",
        serialization_alias="customArgs",
    )
    cooldown_time: int | None = Field(
        default=None,
        validation_alias="cooldownTime",
        serialization_alias="cooldownTime",
    )
    hf_token: str | None = Field(
        default=None,
        validation_alias="hfToken",
        serialization_alias="hfToken",
    )
    min_replicas: int | None = Field(
        default=None,
        validation_alias="minReplicas",
        serialization_alias="minReplicas",
    )
    max_replicas: int | None = Field(
        default=None,
        validation_alias="maxReplicas",
        serialization_alias="maxReplicas",
    )
    scale_up_threshold: int | None = Field(
        default=None,
        validation_alias="scaleUpRequestThreshold",
        serialization_alias="scaleUpRequestThreshold",
    )
    max_total_tokens: int | None = Field(
        default=None,
        validation_alias="maxTotalTokens",
        serialization_alias="maxTotalTokens",
    )
    lorax_image_tag: str | None = Field(
        default=None,
        validation_alias="loraxImageTag",
        serialization_alias="loraxImageTag",
    )
    request_logging_enabled: bool | None = Field(
        default=None,
        validation_alias="requestLoggingEnabled",
        serialization_alias="requestLoggingEnabled",
    )
    direct_ingress: bool | None = Field(
        default=None,
        validation_alias="directIngress",
        serialization_alias="directIngress",
    )
    preloaded_adapters: list[str] | None = Field(
        default=None,
        validation_alias="preloadedAdapters",
        serialization_alias="preloadedAdapters",
    )
    speculator: str | None = Field(
        default=None,
        validation_alias="speculator",
        serialization_alias="speculator",
    )
    prefix_caching: bool | None = Field(
        default=None,
        validation_alias="prefixCaching",
        serialization_alias="prefixCaching",
    )


class DeploymentConfig(UpdateDeploymentConfig):
    base_model: str = Field(
        ...,
        validation_alias="baseModel",
        serialization_alias="baseModel",
    )
    accelerator: str | None = Field(default=None)
    uses_guaranteed_capacity: bool | None = Field(
        default=None,
        validation_alias="usesGuaranteedCapacity",
        serialization_alias="usesGuaranteedCapacity",
    )
    quantization: str | None = Field(default=None)


class AugmentationConfig(BaseModel):
    """Configuration for synthetic data generation tasks.

    # Attributes
    :param base_model: (str) The OpenAI model to prompt.
    :param num_samples_to_generate: (int) The number of synthetic examples to generate.
    :param num_seed_samples: (int) The number of seed samples to use for generating synthetic examples.
    :param task_context: (str) The user-provided task context for generating candidates.
    """

    base_model: str
    num_samples_to_generate: int = Field(default=1000)
    num_seed_samples: int | str = Field(default="all")
    augmentation_strategy: str = Field(default="mixture_of_agents")
    task_context: str = Field(default="")

    @field_validator("base_model")
    @classmethod
    def validate_base_model(cls, base_model) -> str:
        supoorted_base_models = {
            "gpt-4-turbo",
            "gpt-4-0125-preview",
            "gpt-4-1106-preview",
            "gpt-4o",
            "gpt-4o-2024-08-06",
            "gpt-4o-mini",
        }
        if base_model not in supoorted_base_models:
            raise ValueError(
                f"base_model must be one of {supoorted_base_models}.",
            )
        return base_model

    @field_validator("num_samples_to_generate")
    @classmethod
    def validate_num_samples_to_generate(cls, num_samples_to_generate) -> int:
        if num_samples_to_generate < 1:
            raise ValueError("num_samples_to_generate must be >= 1.")
        return num_samples_to_generate

    @field_validator("num_seed_samples")
    @classmethod
    def validate_num_seed_samples(cls, num_seed_samples):
        if isinstance(num_seed_samples, str) and num_seed_samples != "all":
            raise ValueError("num_seed_samples can only be an integer or the string 'all'.")
        elif isinstance(num_seed_samples, int) and num_seed_samples < 1:
            raise ValueError("num_seed_samples must be >= 1.")
        return num_seed_samples

    @field_validator("augmentation_strategy")
    @classmethod
    def validate_augmentation_strategy(cls, augmentation_strategy):
        if augmentation_strategy not in {"single_pass", "mixture_of_agents"}:
            raise ValueError("augmentation_strategy must be 'single_pass' or 'mixture_of_agents'.")
        return augmentation_strategy
