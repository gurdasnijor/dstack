from typing import Annotated, Any, Literal, Optional, Union

from pydantic import Field, PositiveInt, root_validator, validator

from dstack._internal.core.models.common import (
    CoreModel,
    EntityReference,
    generate_dual_core_model,
)
from dstack._internal.core.models.envs import Env
from dstack._internal.core.models.profiles import ProfileParams, ProfileParamsConfig
from dstack._internal.utils.json_schema import add_extra_schema_types


def _validate_model_label(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Endpoint model metadata must be a non-empty string")
    return value


def _validate_optional_model_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    return _validate_model_label(value)


class EndpointModelRepo(CoreModel):
    repo: Annotated[
        Optional[str],
        Field(description="Legacy spelling for the exact model locator to deploy"),
    ] = None
    locator: Annotated[
        Optional[str],
        Field(
            description=(
                "The exact model locator to deploy, such as a Hugging Face ID, URL, "
                "registry reference, object store URI, or path"
            )
        ),
    ] = None
    name: Annotated[
        Optional[str], Field(description="The client-facing model name. Defaults to the locator")
    ] = None
    source: Annotated[
        str,
        Field(
            description=(
                "Model source type such as `huggingface`, `url`, `path`, or `custom`. "
                "Defaults to `auto` detection"
            )
        ),
    ] = "auto"
    revision: Annotated[
        Optional[str],
        Field(description="Requested model revision, commit, version, or immutable digest"),
    ] = None
    modality: Annotated[
        str,
        Field(
            description=(
                "Requested modality such as `text-generation`, `image-generation`, "
                "`video-generation`, or `auto`"
            )
        ),
    ] = "auto"

    @property
    def api_model_name(self) -> str:
        return self.name or self.exact_repo

    @property
    def exact_repo(self) -> str:
        return self.locator or self.repo or ""

    @property
    def allows_variant_selection(self) -> bool:
        return False

    @property
    def source_type(self) -> str:
        return self.source

    @property
    def requested_revision(self) -> Optional[str]:
        return self.revision

    @property
    def requested_modality(self) -> str:
        return self.modality

    @root_validator
    def validate_locator(cls, values: dict) -> dict:
        locators = [value for field in ("repo", "locator") if (value := values.get(field))]
        if len(locators) != 1:
            raise ValueError("Endpoint model must specify exactly one of repo or locator")
        return values

    @validator("repo", "locator")
    def validate_repo(cls, value: Optional[str], field) -> Optional[str]:
        if value is None:
            return None
        return _validate_model(value, field=field.name)

    @validator("name")
    def validate_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return _validate_model(value, field="name")

    _validate_source = validator("source", allow_reuse=True)(_validate_model_label)
    _validate_modality = validator("modality", allow_reuse=True)(_validate_model_label)
    _validate_revision = validator("revision", allow_reuse=True)(_validate_optional_model_label)


class EndpointModelBase(CoreModel):
    base: Annotated[
        str,
        Field(description="The base model for which the agent may select a compatible variant"),
    ]
    source: Annotated[
        str,
        Field(description="Model source type. Defaults to `auto` detection"),
    ] = "auto"
    revision: Annotated[
        Optional[str],
        Field(description="Requested base-model revision, commit, version, or digest"),
    ] = None
    modality: Annotated[
        str,
        Field(description="Requested modality. Defaults to `auto` detection"),
    ] = "auto"

    @property
    def api_model_name(self) -> str:
        return self.base

    @property
    def exact_repo(self) -> None:
        return None

    @property
    def allows_variant_selection(self) -> bool:
        return True

    @property
    def source_type(self) -> str:
        return self.source

    @property
    def requested_revision(self) -> Optional[str]:
        return self.revision

    @property
    def requested_modality(self) -> str:
        return self.modality

    @validator("base")
    def validate_base(cls, value: str) -> str:
        return _validate_model(value, field="base")

    _validate_source = validator("source", allow_reuse=True)(_validate_model_label)
    _validate_modality = validator("modality", allow_reuse=True)(_validate_model_label)
    _validate_revision = validator("revision", allow_reuse=True)(_validate_optional_model_label)


EndpointModelSpec = Union[EndpointModelRepo, EndpointModelBase]


class EndpointConfigurationConfig(ProfileParamsConfig):
    @staticmethod
    def schema_extra(schema: dict[str, Any]):
        ProfileParamsConfig.schema_extra(schema)
        add_extra_schema_types(
            schema["properties"]["model"],
            extra_types=[{"type": "string"}],
        )


class EndpointConfiguration(
    ProfileParams,
    generate_dual_core_model(EndpointConfigurationConfig),
):
    type: Annotated[Literal["endpoint"], Field(description="The configuration type")] = "endpoint"
    name: Annotated[
        Optional[str],
        Field(description="The endpoint name. Required unless passed with `--name`"),
    ] = None
    model: Annotated[
        EndpointModelSpec,
        Field(
            description=(
                "The model to serve. Use a string, `locator`, or legacy `repo` for an exact model, "
                "or `base` to allow compatible model variants."
            )
        ),
    ]
    context_length: Annotated[
        Optional[PositiveInt], Field(description="The minimum required context length")
    ] = None
    preset: Annotated[
        Optional[str], Field(description="The preset ID to use when applying the endpoint")
    ] = None
    gateway: Annotated[
        Optional[Union[bool, EntityReference, str]],
        Field(
            description=(
                "The name of the gateway. Specify boolean `false` to run without a gateway."
                " Specify boolean `true` to run with the default gateway."
                " Omit to run with the default gateway if there is one, or without a gateway otherwise"
            )
        ),
    ] = None
    env: Annotated[Env, Field(description="The mapping or the list of environment variables")] = (
        Env()
    )

    @validator("model", pre=True)
    def parse_model(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"repo": _validate_model(value, field="model")}
        return value

    @validator("preset")
    def validate_preset(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("Endpoint preset must be a non-empty string")
        return value


def _validate_model(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Endpoint model {field} must be a non-empty string")
    return value
