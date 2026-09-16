"""Small explicit encoder registry; registration never imports upstream packages."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from stageworld.errors import ConfigurationError

EncoderFactory = Callable[..., Any]


class EncoderRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, EncoderFactory] = {}

    def register(self, name: str, factory: EncoderFactory) -> None:
        key = name.strip().lower()
        if not key:
            raise ConfigurationError(
                code="INVALID_ENCODER_NAME", message="Encoder name cannot be empty."
            )
        if key in self._factories:
            raise ConfigurationError(
                code="DUPLICATE_ENCODER", message=f"Encoder '{key}' is already registered."
            )
        self._factories[key] = factory

    def create(self, name: str, **kwargs: Any) -> Any:
        key = name.strip().lower()
        try:
            factory = self._factories[key]
        except KeyError as error:
            raise ConfigurationError(
                code="UNKNOWN_ENCODER",
                message=f"Unknown encoder '{key}'.",
                details={"available": self.names()},
            ) from error
        return factory(**kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def default_encoder_registry() -> EncoderRegistry:
    """Build the project registry without constructing or importing upstream backends."""

    from .medical import (
        MerlinEncoder,
        PRISM2Encoder,
        SwinUNETREncoder,
        TITANCONCHEncoder,
        UNI2HEncoder,
    )
    from .synthetic import SyntheticEncoder

    registry = EncoderRegistry()
    registry.register("synthetic", SyntheticEncoder)
    registry.register("merlin", MerlinEncoder)
    registry.register("swinunetr", SwinUNETREncoder)
    registry.register("titan_conch_v1_5", TITANCONCHEncoder)
    registry.register("uni2_h", UNI2HEncoder)
    registry.register("prism2_base", PRISM2Encoder)
    return registry
