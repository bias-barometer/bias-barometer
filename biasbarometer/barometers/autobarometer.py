"""Borrowed logic inspiration from https://github.com/bigscience-workshop/evaluation/blob/main/evaluation/tasks/auto_task.py"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict
from dataclasses import fields, MISSING

from biasbarometer.models import Model
from biasbarometer.config import BarometerConfig


class AutoBarometer(ABC):
    def __init__(self, config: Dict) -> None:
        self.results = {}
        self.config = config

    @classmethod
    def _get_barometer(cls, barometer_name: str) -> AutoBarometer:
        all_barometers = cls.__subclasses__()
        for barometer in all_barometers:
            if barometer.get_display_name() == barometer_name:
                return barometer
        raise ValueError(f"Invalid barometer: {barometer_name}")

    @classmethod
    def from_config(
        cls,
        config: BarometerConfig,
    ) -> AutoBarometer:
        barometer = cls._get_barometer(config.method)
        return barometer(config=vars(config))

    @classmethod
    def from_spec(
        cls,
        barometer_name: str,
        **kwargs,
    ) -> AutoBarometer:
        barometer = cls._get_barometer(barometer_name)
        config_class = barometer.get_config_class()

        # Check if all required arguments are provided
        required_arguments = [
            field.name for field in fields(config_class) if field.default == MISSING
        ]
        missing_arguments = set(required_arguments).difference(set(kwargs))
        if missing_arguments:
            raise TypeError(
                f"The following keyword arguments are missing for this barometer: {', '.join(missing_arguments)}"
            )
        return barometer(config=vars(config_class(**kwargs)))

    @staticmethod
    @abstractmethod
    def get_display_name() -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_config_class() -> type[BarometerConfig]:
        """Returns the class specifying the config for that barometer type."""
        pass

    @abstractmethod
    def evaluate(self, model: Model) -> None:
        """Runs evaluations and writes results to self.results."""
        pass
