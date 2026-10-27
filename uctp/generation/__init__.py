"""Synthetic instances for the manuscript's curricular demand model."""

from .instance_generator import generate_instance
from .parameters import DatasetGroup, GenerationOptions

__all__ = ["DatasetGroup", "GenerationOptions", "generate_instance"]
