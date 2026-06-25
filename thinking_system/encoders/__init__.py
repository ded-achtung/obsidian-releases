"""Сенсорный ингест: obs → латент."""

from thinking_system.encoders.base import Encoder
from thinking_system.encoders.random_projection import RandomProjectionEncoder

__all__ = ["Encoder", "RandomProjectionEncoder"]
