"""
Neural network models for synthetic data generation.

This module contains implementations of various GAN architectures
optimized for tabular data with fairness and privacy constraints.
"""

from .generator import Generator, ConditionalGenerator
from .discriminator import Discriminator, ConditionalDiscriminator
from .wgan_gp import WGAN_GP
from .cgan import ConditionalGAN

__all__ = [
    "Generator",
    "ConditionalGenerator", 
    "Discriminator",
    "ConditionalDiscriminator",
    "WGAN_GP",
    "ConditionalGAN"
]
