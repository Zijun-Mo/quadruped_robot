"""Packaging metadata for the rl_base Python package."""

from setuptools import find_packages, setup


setup(
    name="rl-base-lib",
    version="2.3.3",
    description="Fast and simple RL algorithms implemented in PyTorch",
    packages=find_packages(),
    install_requires=["torch>=1.10.0", "torchvision>=0.5.0", "numpy>=1.16.4", "GitPython", "onnx"],
    python_requires=">=3.8",
)
