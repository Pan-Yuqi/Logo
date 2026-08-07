# -*- coding: utf-8 -*-
"""Packaging for the LoGo model.

Install (editable) from the project root:

    cd ./LoGo && pip install -e .

Then import from anywhere:

    from logo import LoGoConfig, LoGoForCausalLM
"""

import os

from setuptools import find_packages, setup

this_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(this_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="logo-attn",
    version="0.1.0",
    description="LoGo: Token-Level Dynamic Local-Global Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("tests", "configs")),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.44.0,<4.52.0",
        "triton>=3.0.0",
        "einops>=0.7.0",
    ],
    extras_require={
        "flash": ["flash-attn>=2.1.0"],
    },
    python_requires=">=3.9",
)
