from setuptools import setup, find_packages
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="cyclereward",
    version="0.1.3",
    author="Hyojin Bahng",
    description="Image-text alignment metric trained on cycle consistency preferences",
    packages=find_packages(), 
    include_package_data=True,
    package_data={
        "cyclereward.blip": ["med_config.json"],
    },
    install_requires=[
        "sentencepiece",
        "six",
        "sniffio",
        "submitit",
        "sympy",
        "tenacity",
        "tensorboardX",
        "threadpoolctl",
        "tiktoken",
        "timm",
        "tokenizers>=0.21.1",
        "torch>=2.6.0",
        "torchvision>=0.21.0",
        "tqdm",
        "transformers",
        "triton",
        "trl",
        "fairscale"
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hjbahng/cyclereward",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    license="MIT"
)