from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="moralclip",
    version="0.1.0",
    author="Ana Carolina Condez, Diogo Tavares, João Magalhães",
    author_email="a.condez@campus.fct.unl.pt",
    description="Contrastive Alignment of Vision-and-Language Representations with Moral Foundations Theory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/moralclip",  # Update with your actual repo
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "moralclip-train=scripts.train:main",
            "moralclip-inference=scripts.inference:main",
        ],
    },
)
