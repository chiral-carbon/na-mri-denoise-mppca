from setuptools import setup, find_packages

setup(
    name="na-mri-denoise-mppca",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.6.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.50.0",
        "seaborn>=0.12.0",
        "pandas>=1.5.0",
        "scikit-image>=0.19.0",
        "pathlib>=1.0.1"
    ],
    description="MP-PCA denoising algorithms for Sodium MRI, featuring the improved MP-PCA implementation",
    author="NYU Langone",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 