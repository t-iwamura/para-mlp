from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="para_mlp",
    version="1.4.1",
    author="Taiki Iwamura",
    author_email="takki.0206@gmail.com",
    description="paramagnetic machine learning potential package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iwamura-lab/para-mlp",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    include_package_data=True,
    python_requires=">= 3.8",
    install_requires=[
        "numpy",
        "scikit-learn==1.1.2",
        "joblib",
        "tqdm",
        "dataclasses_json",
        "click",
        "pymatgen",
        "mlp_build_tools",
        "mlpcpp",
    ],
    entry_points={
        "console_scripts": [
            "para-mlp=para_mlp.scripts.main:main",
            "spin-average=para_mlp.scripts.calc_spin_average:main",
        ],
    },
)
