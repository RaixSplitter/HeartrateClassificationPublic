from setuptools import setup, find_packages

setup(
    name='bachelors_thesis',
    version='1.0',
    author='Markus Ngo',
    author_email='fugleboefstudies@gmail.com',
    description='A short description of your package',
    packages=find_packages(),
    install_requires=[
        # Add any dependencies your package requires
        "torch",
        "numpy",
        "scikit-learn",
        "pandas",
        "matplotlib",
        "seaborn",
        "dvc",
        "dvc-gdrive",
        "pyedflib",
        "opencv-python",
        "jupyter",
        "ipywidgets ",
        "widgetsnbextension ",
        "pandas-profiling",
        "sleepecg",
        "wandb",
        "hydra-core",
        "mne",
        "python-dotenv"
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
)