# d:\path\to\your\psyModuleToolbox\setup.py
from setuptools import setup, find_packages

setup(
    name='psyModuleToolbox',
    version='0.1.0',  # Or your desired version
    packages=find_packages(),
    description='A toolbox for psychophysiological data processing and analysis.',
    author='Your Name',
    author_email='your.email@example.com',
    # Add other dependencies here if your toolbox itself has them
    # install_requires=[
    #     'numpy',
    #     'pandas',
    #     'mne',
    #     'neurokit2', # etc.
    # ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # Or your chosen license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8', # Or your minimum Python version
)