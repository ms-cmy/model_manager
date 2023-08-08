from setuptools import setup, find_packages

setup(
    name='sklearn_pipeline_manager',
    version='0.0.1',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
)