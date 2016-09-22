from setuptools import setup, find_packages

setup(
    name='prada_bayes_opt',
    version='0.1',
    packages = find_packages(),
    include_package_data = True,
    description='PradaBayesianOptimization package',
    install_requires=[
        "numpy >= 1.9.0",
        "scipy >= 0.14.0",
        "scikit-learn >= 0.16.1",
    ],
)
