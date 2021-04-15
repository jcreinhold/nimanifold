#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'bokeh',
    'colorcet',
    'datashader',
    'holoviews',
    'matplotlib',
    'nibabel',
    'numba',
    'numpy',
    'pandas',
    'pynndescent',
    'scikit-image',
    'scikit-learn',
    'seaborn',
    'umap-learn'
]

setup_requirements = []

test_requirements = []

setup(
    author="Jacob C Reinhold",
    author_email='jcreinhold@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="nonlinear dimensionality reduction exploration for medical image datasets",
    entry_points={
        'console_scripts': [
            'nimanifold=nimanifold.cli:main',
        ],
    },
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='nimanifold',
    name='nimanifold',
    packages=find_packages(include=['nimanifold', 'nimanifold.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/jcreinhold/nimanifold',
    version='0.1.0',
    zip_safe=False,
)
