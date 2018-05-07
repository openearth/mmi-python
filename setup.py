from setuptools import setup, find_packages
import os

# get py3 compatible open
from io import open

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the relevant file
with open('DESCRIPTION.rst', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="mmi",
    version='0.5.0',
    description="Model Message Interface",
    long_description=long_description,

    # The project URL.
    url='http://github.com/openearth/python-mmi',

    # Author details
    author='Fedor Baart',
    author_email='fedor.baart@deltares.nl',

    # Choose your license
    license='GPLv3+',

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'Topic :: System :: Distributed Computing',

        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    # What does your project relate to?
    keywords='array messages model',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages.
    packages=find_packages(exclude=["contrib", "docs", "tests*"]),
    # If there are data files included in your packages, specify them here.
    package_data={
        # 'sample': ['*.dat'],
    },
    install_requires=[
        'numpy',
        'pyzmq',
        'bmi-python',
        'tornado',
        'requests',
        'six',
        'shapely',
        'click'
    ],
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'mmi=mmi.cli:cli',
            'mmi-runner=mmi.runner_legacy:main',
            'mmi-curl=mmi.curl:main',
            'mmi-tracker=mmi.tracker:main'
        ],
    }
)
