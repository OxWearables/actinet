import sys
import os.path

# https://github.com/python-versioneer/python-versioneer/issues/193
sys.path.insert(0, os.path.dirname(__file__))

from setuptools import setup, find_packages
import codecs

import versioneer


def main():

    with open("README.md", "r") as fh:
        long_description = fh.read()

    setup(
        name="ActiNet",
        version=versioneer.get_version(),  # Do not edit
        cmdclass=versioneer.get_cmdclass(),  # Do not edit
        python_requires=">=3.8, <4",
        description="Activity detection algorithm compatible with the UK Biobank Accelerometer Dataset",
        long_description=long_description,  # Do not edit. See README.md
        long_description_content_type="text/markdown",
        keywords="example, setuptools, versioneer",
        url="https://github.com/OxWearables/actinet",
        download_url="https://github.com/OxWearables/actinet",
        author=get_string("__author__"),  # Do not edit. see src/actinet/__init__.py
        maintainer=get_string(
            "__maintainer__"
        ),  # Do not edit. see src/actinet/__init__.py
        maintainer_email=get_string(
            "__maintainer_email__"
        ),  # Do not edit. See src/actinet/__init__.py
        license=get_string("__license__"),  # Do not edit. See src/actinet/__init__.py
        # This is for PyPI to categorize your project. See: https://pypi.org/classifiers/
        classifiers=[
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Bio-Informatics",
            "Topic :: Scientific/Engineering :: Information Analysis",
            "Topic :: Scientific/Engineering :: Medical Science Apps.",
        ],
        # Where to find the source code
        packages=find_packages(where="src", exclude=("test", "tests")),
        package_dir={"": "src"},
        # What other files to include, e.g. *.class if the package uses some Java code.
        package_data={"actinet": ["*.txt", "*.rst", "*.md"]},
        include_package_data=False,
        # Dependencies
        install_requires=[
            "actipy>=3.0.5",
            "numpy==1.24.*",
            "scipy==1.10.*",
            "pandas==2.0.*",
            "tqdm==4.64.*",
            "matplotlib==3.5.*",
            "joblib==1.2.*",
            "scikit-learn==1.1.1",
            "imbalanced-learn==0.9.1",
            "torch==1.13.*",
            "torchvision==0.14.*",
            "transforms3d==0.4.*",
        ],
        extras_require={
            "dev": [
                "versioneer",
                "twine",
                "ipython",
                "ipdb",
                "flake8",
                "autopep8",
                "tomli",
                "jupyter",
            ],
            "docs": [
                "sphinx>=4.2",
                "sphinx_rtd_theme>=1.0",
                "readthedocs-sphinx-search>=0.1",
                "docutils<0.18",
            ],
        },
        # Define entry points for command-line scripts, e.g.: `$ hello --name Alice`
        entry_points={
            "console_scripts": [
                "actinet=actinet.actinet:main",
                "actinet-collate-outputs=actinet.utils.collate_outputs:main",
            ],
        },
    )


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_string(string, rel_path="src/actinet/__init__.py"):
    for line in read(rel_path).splitlines():
        if line.startswith(string):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError(f"Unable to find {string}.")


if __name__ == "__main__":
    main()
