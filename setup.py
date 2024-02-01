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
        name="sample-package-chanshing",
        version=versioneer.get_version(),    # Do not edit
        cmdclass=versioneer.get_cmdclass(),  # Do not edit
        python_requires=">=3.8, <4",
        description="An example Python project",
        long_description=long_description,  # Do not edit. See README.md
        long_description_content_type="text/markdown",
        keywords="example, setuptools, versioneer",
        url="https://github.com/chanshing/sample-package",
        download_url="https://github.com/chanshing/sample-package",
        author=get_string("__author__"),                      # Do not edit. see src/sample_package/__init__.py
        maintainer=get_string("__maintainer__"),              # Do not edit. see src/sample_package/__init__.py
        maintainer_email=get_string("__maintainer_email__"),  # Do not edit. See src/sample_package/__init__.py
        license=get_string("__license__"),                    # Do not edit. See src/sample_package/__init__.py

        # This is for PyPI to categorize your project. See: https://pypi.org/classifiers/
        classifiers=[
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering",
        ],

        # Where to find the source code
        packages=find_packages(where="src", exclude=("test", "tests")),
        package_dir={"": "src"},
        # What other files to include, e.g. *.class if the package uses some Java code.
        package_data={"sample_package": ["*.txt", "*.rst", "*.md"]},

        # This option will include all files in the `src/sample_package` directory provided they
        # are listed in the `MANIFEST.in` file, OR are being tracked by git.
        # See: https://setuptools.pypa.io/en/latest/userguide/datafiles.html
        # include_package_data=True,

        # Dependencies
        install_requires=[],

        # Optional packages. Can be installed with:
        # `$ pip install sample-package-chanshing[dev]` or
        # `$ pip install sample-package-chanshing[docs]` or
        # `$ pip install sample-package-chanshing[dev,docs]`
        extras_require={
            # Will be installed with `$ pip install sample-package-chanshing[dev]`
            "dev": [
                "versioneer",
                "twine",
                "ipdb",
                "flake8",
                "autopep8",
            ],
            # Will be installed with `$ pip install sample-package-chanshing[docs]`
            "docs": [
                "sphinx>=4.2",
                "sphinx_rtd_theme>=1.0",
                "readthedocs-sphinx-search>=0.1",
                "docutils<0.18",
            ]
        },

        # Define entry points for command-line scripts, e.g.: `$ hello --name Alice`
        entry_points={
            "console_scripts": [
                "hello=sample_package.main:main",
            ],
        },

    )


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_string(string, rel_path="src/sample_package/__init__.py"):
    for line in read(rel_path).splitlines():
        if line.startswith(string):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError(f"Unable to find {string}.")



if __name__ == "__main__":
    main()
