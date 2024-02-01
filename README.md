# actinet

A tool to extract meaningful health information from large accelerometer datasets. 
The software generates time-series and summary metrics useful for answering key questions such as how much time is spent in sleep, sedentary behaviour, or doing physical activity.
The backbone of this repository is a self-supervised Resnet18 model.

## Install

*Minimum requirements*: Python>=3.8, Java 8 (1.8)

The following instructions make use of Anaconda to meet the minimum requirements:

1. Download & install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (light-weight version of Anaconda).
1. (Windows) Once installed, launch the **Anaconda Prompt**.
1. Create a virtual environment:
    ```console
    $ conda create -n actinet python=3.9 openjdk pip
    ```
    This creates a virtual environment called `actinet` with Python version 3.9, OpenJDK, and Pip.
1. Activate the environment:
    ```console
    $ conda activate actinet
    ```
    You should now see `(actinet)` written in front of your prompt.
1. Install `actinet`:
    ```console
    $ pip install actinet
    ```

You are all set! The next time that you want to use `actinet`, open the Anaconda Prompt and activate the environment (step 4). If you see `(actinet)` in front of your prompt, you are ready to go!

## Usage

```bash
# Process an AX3 file
$ actinet sample.cwa

# Or an ActiGraph file
$ actinet sample.gt3x

# Or a GENEActiv file
$ actinet sample.bin

# Or a CSV file (see data format below)
$ actinet sample.csv
```


### Troubleshooting 
Some systems may face issues with Java when running the script. If this is your case, try fixing OpenJDK to version 8:
```console
$ conda install -n actinet openjdk=8
```

### Output files
By default, output files will be stored in a folder named after the input file, `outputs/{filename}/`, created in the current working directory. You can change the output path with the `-o` flag:

```console
$ actinet sample.cwa -o /path/to/some/folder/
```

The following output files are created:

- *Info.json* Summary info, as shown above.
- *timeSeries.csv* Raw time-series of activity levels

See [Data Dictionary](https://biobankaccanalysis.readthedocs.io/en/latest/datadict.html) for the list of output variables.


### Crude vs. Adjusted Estimates
Adjusted estimates are provided that account for missing data.
Missing values in the time-series are imputed with the mean of the same timepoint of other available days.
For adjusted totals and daily statistics, 24h multiples are needed and will be imputed if necessary.
Estimates will be NaN where data is still missing after imputation.


### Processing CSV files
If a CSV file is provided, it must have the following header: `time`, `x`, `y`, `z`. 

Example:
```console
time,x,y,z
2013-10-21 10:00:08.000,-0.078923,0.396706,0.917759
2013-10-21 10:00:08.010,-0.094370,0.381479,0.933580
2013-10-21 10:00:08.020,-0.094370,0.366252,0.901938
2013-10-21 10:00:08.030,-0.078923,0.411933,0.901938
...
```

### Processing multiple files
#### Windows
To process multiple files you can create a text file in Notepad which includes one line for each file you wish to process, as shown below for *file1.cwa*, *file2.cwa*, and *file2.cwa*.

Example text file *commands.txt*: 
```console
actinet file1.cwa &
actinet file2.cwa &
actinet file3.cwa 
:END
````
Once this file is created, run `cmd < commands.txt` from the terminal.

#### Linux
Create a file *command.sh* with:
```console
actinet file1.cwa
actinet file2.cwa
actinet file3.cwa
```
Then, run `bash command.sh` from the terminal.

#### Collating outputs

A utility script is provided to collate outputs from multiple runs:

```console
$ actinet-collate-outputs outputs/
```
This will collate all *-Info.json files found in outputs/ and generate a CSV file.

## Citing our work

When using this tool, please consider citing the works listed in [CITATION.md](https://github.com/OxWearables/actinet/blob/main/CITATION.md).


## Licence
See [LICENSE.md](https://github.com/OxWearables/actinet/blob/main/LICENSE.md).


## Acknowledgements
We would like to thank all our code contributors, manuscript co-authors, and research participants for their help in making this work possible.

# Sample PyPI package + GitHub Actions + Versioneer

This template aims to automate the tedious and error-prone steps of tagging/versioning, building and publishing new package versions. This is achieved by syncing git tags and versions with Versioneer, and automating the build and release with GitHub Actions, so that publishing a new version is as painless as:

```console
$ git tag vX.Y.Z && git push --tags
```

The following guide assumes familiarity with `setuptools` and PyPI. For an introduction to Python packaging, see the references at the bottom.

## How to use this template

1. Click on the *Use this template* button to get a copy of this repository.
1. Rename *src/sample_package* folder to your package name &mdash; *src/* is where your package must reside.
1. Go through each of the following files and rename all instances of *sample-package* or *sample_package* to your package name. Also update the package information such as author names, URLs, etc.
    1. setup.py
    1. pyproject.toml
    1. \_\_init\_\_.py
1. Install `versioneer` and `tomli`, and run `versioneer`:

    ```console
    $ pip install tomli
    $ pip install versioneer
    $ versioneer install
    ```
    Then *commit* the changes produced by `versioneer`. See [here](https://github.com/python-versioneer/python-versioneer/blob/master/INSTALL.md) to learn more.
1. Setup your PyPI credentials. See the section *Saving credentials on Github* of [this guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/). You should use the variable names `TEST_PYPI_API_TOKEN` and `PYPI_API_TOKEN` for the TestPyPI and PyPI tokens, respectively. See *.github/workflows/release.yaml*.

You are all set! It should now be possible to run `git tag vX.Y.Z && git push --tags` to automatically version, build and publish a new release to PyPI.

Finally, it is a good idea to [configure tag protection rules](https://docs.github.com/en/enterprise-server@3.8/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/configuring-tag-protection-rules) in your repository.

## References
- Python packaging guide: https://packaging.python.org/en/latest/tutorials/packaging-projects/
    - ...and how things are changing: https://snarky.ca/what-the-heck-is-pyproject-toml/ &mdash; in particular, note that while *pyproject.toml* seems to be the future, currently Versioneer still depends on *setup.py*.
- Versioneer: https://github.com/python-versioneer/python-versioneer
- GitHub Actions: https://docs.github.com/en/actions
