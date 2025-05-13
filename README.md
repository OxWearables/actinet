# ActiNet

[![GitHub all releases](https://img.shields.io/github/release/OxWearables/actinet.svg)](https://github.com/OxWearables/actinet/releases/)
[![DOI](https://zenodo.org/badge/751360921.svg)](https://doi.org/10.5281/zenodo.15310683)

A tool to extract meaningful health information from large accelerometer datasets.
The software generates time-series and summary metrics useful for answering key questions such as how much time is spent in sleep, sedentary behaviour, or doing physical activity.

## Install

*Minimum requirements*: Python>=3.9, Java 8 (1.8)

The following instructions make use of Anaconda to meet the minimum requirements:

1. Download & install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (light-weight version of Anaconda).
1. (Windows) Once installed, launch the **Anaconda Prompt**.
1. Create a virtual environment:

    ```console
    conda create -n actinet python=3.9 openjdk pip
    ```

    This creates a virtual environment called `actinet` with Python version 3.9, OpenJDK, and Pip.
1. Activate the environment:

    ```console
    conda activate actinet
    ```

    You should now see `(actinet)` written in front of your prompt.
1. Install `actinet`:

    ```console
    pip install actinet
    ```

You are all set! The next time that you want to use `actinet`, open the Anaconda Prompt and activate the environment (step 4). If you see `(actinet)` in front of your prompt, you are ready to go!

## Usage

```bash
# Process an AX3 file
$ actinet sample.cwa.gz

# Or an ActiGraph file
$ actinet sample.gt3x

# Or a GENEActiv file
$ actinet sample.bin

# Or a CSV file (see data format below)
$ actinet sample.csv
```

See the [Usage](https://actinet.readthedocs.io/en/latest/usage.html) page for further uses of the tool.

### Troubleshooting

Some systems may face issues with Java when running the script. If this is your case, try fixing OpenJDK to version 8:

```console
conda create -n actinet openjdk=8
```

### Output files

By default, output files will be stored in a folder named after the input file, `outputs/{filename}/`, created in the current working directory.
You can change the output path with the `-o` flag:

```console
$ actinet sample.cwa -o /path/to/some/folder/

<Output summary written to: /path/to/some/folder/sample-outputSummary.json>
<Time series output written to: /path/to/some/folder/sample-timeSeries.csv.gz>
```

The following output files are created:

- *Info.json* Summary info, as shown above.
- *timeSeries.csv* Raw time-series of activity levels

See [Data Dictionary](https://actinet.readthedocs.io/en/latest/datadict.html) for the list of output variables.

### Plotting activity profiles

To plot the activity profiles, you can use the -p flag:

```console
$ actinet sample.cwa -p
<Output plot written to: data/sample-timeSeries-plot.png>
```

### Crude vs. Adjusted Estimates

Adjusted estimates are provided that account for missing data.
Missing values in the time-series are imputed with the mean of the same timepoint of other available days.
For adjusted totals and daily statistics, 24h multiples are needed and will be imputed if necessary.
Estimates will be NaN where data is still missing after imputation.

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

### Collating outputs

A utility script is provided to collate outputs from multiple runs:

```console
actinet-collate-outputs outputs/
```

This will collate all *-Info.json files found in outputs/ and generate a CSV file.

## Citing our work

When using this tool, please consider citing the works listed in [CITATION.md](https://github.com/OxWearables/actinet/blob/master/CITATION.md).

## Licence

See [LICENSE.md](https://github.com/OxWearables/actinet/blob/master/LICENSE.md).

## Acknowledgements

We would like to thank all our code contributors, manuscript co-authors, and research participants for their help in making this work possible.
