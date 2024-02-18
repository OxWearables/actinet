# actinet

A tool to extract meaningful health information from large accelerometer datasets.
The software generates time-series and summary metrics useful for answering key questions such as how much time is spent in sleep, sedentary behaviour, or doing physical activity.
The backbone of this repository is a self-supervised Resnet18 model.

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
$ actinet -f sample.cwa

# Or an ActiGraph file
$ actinet -f sample.gt3x

# Or a GENEActiv file
$ actinet -f sample.bin

# Or a CSV file (see data format below)
$ actinet -f sample.csv
```

### Troubleshooting

Some systems may face issues with Java when running the script. If this is your case, try fixing OpenJDK to version 8:

```console
conda create -n actinet openjdk=8
```

### Offline usage

To use this package offline, one must first download and install the relevant classifier file and model modules.
This repository offers two ways of doing this.

Run the following code when you have internet access:
```console
actinet --cache-classifier
```
 
Following this, the actinet classifier can be used as standard without internet access, without needing to specify the flags relating to the model repository.
 
Alternatively, you can download or git clone the ssl modules from the [ssl-wearables repository](https://github.com/OxWearables/ssl-wearables).

In addition, you can donwload/prepare a custom classifier file.

Once this is downloaded to an appopriate location, you can run the actinet model using:
 
```console
actinet -f sample.cwa -c /path/to/classifier.joblib.lzma -m /path/to/ssl-wearables
```

### Output files

By default, output files will be stored in a folder named after the input file, `outputs/{filename}/`, created in the current working directory. You can change the output path with the `-o` flag:

```console
$ actinet -f sample.cwa -o /path/to/some/folder/

<Output summary written to: /path/to/some/folder/sample-outputSummary.json>
<Time series output written to: /path/to/some/folder/sample-timeSeries.csv.gz>
```

The following output files are created:

- *Info.json* Summary info, as shown above.
- *timeSeries.csv* Raw time-series of activity levels

See [Data Dictionary](https://biobankaccanalysis.readthedocs.io/en/latest/datadict.html) for the list of output variables.

### Plotting activity profiles

To plot the activity profiles, you can use the -p flag:

```console
$ actinet -f sample.cwa -p
<Output plot written to: data/sample-timeSeries-plot.png>
```

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
actinet -f file1.cwa &
actinet -f file2.cwa &
actinet -f file3.cwa 
:END
````

Once this file is created, run `cmd < commands.txt` from the terminal.

#### Linux

Create a file *command.sh* with:

```console
actinet -f file1.cwa
actinet -f file2.cwa
actinet -f file3.cwa
```

Then, run `bash command.sh` from the terminal.

### Collating outputs

A utility script is provided to collate outputs from multiple runs:

```console
actinet-collate-outputs outputs/
```

This will collate all *-Info.json files found in outputs/ and generate a CSV file.

## Citing our work

When using this tool, please consider citing the works listed in [CITATION.md](https://github.com/OxWearables/actinet/blob/main/CITATION.md).

## Licence

See [LICENSE.md](https://github.com/OxWearables/actinet/blob/main/LICENSE.md).

## Acknowledgements

We would like to thank all our code contributors, manuscript co-authors, and research participants for their help in making this work possible.
