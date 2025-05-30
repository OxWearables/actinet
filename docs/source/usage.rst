Usage
#####

Our tool uses published methods to extract summary sleep and activity statistics from accelerometer data.

Basic usage
===========
To extract a summary of movement metrics from raw accelerometer files:

.. code-block:: console

    $ actinet data/sample.cwa.gz
    <summary output written to outputs/sample/sample-summary.json>
    <time-series output written to outputs/sample/sample-timeSeries.csv.gz>

See :doc:`cliapi` for more details.

This will output a number of files, described in the table below. 
The sample-timeSeries-plot.png file is optional, by adding the '-p' flag,
to visualise the output time-series.

+---------------------+--------------------------------------------------------+
| File                | Description                                            |
+=====================+========================================================+
| outputSummary.json  | Summary statistics for the entire input file, such as  |
|                     | data quality, acceleration and non-wear time grouped   |
|                     | hour of day, and histograms of acceleration levels.    |
+---------------------+--------------------------------------------------------+
| timeSeries.csv.gz   | Acceleration time-series and predicted activities.     |
+---------------------+--------------------------------------------------------+
| timeSeries-plot.png | Output plot of overall activity and class predictions  |
|                     | for each 30sec time window.                            |
+---------------------+--------------------------------------------------------+

Processing a CSV file
---------------------

.. code-block:: console

    $ actinet data/sample.csv.gz

The CSV file must have at least four columns, expected to be named "time", "x", "y" and "z".
The "time" column should contain the date and time of each measurement as a string.
The "x", "y" and "z" columns should contain the numeric tri-axial acceleration values.
A template can be downloaded as follows:

.. code-block:: console

    $ wget "http://gas.ndph.ox.ac.uk/aidend/accModels/sample-small.csv.gz"
    $ mv sample-small.csv.gz data/
    $ gunzip data/sample.csv.gz
    $ head -3 data/sample.csv
    time,x,y,z
    2014-05-07 13:29:50.439+0100 [Europe/London],-0.514,0.07,1.671
    2014-05-07 13:29:50.449+0100 [Europe/London],-0.089,-0.805,-0.59

If the CSV file has a different header, use the option --txyz to specify the time and x-y-z columns, in that order.
The --csvStartRow option can be used to specify the first row of data in the CSV file, starting from the header row.
The --csvDateFormat option can be used to specify the date format of the time column. 
Further information on valid date format strings can be found in the `Python datetime documentation <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes>`_.

For example:
.. code-block:: console
SAMPLE CSV FILE

HEADER_TIMESTAMP,temperature,X,Y,Z
21/10/2013 10:00:08.000,26.3,-0.078923,0.396706,0.917759
21/10/2013 10:00:08.010,26.2,-0.094370,0.381479,0.933580
21/10/2013 10:00:08.020,26.2,-0.094370,0.366252,0.901938
21/10/2013 10:00:08.030,26.2,-0.078923,0.411933,0.901938
...

Then run the command as follows:
.. code-block:: console

    $ actinet data/sample.csv.gz --txyz HEADER_TIMESTAMP,X,Y,Z --csvStartRow 3 --csvDateFormat "%d/%m/%Y %H:%M:%S.%f"

Other accelerometer file formats
--------------------------------

Multiple accelerometer file formats can be processed by this model. 
These are limited to:

- Axivity (.cwa/.cwa.gz)
- GENEActiv (.bin/.bin.gz)
- Actigraph (.gt3x/.gt3x.gz)
- CSV (.csv/.csv.gz)
- Pickle (.pkl/.pkl.gz)


Visualisation
-------------

The quickest way to visualise the activity intensty classification output is to run the model with the '-p' flag, saving the time-series-plot.png file. 

.. code-block:: console

    $ actinet data/sample.cwa.gz -p
    <summary output written to outputs/sample/sample-summary.json>
    <time-series output written to outputs/sample/sample-timeSeries.csv.gz>
    <plot output written to outputs/sample/sample-timeSeries-plot.png>

Alternatively, you can use the accPlot tool to visualise the time-series output.
This tool offers more options for customisation.

.. code-block:: console

    $ accPlot sample-timeSeries.csv.gz --showFirstNDays 4 --showFileName True --plotFile my_plot.png
    <plot output written to my_plot.png>


Offline usage
=============

To use the classifier and model without internet access:

Option 1: Cache them while online::

    actinet --cache-classifier

Option 2: Manually download from the `ssl-wearables repository <https://github.com/OxWearables/ssl-wearables>`_ and specify paths::

    actinet sample.cwa -c /path/to/classifier.joblib.lzma -m /path/to/ssl-wearables


Processing multiple files
=========================

**Windows**: Create a file *commands.txt* with:

.. code-block:: console

    actinet file1.cwa &
    actinet file2.cwa &
    actinet file3.cwa 
    :END

Run with::

    cmd < commands.txt

**Linux**: Create a file *command.sh* with:

.. code-block:: console

    actinet file1.cwa
    actinet file2.cwa
    actinet file3.cwa

Run with::

    bash command.sh


Collating multiple runs
=======================

To combine output summaries from multiple runs::

    actinet-collate-outputs outputs/


Crude vs. Adjusted Activity Estimates
=====================================

Adjusted estimates account for missing data using imputation:

- Imputes based on means of corresponding timepoints on other days
- Requires full 24h blocks
- Outputs ``NaN`` if still missing after imputation


Troubleshooting
===============

If Java errors occur, try explicitly setting OpenJDK version 8:

.. code-block:: console

    conda create -n actinet openjdk=8


Training a bespoke model
========================

It is also possible to train a bespoke activity classification model.
This requires a labelled dataset of accelerometer data. To do so,
you can use the TrainModel.ipynb notebook with clear instructions,
to show the training of the ActiNet model to your own data.
It should be noted that as the ActiNet model is a deep learning model,
it is strongly advised to use a GPU for training.

To deploy this model, trained and saved locally, to a new set of data,
you can use the command line interface as follows:

.. code-block:: console

    actinet data/sample.cwa -c /path/to/bespoke_classifier.joblib.lzma


Tool versions
==============

Data processing methods are under continual development. We periodically retrain
the classifiers to reflect developments in data processing or the training data.
This means data processed with different versions of the tool may not be
directly comparable.