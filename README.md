# plot.py

Uses plotly to visualize text files (tab, space, comma separated or any content that has a structure) or pickled pandas dataframes. A lot of features are supported like reading in files with different separators, selecting and sorting columns, processing of multiple files, plotting line, scatter, bar, violin, box and gantt charts and customizing the graphs to all the needs.

This script acts as a master script generating a plotting script. In case this script doesn't cover something that is required for a graph, it can output the final plotting script to make manual adjustments like adding annotations.

## Requirements

PIP3 packages:
* plotly
* numpy
* pandas
* colour

```
pip3 install plotly numpy pandas colour
```

Plotly also requires a special orca version to export any other format than HTML. The plotly orca will be automatically downloaded up on first usage or can be retrieved from https://github.com/plotly/orca/releases/latest .

## Usage

The script is designed to process any formatted data file (e.g. tsv, csv, pandas df) into a table of rows and columns. Every row is a data point while every column is a data trace. Columns starting with '_' are treated as special columns (e.g. _error _labels, _colour or _offset). Rows starting with '#' are treated as comments and are ignored. (Those special characters can be adjusted with the rescpective parameters).

There are global parameters which can be set once per call like all colour options, legend options, width, height or output. The next set or parameters can only be specified on an input file and affect how the file is treated and plotted like row, col, sorting options, selection and ignoring of data points or plot type (line, bar, box, violin). However some options will be inherited by the following input files e.g. row, col and plot type and some won't be inherited like all input file parsing options (sorting, selection...) or options that affect the actual plot like axis titles or ranges. Options that are inherited can also be specified before an input file.

By default plot.py will create an html plot and opens it in the default browser:
```
plot.py -i lines.tsv
```

plot.py can also just print the data it has parsed from the file:
```
plot.py -i lines.tsv --print
```

Or create a standalone plotting script out of the input data which incorporates all the options that can be set and are shown later:
```
plot.py -i lines.tsv --script lines.py
```

All these options can also be combined to print the data, open a browser plot and save it as a PDF:
```
plot.py -i lines.tsv --print --browser --script lines.py --width 1920 --height 1080 --output lines.pdf
```


The full power becomes visisble when working with multiple files (but also the same file more than once) and processing the data and making subplots:
```
plot.py -i lines.tsv --row 1 --select-columns Sin -i lines.tsv --row 2 --index-column Sin --select-columns Cos --sort-rows asc --sort-rows-by column --sort-rows-column Cos
```
Selection of rows and columns is order sensitive and allows custom trace or data point ordering or duplication of traces and datapoints by repeating the names. All data row and column options also support numeric indexing by using e.g. --select-icolumns 0 1 2 etc.

The width and height of individual subplots can be simply adjusted by using colspan and rowspan:
```
plot.py -i lines.tsv --row 1 --rowspan 2 -i lines.tsv --row 3 --rowspan 1
```

It is also possible to plot many files into the same plot by just targetting the same row and col. Setting plot options like x-title, y-title or axis ranges should be done only once per subplot and ideally for the last file of the subplot. 

## Examples

#### Line Chart
```
plot.py -i lines.tsv --plot line --output line.png
```
![Line Chart](/plots/line.png)
#### Scatter Chart
```
plot.py -i scatter.tsv --plot line --line-mode markers --output scatter.png
```
![Scatter Chart](/plots/scatter.png)
#### Bar Chart
```
plot.py -i bar.tsv --plot bar --output bar.png
```
![Bar Chart](/plots/bar.png)
#### Box Chart
```
plot.py -i distribution.tsv --plot box --output box.png
```
![Box Chart](/plots/box.png)
#### Violin Chart
```
plot.py -i distribution.tsv --plot violin --output violin.png
```
![Violin Chart](/plots/violin.png)
#### Gantt Chart
```
plot.py -i gantt.tsv --plot bar --bar-mode stack --bar-text-position inside --orientation h --x-type date --output gantt.png
```
![Gantt Chart](/plots/gantt.png)
```
plot.py -i gantt_time.tsv --plot bar --bar-mode stack --bar-text-position outside --orientation h --x-type date --output gantt_time.png
```
![Gantt Time Chart](/plots/gantt_time.png)
#### Combined Subplots
```
plot.py -i lines.tsv --colspan 2 -i scatter.tsv --row 2 --colspan 1 --line-mode markers -i bar.tsv --col 2 --plot bar -i distribution.csv --row 3 --col 1 --plot violin -i distribution.csv --col 2 --plot box --horizontal-spacing 0.05 --vertical-spacing 0.05 --per-trace-colours --output subplot.png
```
![Subplot Chart](/plots/subplots.png)

# numprops.py

Calculates number properties of the number set passed via paramters or from stdin. Filters any numbers from the given input and can combine number sets given via paramter and stdin. Adjustable precision and POSIX friendly output.

Currently supported number properties:
* count
* sum, min, max
* q1, q2, q3, p% (percentile e.g. p99)
* avg, std, var
* pvalue, spearmanr, pearsonr (require second number set)

```
usage: numprops [-h] [--stdin] [--precision PRECISION]
                [-p PROPERTIES [PROPERTIES ...]] [-q]
                [--secondary [SECONDARY [SECONDARY ...]]] [--debug]
                [primary [primary ...]]

output number properties from numbers passed or read from stdin

positional arguments:
  primary               numbers to calculate properties on (default read from
                        stdin)

optional arguments:
  -h, --help            show this help message and exit
  --stdin               read from stdin even if numbers are provided
  --precision PRECISION
                        force a specific precision
  -p PROPERTIES [PROPERTIES ...], --properties PROPERTIES [PROPERTIES ...]
                        format output (default count, sum, min, max, q2, avg,
                        std) (valid count, sum, min, max, q1, q2, q3, p%, avg,
                        std, var, pvalue, spearmanr, pearsonr)
  -q, --quiet           minimal output
  --secondary [SECONDARY [SECONDARY ...]]
                        secondary number set used e.g. to calculate p-value
                        statistics
  --debug               turn on debug output
```

### Requirements

PIP3 packages:

*  numpy
*  scipy

### Examples

```
> numprops.py 1 2 3 4 5
Count  Sum Min Max  Q2 Avg                  σ
    5 15.0 1.0 5.0 3.0 3.0 1.4142135623730951
    
> numprops.py --properties pvalue --secondary 1 2 3 -- 3 4 5
            P-Value
0.07048399691021993

> cat /proc/cpuinfo | grep MHz | numprops.py
Count                Sum      Min      Max                 Q2        Avg                  σ
   12 45696.465000000004 3781.705 3912.226 3787.1270000000004 3808.03875 43.673346303599956

> cat /proc/cpuinfo | grep MHz | numprops.py --precision 2 -p min max avg p99
    Min     Max     Avg     P99
2444.84 3604.36 3173.42 3590.52
```
