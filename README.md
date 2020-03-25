# plot.py

Uses plotly to visualize CSV files. A lot of features are supported like reading in files with different separators, selecting and sorting columns, processing of multiple files, plotting line, scatter, bar, violin, box and gantt charts and customizing the graphs to all the needs.

This script acts as a master script generating a plotting script. In case this script doesn't cover something that is required for a graph, it can output the final plotting script to make manual adjustments like adding annotations.

To output to any other format than HTML (like pdf, svg, png...), this script requires plotly-orca ( https://github.com/plotly/orca/releases ).

### Examples

#### Line Chart
```
plot.py -i lines.csv --plot line --output line.png
```
![Line Chart](/plots/line.png)
#### Scatter Chart
```
plot.py -i scatter.csv --plot line --line-mode markers --output scatter.png
```
![Scatter Chart](/plots/scatter.png)
#### Bar Chart
```
plot.py -i bar.csv --plot bar --output bar.png
```
![Bar Chart](/plots/bar.png)
#### Box Chart
```
plot.py -i distribution.csv --plot box --output box.png
```
![Box Chart](/plots/box.png)
#### Violin Chart
```
plot.py -i distribution.csv --plot violin --output violin.png
```
![Violin Chart](/plots/violin.png)
#### Gantt Chart
```
plot.py -i gantt.csv --plot bar --bar-mode stack --bar-text-position inside --orientation h --x-type date --output gantt.png
```
![Gantt Chart](/plots/gantt.png)
```
plot.py -i gantt_time.csv --plot bar --bar-mode stack --bar-text-position outside --orientation h --x-type date --output gantt_time.png
```
![Gantt Time Chart](/plots/gantt_time.png)
#### Combined Subplots
```
plot.py -i lines.csv --colspan 2 -i scatter.csv --row 2 --colspan 1 --line-mode markers -i bar.csv --col 2 --plot bar -i distribution.csv --row 3 --col 1 --plot violin -i distribution.csv --col 2 --plot box --horizontal-spacing 0.05 --vertical-spacing 0.05 --per-trace-colours --output subplot.png
```
![Subplot Chart](/plots/subplots.png)

