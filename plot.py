#!/usr/bin/env python

import sys
import argparse
import tempfile
import pandas
import os
import re
import colour
import subprocess
import shutil

range = [[None, None], [None, None]]


def isFloat(val):
    if val is None:
        return False
    try:
        float(val)
        return True
    except ValueError:
        return False


def updateRange(dataList):
    global range
    if not range:
        range = []
    if len(range) < len(dataList):
        range.extend([None, None] * (len(dataList) - len(range)))
    for index, data in enumerate(dataList):
        if data is not None:
            scope = [x for x in data if isFloat(x)]
            if len(scope) > 0:
                range[index][0] = min(scope) if range[index][0] is None else min(range[index][0], min(scope))
                range[index][1] = max(scope) if range[index][1] is None else max(range[index][1], max(scope))


considerAsNaN = ['nan', 'none', 'null', 'zero', 'nodata', '']
detectDelimiter = ['\t', ';', ' ', ',']

parser = argparse.ArgumentParser(description="Visualize csv files")
parser.add_argument("files", help="files to parse", nargs="+")
parser.add_argument("--ignore-line-start", help="ignore lines starting with (default '#')", type=str, default='#')
parser.add_argument("--separator", help="data delimiter", type=str)
parser.add_argument("--no-index", action="store_true", help="data has no index column", default=False)
parser.add_argument("--transpose", action="store_true", help="transpose data", default=False)
parser.add_argument("--sort", action="store_true", help="sort the index or columns (for box and violin plot)", default=False)
parser.add_argument("--sort-files", action="store_true", help="sort input files", default=False)
parser.add_argument("--sort-column", type=int, help="sort after column (only compatible with line and bar plot)", default=0)
parser.add_argument("--descending", action="store_true", help="sort descending", default=False)

parser.add_argument("--name", action='append', help="define names")

parser.add_argument("--lines", action="store_true", help="produce a line chart", default=False)
parser.add_argument("--linemode", choices=['line', 'scatter', 'combined'], help="choose linemode", default='line')
parser.add_argument("--bars", action="store_true", help="produce a bar chart", default=False)
parser.add_argument("--barmode", choices=['stack', 'group', 'overlay', 'relative'], help="choose barmode", default='group')
parser.add_argument("--violins", action="store_true", help="produce a violin chart", default=False)
parser.add_argument("--violinmode", choices=['overlay', 'group', 'halfoverlay', 'halfgroup', 'halfhalf'], help="choose violinmode", default='overlay')
parser.add_argument("--violinwidth", type=float, help="change violin widths", default=0)
parser.add_argument("--violingap", type=float, help="change gap between violins (no compatible with violinwidth)", default=0.3)
parser.add_argument("--violingroupgap", type=float, help="change gap between violin groups (not compatible with violinwidth)", default=0.3)
parser.add_argument("--boxes", action="store_true", help="produce a box chart", default=False)
parser.add_argument("--boxmode", choices=['overlay', 'group'], help="choose boxmode", default='overlay')

parser.add_argument("--line-width", type=int, help="define line width", default=1)
parser.add_argument("--line-colour", type=str, help="define line colour (line charts are using just colour)", default='#333333')
parser.add_argument("--horizontal", action="store_true", help="horizontal chart", default=False)
parser.add_argument("--vertical", action="store_true", help="vertical chart", default=False)
parser.add_argument("--rangemode", choices=['normal', 'tozero', 'nonnegative'], help="choose rangemode", default='normal')
parser.add_argument("--y-rangemode", choices=['normal', 'tozero', 'nonnegative'], help="choose rangemode for x-axis", default=None)
parser.add_argument("--x-rangemode", choices=['normal', 'tozero', 'nonnegative'], help="choose rangemode for y-axis", default=None)
parser.add_argument("--x-range-from", type=float, help="x-axis start", default=None)
parser.add_argument("--x-range-to", type=float, help="x-axis end", default=None)
parser.add_argument("--y-range-from", type=float, help="x-axis start", default=None)
parser.add_argument("--y-range-to", type=float, help="x-axis end", default=None)
parser.add_argument("--x-title", help="x-axis title", default=None)
parser.add_argument("--y-title", help="y-axis title", default=None)

parser.add_argument("--font-size", type=int, help="font size", default=12)
parser.add_argument("--font-family", help="font family", default='"Open Sans", verdana, arial, sans-serif')
parser.add_argument("--font-colour", help="font colour", default='#000000')

parser.add_argument("-c", "--colour", action='append', help="define colours")
parser.add_argument("--colour-from", help="colour gradient start", default="#084A91")
parser.add_argument("--colour-to", help="colour gradient end", default="#97B5CA")
parser.add_argument("--opacity", type=float, help="colour opacity (default 0.8 for overlay modes)", default=False)
parser.add_argument("--per-dataset-colour", action='store_true', help="one colour per dataset (default for box and violin)", default=False)
parser.add_argument("--per-trace-colour", action='store_true', help="one colour per trace (default for bar and lines)", default=False)

parser.add_argument("--legend-x", type=float, help="x legend position (-2 to 3)", default=None)
parser.add_argument("--legend-y", type=float, help="y legend position (-2 to 3)", default=None)
parser.add_argument("--legend-x-anchor", choices=['auto', 'left', 'center', 'right'], help="set legend xanchor", default=None)
parser.add_argument("--legend-y-anchor", choices=['auto', 'top', 'bottom', 'middle'], help="set legend yanchor", default=None)
parser.add_argument("--legend-hide", action="store_true", help="hides legend", default=False)
parser.add_argument("--legend-show", action="store_true", help="forces legend to show up", default=False)
parser.add_argument("--legend-vertical", action="store_true", help="horizontal legend", default=False)
parser.add_argument("--legend-horizontal", action="store_true", help="vertical legend", default=False)

parser.add_argument("--margin-l", type=int, help="sets left margin", default=None)
parser.add_argument("--margin-r", type=int, help="sets right margin", default=None)
parser.add_argument("--margin-t", type=int, help="sets top margin", default=None)
parser.add_argument("--margin-b", type=int, help="sets bottom margin", default=None)
parser.add_argument("--margin-pad", type=int, help="sets padding", default=None)

parser.add_argument("-o", "--output", action='append', help="write plot to file (html, pdf, svg, png,...)")
parser.add_argument("--width", help="plot width (not compatible with html)", type=int, default=1000)
parser.add_argument("--height", help="plot height (not compatible with html)", type=int)
parser.add_argument("--save-script", help="save self-contained plotting script", type=str, default=None)
parser.add_argument("--save-only", action="store_true", help="do not execute plotting script (only comptabile with save-script)", default=False)

parser.add_argument("-q", "--quiet", action="store_true", help="do not automatically open output file", default=False)

args = parser.parse_args()

if (not args.files) or (len(args.files) <= 0):
    print("ERROR: unsufficient amount of csv files passed")
    parser.print_help()
    sys.exit(1)

if (args.violingroupgap < 0 or args.violingroupgap > 1):
    print("ERROR: 0 <= violingroupgap <= 1")
    parser.print_help()
    sys.exit(1)

if (args.opacity and (args.opacity < 0 or args.opacity > 1)):
    print("ERROR: 0 <= violingap <= 1")
    parser.print_help()
    sys.exit(1)

if (args.violingap < 0 or args.violingap > 1):
    print("error: 0 <= violingap <= 1")
    parser.print_help()
    sys.exit(1)

if (args.violinwidth < 0):
    print("ERROR: 0 <= violinwidth")
    parser.print_help()
    sys.exit(1)

if (args.sort and args.sort_column < 0):
    print("ERROR: sort column can't be negative")
    parser.print_help()
    sys.exit(1)

if (not args.lines and not args.bars and not args.violins and not args.boxes):
    args.lines = True

if (not args.per_dataset_colour and not args.per_trace_colour):
    args.per_trace_colour = True if (args.lines or args.bars) else False
    args.per_dataset_colour = not args.per_trace_colour
elif (args.per_trace_colour):
    args.per_dataset_colour = False

if args.opacity is False and ((args.boxes and 'overlay' in args.boxmode) or
                              (args.violins and 'overlay' in args.violinmode) or
                              (args.bars and 'overlay' in args.barmode)):
    args.opacity = 0.8
elif args.opacity is False:
    args.opacity = 1.0

if (args.horizontal and args.vertical):
    print("WARNING: cannot plot horizontal and vertical, going to plot horizontal")
    args.vertical = False

if (not args.horizontal and not args.vertical):
    args.horizontal = True

if (args.legend_vertical and args.legend_horizontal):
    print("WARNING: cannot plot legend horizontal and vertical, going to plot vertical")
    args.legend_horizontal = False

if (not args.legend_vertical and not args.legend_horizontal):
    args.legend_vertical = True

if args.legend_x is None:
    args.legend_x = 1.02 if args.legend_vertical else 0
if args.legend_y is None:
    args.legend_y = 1.00 if args.legend_vertical else -0.1

if (args.legend_x < -2 or args.legend_x > 3):
    print("ERROR: legend-x out of range")
    parser.print_help()
    sys.exit(1)

if (args.legend_y < -2 or args.legend_y > 3):
    print("ERROR: legend-y out of range")
    parser.print_help()
    sys.exit(1)

args.linemode = 'lines' if args.linemode == 'line' else 'markers' if args.linemode == 'scatter' else 'lines+markers'

dataFrames = []

# Start parsing the input files, first check if the separator was passed in manually and check if it was a tab
delimiter = args.separator if args.separator else None
if delimiter == '\\t':
    delimiter = '\t'

traceCount = 0
# Going through all files
for sFilename in args.files:
    # Read in as utf-8 and replace windows crlf if necessary
    sFile = open(sFilename, mode='rb').read().decode('utf-8').replace('\r\n', '\n')

    # Check if we can detect the data delimiter if it was not passed in manually
    if delimiter is None:
        # Try to find delimiters
        for tryDelimiter in detectDelimiter:
            if sum([x.count(tryDelimiter) for x in sFile.split('\n')]) > 0:
                delimiter = tryDelimiter
                break
        # Fallback if there is just one column and no index column
        delimiter = ' ' if delimiter is None and args.no_index else delimiter
        if (delimiter is None):
            raise Exception('Could not identify data separator, please specify it manually')
    # Data delimiters clean up, remove multiple separators and separators from the end
    reDelimiter = re.escape(delimiter)
    sFile = re.sub(reDelimiter + '{1,}\n', '\n', sFile)
    # Tab and space delimiters, replace multiple occurences
    if delimiter == ' ' or delimiter == '\t':
        sFile = re.sub(reDelimiter + '{2,}', delimiter, sFile)
    # Parse the file
    fData = [
        ["NaN" if val.lower() in considerAsNaN else val for val in x.split(delimiter)]
        for x in sFile.split('\n')
        if (len(x) > 0) and  # Ignore empty lines
        (len(args.ignore_line_start) > 0 and not x.startswith(args.ignore_line_start)) and  # Ignore lines starting with
        (args.no_index or x.count(delimiter) > 0)  # Ignore lines which contain no data
    ]
    fData = [[float(val) if isFloat(val) else val for val in row] for row in fData]
    if len(fData) < 1 or len(fData[0]) == 0 or (len(fData[0]) < 2 and not args.no_index):
        raise Exception(f'Could not extract any data from file {sFilename}')

    pName = None
    if (args.name and len(dataFrames) < len(args.name)):
        pName = args.name[len(dataFrames)]

    if pName is None and (args.no_index or fData[0][0] is None or len(fData[0][0]) == 0):
        pName = os.path.basename(sFilename)

    if pName is None:
        pName = fData[0][0]

    pFrame = pandas.DataFrame(fData[1:])
    if (not args.no_index):
        fData[0][0] = 'indexCol'

    pFrame.columns = fData[0]
    if (not args.no_index):
        pFrame.set_index('indexCol', inplace=True)

    if (args.transpose):
        pFrame = pFrame.T

    pFrame.dropna(how='all', axis=0, inplace=True)
    pFrame = pFrame.where((pandas.notnull(pFrame)), None)

    if args.sort:
        # Density plots are sorted based on their column mean values
        if (args.violins or args.boxes):
            pFrame = pFrame[pFrame.mean(axis=0).sort_values(ascending=not args.descending).index]
        else:
            if (not args.no_index and args.sort_column == 0):
                pFrame.sort_index(ascending=not args.descending, inplace=True)
            else:
                pFrame.sort_values(by=pFrame.columns[args.sort_column - 1], ascending=not args.descending, inplace=True)

    traceCount += pFrame.shape[1]
    dataFrames.append({'name': pName, 'file': sFilename, 'frame': pFrame})

if (args.sort_files):
    dataFrames.sort(key=lambda x: x['frame'].mean().mean(), reverse=args.descending)

# Building up the colour array
requiredColours = traceCount if args.per_trace_colour else len(dataFrames)
colours = args.colour if args.colour else [c.hex for c in list(colour.Color(args.colour_from).range_to(colour.Color(args.colour_to), requiredColours))]
colourIndex = 0

plotFd = None
if (args.save_script is None):
    args.save_only = False
    plotFd, plotScriptName = tempfile.mkstemp()
    plotScript = open(plotScriptName, 'w+')
else:
    plotScriptName = args.save_script
    plotScript = open(plotScriptName, 'w+')

plotScript.write("""#!/usr/bin/env python
import plotly
import plotly.graph_objects as go
""")

if args.output:
    for output in args.output:
        if not output.endswith('.html'):
            break

plotScript.write("\n\nplotly.io.templates.default = 'plotly_white'\n")
plotScript.write("fig = go.Figure()\n\n")
if args.lines:
    for dataFrame in dataFrames:
        for col in dataFrame['frame'].columns:
            ydata = list(dataFrame['frame'][col]) if not args.vertical else list(dataFrame['frame'].index)
            xdata = list(dataFrame['frame'][col]) if args.vertical else list(dataFrame['frame'].index)
            updateRange([xdata, ydata])
            plotScript.write(f"""
fig.add_trace(go.Scatter(
    name='{col}',
    mode='{args.linemode}',
    legendgroup='{col}',
    line_color='{colours[colourIndex % len(colours)]}',
    line_width={args.line_width},
    y={ydata},
    x={xdata},
))
""")
            colourIndex += 1 if args.per_trace_colour else 0
        colourIndex += 1 if args.per_dataset_colour else 0
elif args.bars:
    for dataFrame in dataFrames:
        for col in dataFrame['frame'].columns:
            fillcolour = colours[colourIndex % len(colours)]
            markercolour = colour.Color(args.line_colour)
            ydata = list(dataFrame['frame'][col]) if args.vertical else list(dataFrame['frame'].index)
            xdata = list(dataFrame['frame'][col]) if not args.vertical else list(dataFrame['frame'].index)
            updateRange([xdata, ydata])

            plotScript.write(f"""
fig.add_trace(go.Bar(
    name='{col}',
    legendgroup='{col}',
    orientation='{'v' if args.vertical else 'h'}',
    marker_color='{fillcolour}',
    marker_line_width={args.line_width},
    marker_line_color='{markercolour}',
    y={ydata},
    x={xdata},
))
""")
            colourIndex += 1 if args.per_trace_colour else 0
        colourIndex += 1 if args.per_dataset_colour else 0
    plotScript.write(f"fig.update_layout(barmode='{args.barmode}')\n")
elif args.boxes:
    traceIndex = 0
    for dataFrame in dataFrames:
        showLegend = True if len(dataFrames) > 1 else False
        for col in dataFrame['frame'].columns:
            data = [x for x in list(dataFrame['frame'][col]) if x is not None]
            index = f"['{col}'] * {len(data)}"
            ydata = index if not args.vertical else data
            xdata = index if args.vertical else data
            updateRange([xdata, ydata])

            fillcolour = colours[colourIndex % len(colours)]
            markercolour = args.line_colour
            plotScript.write(f"""
fig.add_trace(go.Box(
    name='{dataFrame['name']}',
    legendgroup='{dataFrame['name']}',
    showlegend={showLegend},
    y={ydata},
    x={xdata},
    boxpoints=False,
    boxmean=True,
    fillcolor='{fillcolour}',
    line_width={args.line_width},
    line_color='{markercolour}',
    orientation='{'v' if args.vertical else 'h'}',
))
""")
            showLegend = False
            traceIndex += 1
            colourIndex += 1 if args.per_trace_colour else 0
        colourIndex += 1 if args.per_dataset_colour else 0
    plotScript.write(f"\nfig.update_layout(boxmode='{args.boxmode}')\n")
elif args.violins:
    if args.violinmode == 'halfhalf' and len(dataFrames) != 2:
        print("WARNING: violinmode 'halfhalf' only supported for two datasets, fallback to 'halfoverlay")
        args.violinmode = 'halfoverlay'
    traceIndex = 0
    for dataFrame in dataFrames:
        showLegend = True if len(dataFrames) > 1 else False
        for col in dataFrame['frame'].columns:
            data = [x for x in list(dataFrame['frame'][col]) if x is not None]
            index = f"['{col}'] * {len(data)}"
            ydata = index if not args.vertical else data
            xdata = index if args.vertical else data
            updateRange([xdata, ydata])

            side = 'both'
            if args.violinmode == 'halfhalf':
                side = 'negative' if colourIndex % 2 == 0 else 'positive'
            elif args.violinmode[:4] == 'half':
                side = 'positive'
            fillcolour = colours[colourIndex % len(colours)]
            markercolour = args.line_colour
            plotScript.write(f"""
fig.add_trace(go.Violin(
    name='{dataFrame['name']}',
    legendgroup='{dataFrame['name']}',
    showlegend={showLegend},
    scalegroup='trace{traceIndex}',
    y={ydata},
    x={xdata},
    fillcolor='{fillcolour}',
    line_width={args.line_width},
    line_color='{markercolour}',
    side='{side}',
    orientation='{'v' if args.vertical else 'h'}',
))
""")
            showLegend = False
            traceIndex += 1
            colourIndex += 1 if args.per_trace_colour else 0
        colourIndex += 1 if args.per_dataset_colour else 0

    if (args.violinmode == 'halfgroup'):
        args.violinmode = 'group'
    elif (args.violinmode[:4] == 'half'):
        args.violinmode = 'overlay'
    plotScript.write(f"\nfig.update_traces(scalemode='width', width={args.violinwidth}, points=False)\n")
    plotScript.write(f"fig.update_layout(violinmode='{args.violinmode}', violingap={args.violingap}, violingroupgap={args.violingroupgap})\n")

plotScript.write("\n\n")

plotScript.write(f"\n# Layout of axes\n")
plotScript.write(f"fig.update_traces(opacity={args.opacity})\n")

if args.bars or args.boxes or args.violins:
    if (args.vertical):
        plotScript.write("fig.update_layout(xaxis_type='category')\n")
    else:
        plotScript.write("fig.update_layout(yaxis_type='category')\n")


if (args.y_rangemode is None):
    args.y_rangemode = args.rangemode
if (args.x_rangemode is None):
    args.x_rangemode = args.rangemode

plotScript.write(f"fig.update_xaxes(rangemode='{args.x_rangemode}')\n")
plotScript.write(f"fig.update_yaxes(rangemode='{args.y_rangemode}')\n")
plotScript.write(f"# fig.update_xaxes(showline=True, linewidth=1, linecolor='#ffffff')\n")
plotScript.write(f"# fig.update_yaxes(showline=True, linewidth=1, linecolor='#ffffff')\n")


plotScript.write(f"\n# Axes Title\n")
if args.x_title is not None:
    plotScript.write(f"fig.update_layout(xaxis_title='{args.x_title}')\n")
else:
    plotScript.write(f"# fig.update_layout(xaxis_title='No Title')\n")
if args.y_title is not None:
    plotScript.write(f"fig.update_layout(yaxis_title='{args.y_title}')\n")
else:
    plotScript.write(f"# fig.update_layout(yaxis_title='No Title')\n")

plotScript.write(f"\n# Layout Legend\n")
if args.legend_hide:
    plotScript.write(f"fig.update_layout(showlegend=False)\n")
elif args.legend_show:
    plotScript.write(f"fig.update_layout(showlegend=True)\n")
else:
    plotScript.write(f"# fig.update_layout(showlegend=False)\n")

if args.legend_x_anchor:
    plotScript.write(f"fig.update_layout(legend_xanchor='{args.legend_x_anchor}')\n")
else:
    plotScript.write(f"# fig.update_layout(legend_xanchor='auto')\n")

if args.legend_y_anchor:
    plotScript.write(f"fig.update_layout(legend_xanchor='{args.legend_y_anchor}')\n")
else:
    plotScript.write(f"# fig.update_layout(legend_yanchor='auto')\n")

plotScript.write(f"fig.update_layout(legend=dict(x={args.legend_x}, y={args.legend_y}, orientation='{'v' if args.legend_vertical else 'h'}', bgcolor='rgba(255,255,255,0)'))\n")

plotScript.write(f"\n# Layout Plot and Background\n")
plotScript.write(f"fig.update_layout(paper_bgcolor='rgba(255, 255, 255, 0)', plot_bgcolor='rgba(255, 255, 255, 0)')\n")
plotScript.write(f"# fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')\n")
plotScript.write(f"# fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')\n")
if args.y_range_from is not None or args.y_range_to is not None:
    args.y_range_from = args.y_range_from if args.y_range_from is not None else range[1][0]
    args.y_range_to = args.y_range_to if args.y_range_to is not None else range[1][1]
    plotScript.write(f"fig.update_yaxes(range=[{args.y_range_from}, {args.y_range_to}])\n")
else:
    plotScript.write(f"# fig.update_yaxes(range=[{range[1][0]}, {range[1][1]}])\n")

if args.x_range_from is not None or args.x_range_to is not None:
    args.x_range_from = args.x_range_from if args.x_range_from is not None else range[0][0]
    args.x_range_to = args.x_range_to if args.x_range_to is not None else range[0][1]
    plotScript.write(f"fig.update_xaxes(range=[{args.x_range_from}, {args.x_range_to}])\n")
else:
    plotScript.write(f"# fig.update_xaxes(range=[{range[0][0]}, {range[0][1]}])\n")

plotScript.write(f"fig.update_layout(margin=dict(t={0 if args.margin_t is None else args.margin_t}, l={args.margin_l if args.margin_l else 0 if args.y_title is None else None}, r={0 if args.margin_r is None else args.margin_r}, b={args.margin_b if args.margin_b else 0 if args.x_title is None else None}, pad={args.margin_pad}))\n")


plotScript.write(f"\n# Plot Font\n")
plotScript.write(f"fig.update_layout(font=dict(family='{args.font_family}', size={args.font_size}, color='{args.font_colour}'))\n")

plotScript.write("\n\n")

plotScript.write("""
# To export to any other format than html, you need a special orca version from plotly
# https://github.com/plotly/orca/releases
orcaBin = None
import os
import shutil
import subprocess
import tempfile


def determineOrca():
    global orcaBin
    if orcaBin is not None:
        orcaBin = shutil.which(orcaBin)
    else:
        orcaBin = os.getenv('PLOTLY_ORCA')

    if orcaBin is None:
        for executable in ['/opt/plotly-orca/orca', '/opt/plotly/orca', '/opt/orca/orca', '/usr/bin/orca', 'orca']:
            orcaBin = shutil.which(executable)
            if orcaBin is not None:
                break

    if orcaBin is None:
        raise Exception('Could not find orca!')


def exportFigure(fig, width, height, exportFile):
    if exportFile.endswith('.html'):
        plotly.offline.plot(fig, filename=exportFile, auto_open=False)
        return
    else:
        global orcaBin
        if orcaBin is None:
            determineOrca()

        tmpFd, tmpFile = tempfile.mkstemp()
        try:
            exportFile = os.path.abspath(exportFile)
            exportDir = os.path.dirname(exportFile)
            exportFilename = os.path.basename(exportFile)
            _, fileExtension = os.path.splitext(exportFilename)
            fileExtension = fileExtension.lstrip('.')

            go.Figure(fig).write_json(tmpFile)
            cmd = [orcaBin, 'graph', tmpFile, '--output-dir', exportDir, '--output', exportFilename, '--format', fileExtension, '--mathjax', 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js']
            if width is not None:
                cmd.extend(['--width', f'{width}'])
            if height is not None:
                cmd.extend(['--height', f'{height}'])
            subprocess.run(cmd, check=True)
        finally:
            os.remove(tmpFile)


""")

if not args.output:
    plotScript.write("fig.show()\n")
    plotScript.write("# exportFigure(fig, 1920, 1080, 'figure.pdf')\n")
else:
    plotScript.write("# fig.show()\n")
    for output in args.output:
        plotScript.write(f"exportFigure(fig, {args.width if args.width else None}, {args.height if args.height else None}, '{output}')\nprint('Saved to {output}')\n")
    if not args.quiet:
        openWith = None
        for app in ['xdg-open', 'open', 'start']:
            if shutil.which(app) is not None:
                openWith = app
                break
        if openWith is not None:
            for output in args.output:
                plotScript.write(f"""
try:
    subprocess.check_call(['{openWith}', '{output}'], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
except Exception:
    print('Could not open {output}!')
""")

plotScript.close()
if (not args.save_only):
    subprocess.check_call(['python', plotScriptName])
if (args.save_script is None):
    os.close(plotFd)
    os.remove(plotScriptName)
else:
    print(f"Plot script saved to {plotScriptName}")
