#!/usr/bin/env python

import argparse
import tempfile
import pandas
import numpy
import os
import re
import colour
import subprocess
import shutil
import statistics
import bz2
import random


class ParentAction(argparse.Action):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, default=[], **kwargs)
        self.children = []

    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest)
        nspace = type(namespace)()
        for child in self.children:
            if (child.name in ChildAction._adjusting_defaults):
                setattr(nspace, child.name, ChildAction._adjusting_defaults[child.name])
            else:
                setattr(nspace, child.name, child.default)
        items.append({'value': values, 'children': nspace})


class ChildAction(argparse.Action):
    _adjusting_defaults = {}

    def __init__(self, *args, parent, sub_action='store', **kwargs):
        super().__init__(*args, **kwargs)

        self.dest, self.name = parent.dest, self.dest
        self.action = sub_action
        self._action = None
        self.parent = parent

        parent.children.append(self)

    def get_action(self, parser):
        if self._action is None:
            action_cls = parser._registry_get('action', self.action, self.action)
            self._action = action_cls(self.option_strings, self.name)
        return self._action

    def __call__(self, parser, namespace, values, option_string=None):
        ChildAction._adjusting_defaults[self.name] = values
        items = getattr(namespace, self.dest)
        try:
            lastParent = items[-1]['children']
        except Exception:
            return
        action = self.get_action(parser)
        action(parser, lastParent, values, option_string)


class Range(object):
    def __init__(self, start=None, end=None):
        if (start is None and end is None):
            raise Exception("Invalid use of Range class!")
        self.start = start
        self.end = end

    def __eq__(self, other):
        if (self.start is None):
            return other <= self.end
        elif (self.end is None):
            return self.start <= other
        else:
            return self.start <= other <= self.end

    def __contains__(self, item):
        return self.__eq__(item)

    def __iter__(self):
        yield self

    def __str__(self):
        if (self.start is None):
            return f'[,{self.end}]'
        elif (self.end is None):
            return f'[{self.start},]'
        else:
            return f'[{self.start},{self.end}]'


def isFloat(val):
    if val is None:
        return False
    try:
        float(val)
        return True
    except ValueError:
        return False


def updateRange(_range, dataList):
    if type(_range) != list:
        raise Exception('updateRange needs a mutable list of min/max directories')
    for a in _range:
        if type(a) != dict:
            raise Exception('updateRange needs a mutable list of directories')
        if 'min' not in a:
            a['min'] = None
        if 'max' not in a:
            a['max'] = None
    while len(_range) < len(dataList):
        _range.extend([{'min': None, 'max': None}])
    for index, data in enumerate(dataList):
        if data is not None:
            if type(data) != list:
                data = [data]
            scope = [x for x in data if isFloat(x)]
            if len(scope) > 0:
                _range[index]['min'] = min(scope) if _range[index]['min'] is None else min(_range[index]['min'], min(scope))
                _range[index]['max'] = max(scope) if _range[index]['max'] is None else max(_range[index]['max'], max(scope))


considerAsNaN = ['nan', 'none', 'null', 'zero', 'nodata', '']
detectDelimiter = ['\t', ';', ' ', ',']

parser = argparse.ArgumentParser(description="Visualize csv files")
# Global Arguments
parser.add_argument("--sort-files", help="sort input files", choices=['asc', 'desc'])
parser.add_argument("-c", "--colour", help="define colours", default=[], action='append', type=colour.Color)
parser.add_argument("--colour-from", help="colour gradient start", default=colour.Color("#084A91"), type=colour.Color)
parser.add_argument("--colour-to", help="colour gradient end", default=colour.Color("#97B5CA"), type=colour.Color)
parser.add_argument("--per-trace-colours", help="one colours for each trace", action='store_true')
parser.add_argument("--per-frame-colours", help="one colour to each input frame file", action='store_true')
parser.add_argument("--per-input-colours", help="one colour to each input file", action='store_true')

inputFileArgument = parser.add_argument('-i', '--input', type=str, help="input file to parse", nargs="+", action=ParentAction)
# Per File Parsing Arguments
parser.add_argument("--special-column-start", help="ignores lines starting with (default '_')", type=str, default='_', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--ignore-line-start", help="ignores lines starting with (default '#')", type=str, default='#', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--separator", help="data delimiter (default auto detection)", type=str, default=None, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--transpose", help="transpose data", default=False, sub_action="store_true", action=ChildAction, parent=inputFileArgument)
parser.add_argument("--no-columns", help="data has no column row", default=False, nargs=0, sub_action="store_true", action=ChildAction, parent=inputFileArgument)
parser.add_argument("--no-index", help="data has no index column", default=False, nargs=0, sub_action="store_true", action=ChildAction, parent=inputFileArgument)
parser.add_argument("--index-icolumn", help="set index column after column index", type=int, choices=Range(0, None), default=None, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--index-column", help="set index column", default=None, type=str, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--split-icolumn", help="split dataset for each unique column value in column index", type=int, choices=Range(0, None), default=None, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--split-column", help="split dataset for each unique column value in column name", default=None, type=str, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--sort", help="sort input data", default=None, choices=['asc', 'desc'], action=ChildAction, parent=inputFileArgument)
parser.add_argument("--sort-column", help="sort after column (only compatible with line and bar plot)", type=int, choices=Range(0, None), default=0, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--name", help="name input data", default=None, type=str, action=ChildAction, parent=inputFileArgument)

# Per File Plotting Arguments:
parser.add_argument('--plot', choices=['line', 'bar', 'box', 'violin'], help='plot type', default='line', action=ChildAction, parent=inputFileArgument)

# parser.add_argument('--plot-width', type=int, choices=Range(0, 1), help='subplot width (relative)', default=1, action=ChildAction, parent=inputFileArgument)
# parser.add_argument('--plot-height', type=int, choices=Range(0, 1), help='subplot height (relative)', default=1, action=ChildAction, parent=inputFileArgument)
parser.add_argument('--row', type=int, choices=Range(1, None), help='subplot row', default=1, action=ChildAction, parent=inputFileArgument)
parser.add_argument('--rowspan', type=int, choices=Range(1, None), help='subplot rowspan', default=1, action=ChildAction, parent=inputFileArgument)
parser.add_argument('--col', type=int, choices=Range(1, None), help='subplot column', default=1, action=ChildAction, parent=inputFileArgument)
parser.add_argument('--colspan', type=int, choices=Range(1, None), help='subplot columnspan', default=1, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--use-input-name", help="use input name for traces", default=False, nargs=0, sub_action="store_true", action=ChildAction, parent=inputFileArgument)
parser.add_argument("--trace-name", help="set trace name", default=[], type=str, sub_action="append", action=ChildAction, parent=inputFileArgument)

parser.add_argument("--line-mode", choices=['lines', 'markers', 'lines+markers', 'lines+text', 'markers+text', 'lines+markers+text'], help="choose linemode", default='lines', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--line-text-position", choices=["top left", "top center", "top right", "middle left", "middle center", "middle right", "bottom left", "bottom center", "bottom right"], help="choose line text positon", default='middle center', action=ChildAction, parent=inputFileArgument)

parser.add_argument("--bar-mode", help="choose barmode", choices=['stack', 'group', 'overlay', 'relative'], default='group')
parser.add_argument("--bar-width", help="use bar width", type=float, choices=Range(0,), default=None, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--bar-text-position", help="choose bar text position", choices=["inside", "outside", "auto", "none"], default='none', action=ChildAction, parent=inputFileArgument)

parser.add_argument("--violin-mode", help="choose violinmode", choices=['overlay', 'group', 'halfoverlay', 'halfgroup', 'halfhalf'], default='overlay')
parser.add_argument("--violin-width", help="change violin widths", type=float, default=0, choices=Range(0,), action=ChildAction, parent=inputFileArgument)
parser.add_argument("--violin-gap", help="change gap between violins (no compatible with violinwidth)", type=float, default=0.3, choices=Range(0, 1))
parser.add_argument("--violin-group-gap", help="change gap between violin groups (not compatible with violinwidth)", type=float, default=0.3, choices=Range(0, 1))

parser.add_argument("--box-mode", choices=['overlay', 'group'], help="choose boxmode", default='overlay')
parser.add_argument("--box-mean", choices=['none', 'line', 'dot'], help="choose box mean", default='dot', action=ChildAction, parent=inputFileArgument)

parser.add_argument("--show-errors", help="show errors if supplied", default=False, nargs=0, sub_action="store_true", action=ChildAction, parent=inputFileArgument)

parser.add_argument("--line-width", help="define line width", type=int, default=1, choices=Range(0,), action=ChildAction, parent=inputFileArgument)
parser.add_argument("--line-colour", help="define line colour (line charts are using just colour)", type=colour.Color, default=colour.Color('#222222'), action=ChildAction, parent=inputFileArgument)

parser.add_argument("--horizontal", help="horizontal chart (default for line)", default=False, nargs=0, sub_action="store_true", action=ChildAction, parent=inputFileArgument)
parser.add_argument("--vertical", help="vertical chart (default for bar, box and violin)", default=False, nargs=0, sub_action="store_true", action=ChildAction, parent=inputFileArgument)
parser.add_argument("--x-range-mode", help="choose range mode for y-axis", choices=['normal', 'tozero', 'nonnegative'], default='normal', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--y-range-mode", help="choose range mode for x-axis", choices=['normal', 'tozero', 'nonnegative'], default='normal', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--x-range-from", help="x-axis start", type=float, default=None, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--y-range-from", help="x-axis start", type=float, default=None, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--x-range-to", help="x-axis end", type=float, default=None, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--y-range-to", help="x-axis end", type=float, default=None, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--x-type", help="choose type for x-axis", choices=['-', 'linear', 'log', 'date', 'category'], default='-', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--y-type", help="choose type for y-axis", choices=['-', 'linear', 'log', 'date', 'category'], default='-', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--x-tick-format", help="change format of x-axis ticks", default='', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--y-tick-format", help="change format of y-axis ticks", default='', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--x-tick-suffix", help="add suffix to x-axis ticks ", default='', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--y-tick-suffix", help="add suffix to y-axis ticks ", default='', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--x-tick-prefix", help="add prefix to x-axis ticks ", default='', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--y-tick-prefix", help="add prefix to y-axis ticks ", default='', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--x-title", help="x-axis title", default=None, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--y-title", help="y-axis title", default=None, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--x-hide", help="hide x-axis", default=False, nargs=0, sub_action="store_true", action=ChildAction, parent=inputFileArgument)
parser.add_argument("--y-hide", help="hide y-axis", default=False, nargs=0, sub_action="store_true", action=ChildAction, parent=inputFileArgument)
parser.add_argument("--opacity", help="colour opacity (default 0.8 for overlay modes)", type=float, choices=Range(0, 1), default=None, action=ChildAction, parent=inputFileArgument)

parser.add_argument("--x-share", help="share subplot x-axis", default=False, action="store_true")
parser.add_argument("--y-share", help="share subplot y-axis", default=False, action="store_true")
parser.add_argument("--vertical-spacing", type=float, help="vertical spacing between subplots", default=0.0, choices=Range(0, 1))
parser.add_argument("--horizontal-spacing", type=float, help="horizontal spacing between subplots", default=0.0, choices=Range(0, 1))
parser.add_argument("--font-size", type=int, help="font size", default=12)
parser.add_argument("--font-family", help="font family", default='"Open Sans", verdana, arial, sans-serif')
parser.add_argument("--font-colour", help="font colour", default='#000000')

parser.add_argument("--legend-x", help="x legend position (-2 to 3)", type=float, choices=Range(-2, 3), default=None)
parser.add_argument("--legend-y", help="y legend position (-2 to 3)", type=float, choices=Range(-2, 3), default=None)
parser.add_argument("--legend-x-anchor", help="set legend xanchor", choices=['auto', 'left', 'center', 'right'], default=None)
parser.add_argument("--legend-y-anchor", help="set legend yanchor", choices=['auto', 'top', 'bottom', 'middle'], default=None)
parser.add_argument("--legend-hide", help="hides legend", default=False, action="store_true")
parser.add_argument("--legend-show", help="forces legend to show up", default=False, action="store_true")
parser.add_argument("--legend-vertical", help="horizontal legend", default=False, action="store_true")
parser.add_argument("--legend-horizontal", help="vertical legend", default=False, action="store_true")

parser.add_argument("--margins", help="sets all margins", type=int, choices=Range(1, None), default=None)
parser.add_argument("--margin-l", help="sets left margin", type=int, choices=Range(1, None), default=None)
parser.add_argument("--margin-r", help="sets right margin", type=int, choices=Range(1, None), default=None)
parser.add_argument("--margin-t", help="sets top margin", type=int, choices=Range(1, None), default=None)
parser.add_argument("--margin-b", help="sets bottom margin", type=int, choices=Range(1, None), default=None)
parser.add_argument("--margin-pad", help="sets padding", type=int, choices=Range(1, None), default=None)

parser.add_argument("-o", "--output", action='append', help="write plot to file (html, pdf, svg, png,...)")
parser.add_argument("--width", help="plot width (not compatible with html)", type=int, default=1000)
parser.add_argument("--height", help="plot height (not compatible with html)", type=int)
parser.add_argument("--save-script", help="save self-contained plotting script", type=str, default=None)
parser.add_argument("--save-only", action="store_true", help="do not execute plotting script (only comptabile with save-script)", default=False)

parser.add_argument("-q", "--quiet", action="store_true", help="do not automatically open output file", default=False)

args = parser.parse_args()

args.per_trace_colours = True if (not args.per_trace_colours and not args.per_input_colours and not args.per_frame_colours) else args.per_trace_colours
args.per_frame_colours = False if (args.per_trace_colours) else args.per_frame_colours
args.per_input_colours = False if (args.per_trace_colours or args.per_frame_colours) else args.per_input_colours

for input in args.input:
    options = input['children']
    if (options.opacity is None and
        ((options.plot == 'box' and 'overlay' in args.box_mode) or
         (options.plot == 'violin' and 'overlay' in args.violin_mode) or
         (options.plot == 'bar' and 'overlay' in args.bar_mode))):
        options.opacity = 0.8
    elif options.opacity is None:
        options.opacity = 1.0

    if (options.horizontal == options.vertical):
        options.vertical = options.plot != 'line'
        options.horizontal = not options.vertical

args.legend_show = not args.legend_hide
args.legend_vertical = True if not args.legend_horizontal else False
args.legend_horizontal = True if not args.legend_vertical else False
if args.legend_x is None:
    args.legend_x = 1.02 if args.legend_vertical else 0
if args.legend_y is None:
    args.legend_y = 1.00 if args.legend_vertical else -0.1

totalTraceCount = 0
totalFrameCount = 0
inputCount = 0
subplotGrid = [{'min': 1, 'max': 1}, {'min': 1, 'max': 1}]
subplotGridDefinition = {}
data = []
defaultBottomMargin = False
defaultLeftMargin = False
defaultRightMargin = False
defaultTopMargin = False
defaultPadMargin = False

for input in args.input:
    options = input['children']
    options.traceCount = 0
    options.frameCount = 0
    subFrames = []
    for fileIndex, filename in enumerate(input['value']):
        if (not os.path.isfile(filename)):
            raise Exception(f'Could not find input file {filename}!')

        if (filename.endswith('.bz2')):
            fFile = bz2.BZ2File(filename, mode='rb').read().decode('utf-8').replace('\r\n', '\n')
        else:
            fFile = open(filename, mode='rb').read().decode('utf-8').replace('\r\n', '\n')

        localSeparator = options.separator
        # Check if we can detect the data delimiter if it was not passed in manually
        if localSeparator is None:
            # Try to find delimiters
            for tryDelimiter in detectDelimiter:
                if sum([x.count(tryDelimiter) for x in fFile.split('\n')]) > 0:
                    localSeparator = tryDelimiter
                    break
            # Fallback if there is just one column and no index column
            localSeparator = ' ' if localSeparator is None and options.no_index else localSeparator
            if (localSeparator is None):
                raise Exception('Could not identify data separator, please specify it manually')

        # Data delimiters clean up, remove multiple separators and separators from the end
        reDelimiter = re.escape(localSeparator)
        fFile = re.sub(reDelimiter + '{1,}\n', '\n', fFile)
        # Tab and space delimiters, replace multiple occurences
        if localSeparator == ' ' or localSeparator == '\t':
            fFile = re.sub(reDelimiter + '{2,}', localSeparator, fFile)
        # Parse the file
        fData = [
            ["NaN" if val.lower() in considerAsNaN else val for val in x.split(localSeparator)]
            for x in fFile.split('\n')
            if (len(x) > 0) and  # Ignore empty lines
            (len(options.ignore_line_start) > 0 and not x.startswith(options.ignore_line_start)) and  # Ignore lines starting with
            (options.no_index or x.count(localSeparator) > 0)  # Ignore lines which contain no data
        ]
        fData = [[float(val) if isFloat(val) else val for val in row] for row in fData]
        if len(fData) < 1 or len(fData[0]) == 0 or (len(fData[0]) < 2 and not options.no_index):
            raise Exception(f'Could not extract any data from file {filename}')

        if options.name is None:
            if (options.no_columns or options.no_index or fData[0][0] is None or len(fData[0][0]) == 0):
                options.name = os.path.basename(filename)
            else:
                options.name = fData[0][0]

        if (options.no_columns):
            frame = pandas.DataFrame(fData)
        else:
            frame = pandas.DataFrame(fData[1:])
            frame.columns = fData[0]

        if (options.transpose):
            frame = frame.T

        # Drop only columns/rows NaN values and replace NaN with None
        frame.dropna(how='all', axis=0, inplace=True)
        frame = frame.where((pandas.notnull(frame)), None)

        if (options.split_icolumn is not None) or (options.split_column is not None):
            iSplitColumn = None
            if (options.split_icolumn is not None):
                if (options.split_icolumn >= frame.shape[1]):
                    raise Exception(f"Split column index {options.split_icolumn} out of bounds in {filename}!")
                else:
                    iSplitColumn = options.split_icolumn
            elif (options.split_column is not None):
                if (options.split_column not in frame.columns):
                    raise Exception(f"Split column {options.split_column} not found in {filename}!")
                else:
                    iSplitColumn = frame.columns.tolist().index(options.split_column)

            for v in frame.iloc[:, iSplitColumn].unique():
                subFrames.append(frame[frame.iloc[:, iSplitColumn] == v])
        else:
            subFrames.append(frame)

    for frame in subFrames:
        if (not options.no_index):
            iIndexColumn = 0
            if (options.index_icolumn is not None):
                if (options.index_icolumn >= frame.shape[1]):
                    raise Exception(f"Index column index {options.index_icolumn} out of bounds in {filename}!")
                else:
                    iIndexColumn = options.index_icolumn
            elif (options.index_column is not None):
                if (options.index_column not in frame.columns):
                    raise Exception(f"Index column {options.index_column} not found in {filename}!")
                else:
                    iIndexColumn = frame.columns.tolist().index(options.index_column)

            newColumns = frame.columns.tolist()
            uniqueColumnName = '_delete_column'
            while uniqueColumnName in frame.columns:
                uniqueColumnName += str(random.randrange(9999))
            newColumns[iIndexColumn] = uniqueColumnName
            frame.set_index(frame.iloc[:, iIndexColumn], inplace=True)
            frame.columns = newColumns
            frame.drop(uniqueColumnName, axis=1, inplace=True)

        if options.sort is not None:
            # Density plots are sorted based on their column mean values
            if (options.plot == 'violin' or options.plot == 'box'):
                frame = frame[frame.mean(axis=0).sort_values(ascending=options.sort == 'asc').index]
            else:
                if (options.sort_column > frame.shape[1]):
                    raise Exception(f"Sort column is out of bounds in {filename}!")
                if (not options.no_index and options.sort_column == 0):
                    frame.sort_index(ascending=options.sort == 'asc', inplace=True)
                else:
                    frame.sort_values(by=frame.columns[options.sort_column - 1], ascending=not options.sort == 'asc', inplace=True)

        options.traceCount += len([x for x in frame.columns if not x.startswith(options.special_column_start)])
        options.frameCount += 1
    totalTraceCount += options.traceCount
    totalFrameCount += options.frameCount
    inputCount += 1

    updateRange(subplotGrid, [options.col + (options.colspan - 1), options.row + (options.rowspan - 1)])
    if (options.row not in subplotGridDefinition):
        subplotGridDefinition[options.row] = {}
    if (options.col not in subplotGridDefinition[options.row]):
        subplotGridDefinition[options.row][options.col] = {'rowspan': options.rowspan, 'colspan': options.colspan}
    if (subplotGridDefinition[options.row][options.col]['rowspan'] < options.rowspan):
        subplotGridDefinition[options.row][options.col]['rowspan'] = options.rowspan
    if (subplotGridDefinition[options.row][options.col]['colspan'] < options.colspan):
        subplotGridDefinition[options.row][options.col]['rowspan'] = options.rowspan

    data.append({'options': options, 'frames': subFrames})

if (args.sort_files):
    data.sort(key=lambda x: numpy.array([y.mean() for y in x['frames']]).mean().mean(), reverse=args.sort_files == 'asc')


# Building up the colour array
requiredColours = totalTraceCount if args.per_trace_colours else totalFrameCount if args.per_frame_colours else inputCount
colours = args.colour if args.colour else list(args.colour_from.range_to(args.colour_to, requiredColours))
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

plotScript.write("\n\nplotly.io.templates.default = 'plotly_white'\n")
plotScript.write("# To add secondary axes, toggle the appropriate boolean values for traces and here:\n")
plotScript.write(f"""fig = plotly.subplots.make_subplots(
    cols={subplotGrid[0]['max']},
    rows={subplotGrid[1]['max']},
    shared_xaxes={args.x_share},
    shared_yaxes={args.y_share},
    vertical_spacing={args.vertical_spacing},
    horizontal_spacing={args.horizontal_spacing},
    specs=[""")
for r in range(1, subplotGrid[1]['max'] + 1):
    plotScript.write("\n    [")
    for c in range(1, subplotGrid[0]['max'] + 1):
        if (r in subplotGridDefinition and c in subplotGridDefinition[r]):
            plotScript.write(f"{{'rowspan': {subplotGridDefinition[r][c]['rowspan']}, 'colspan': {subplotGridDefinition[r][c]['colspan']}, 'secondary_y': False}},")
        else:
            plotScript.write("None,")
    plotScript.write("],")
plotScript.write("""
    ],
)""")

traceIndex = 0
inputIndex = 0
for input in data:
    options = input['options']
    plotRange = []
    frameIndex = 0
    frameTraceIndex = 0
    if (options.row + options.rowspan - 1 == subplotGrid[1]['max'] and (options.y_title is not None or not options.y_hide)):
        defaultBottomMargin = True
    if (options.col == 1 and (options.x_title is not None or not options.x_hide)):
        defaultLeftMargin = True
    for frame in input['frames']:
        showLegend = True if options.traceCount > 1 else False
        for colIndex, _ in enumerate(frame.columns):
            col = str(frame.columns[colIndex])
            specialColumnCount = 4
            _errors = None
            _bases = None
            _labels = None
            _colours = None
            if (col.startswith(options.special_column_start)):
                continue
            for nextColIndex in range(colIndex + 1, colIndex + 1 + specialColumnCount if colIndex + 1 + specialColumnCount <= len(frame.columns) else len(frame.columns)):
                nextCol = str(frame.columns[nextColIndex])
                if (not nextCol.startswith(options.special_column_start)):
                    continue
                if (nextCol == options.special_column_start + 'error') and (_errors is None):
                    _errors = [x if (x is not None) else 0 for x in frame.iloc[:, nextColIndex].values.tolist()]
                elif (nextCol == options.special_column_start + 'base') and (_bases is None):
                    _bases = [x if (x is not None) else 0 for x in frame.iloc[:, nextColIndex].values.tolist()]
                elif (nextCol == options.special_column_start + 'label') and (_labels is None):
                    _labels = frame.iloc[:, nextColIndex].values.tolist()
                elif (nextCol == options.special_column_start + 'colour') and (_colours is None):
                    _colours = frame.iloc[:, nextColIndex].values.tolist()

            if (options.plot == 'line'):
                ydata = frame.iloc[:, colIndex].values.tolist() if not options.vertical else list(frame.index)
                xdata = frame.iloc[:, colIndex].values.tolist() if options.vertical else list(frame.index)
                updateRange(plotRange, [xdata, ydata])
            elif (options.plot == 'bar'):
                ydata = frame.iloc[:, colIndex].tolist() if options.vertical else list(frame.index)
                xdata = frame.iloc[:, colIndex].tolist() if not options.vertical else list(frame.index)
                if _bases is not None:
                    rxdata = xdata
                    rydata = ydata
                    if (options.horizontal):
                        rxdata = [a + b if (a is not None and b is not None) else a if a is not None else b for a, b in zip(xdata, _bases)]
                    else:
                        rydata = [a + b if (a is not None and b is not None) else a if a is not None else b for a, b in zip(ydata, _bases)]
                    updateRange(plotRange, [rxdata, rydata])
                else:
                    updateRange(plotRange, [xdata, ydata])
            else:  # Box and Violin
                data = [x for x in frame.iloc[:, colIndex].values.tolist() if x is not None]
                index = f"['{col}'] * {len(data)}"
                ydata = index if not options.vertical else data
                xdata = index if options.vertical else data
                updateRange(plotRange, [xdata, ydata])

            fillcolour = colours[colourIndex % len(colours)]
            markercolour = colour.Color(options.line_colour)

            traceName = col
            if (frameTraceIndex < len(options.trace_name)):
                traceName = options.trace_name[frameTraceIndex]
            elif (options.use_input_name):
                traceName = options.name

            if options.plot == 'line':
                plotScript.write(f"""
fig.add_trace(go.Scatter(
    name='{traceName}',
    legendgroup='{traceName}',
    mode='{options.line_mode}',""")
                if (_colours is not None):
                    plotScript.write(f"""
    marker_color={_colours},
    line_color='{_colours}',""")
                else:
                    plotScript.write(f"""
    marker_color='{fillcolour.hex}',
    line_color='{fillcolour.hex}',""")
                plotScript.write(f"""
    line_width={options.line_width},
    y={ydata},
    x={xdata},""")
                if (_labels is not None):
                    plotScript.write(f"""
    text={_labels},
    textposition='{options.line_text_position}',""")
                if (_errors is not None):
                    plotScript.write(f"""
    error_{'y' if options.horizontal else 'x'}=dict(
        visible={options.show_errors},
        type='data',
        symmetric=True,
        array={_errors},
    ),""")
                plotScript.write(f"""
    opacity={options.opacity},
), col={options.col}, row={options.row}, secondary_y=False)
""")
            elif options.plot == 'bar':
                plotScript.write(f"""
fig.add_trace(go.Bar(
    name='{traceName}',
    legendgroup='{traceName}',
    orientation='{'v' if options.vertical else 'h'}',""")
                if (_colours is not None):
                    plotScript.write(f"""
    marker_color={_colours},""")
                else:
                    plotScript.write(f"""
    marker_color='{fillcolour.hex}',""")
                plotScript.write(f"""
    marker_line_color='{markercolour.hex}',
    marker_line_width={options.line_width},
    y={ydata},
    x={xdata},
    width={options.bar_width},""")
                if (_labels is not None):
                    plotScript.write(f"""
    text={_labels},
    textposition='{options.bar_text_position}',""")
                if (_bases is not None):
                    plotScript.write(f"""
    base={_bases},""")
                if (_errors is not None):
                    plotScript.write(f"""
    error_{'x' if options.horizontal else 'y'}=dict(
        visible={options.show_errors},
        type='data',
        symmetric=True,
        array={_errors},
    ),""")
                plotScript.write(f"""
    opacity={options.opacity},
), col={options.col}, row={options.row}, secondary_y=False)
""")
            elif options.plot == 'box':
                markercolour = options.line_colour
                plotScript.write(f"""
fig.add_trace(go.Box(
    name='{traceName}',
    legendgroup='{traceName}',
    showlegend={showLegend},
    y={ydata},
    x={xdata},
    boxpoints=False,
    boxmean={True if options.box_mean == 'line' else False},
    fillcolor='{fillcolour.hex}',
    line_color='{markercolour.hex}',
    line_width={args.line_width},
    orientation='{'v' if options.vertical else 'h'}',
    opacity={options.opacity},
), col={options.col}, row={options.row}, secondary_y=False)
""")
                if options.box_mean == 'dot':
                    plotScript.write(f"""
fig.add_trace(go.Scatter(
    name='mean_{traceName}',
    legendgroup='{traceName}',
    showlegend=False,
    x={xdata if args.vertical else [statistics.mean(xdata)]},
    y={ydata if not args.vertical else [statistics.mean(ydata)]},
    fillcolor='{fillcolour.hex}',
    line_color='{markercolour.hex}'
    line_width={options.line_width},
    opacity={options.opacity},
), col={options.col}, row={options.row}, secondary_y=False)
""")

            elif options.plot == 'violin':
                if args.violin_mode == 'halfhalf':
                    side = 'negative' if frameTraceIndex % 2 == 0 else 'positive'
                elif args.violin_mode[:4] == 'half':
                    side = 'positive'
                else:
                    side = 'both'
                markercolour = options.line_colour
                plotScript.write(f"""
fig.add_trace(go.Violin(
    name='{traceName}',
    legendgroup='{traceName}',
    showlegend={showLegend},
    scalegroup='trace{frameTraceIndex}',
    y={ydata},
    x={xdata},
    fillcolor='{fillcolour.hex}',
    line_color='{options.line_colour.hex}',
    line_width={options.line_width},
    side='{side}',
    scalemode='width',
    width={options.violin_width},
    points=False,
    orientation='{'v' if options.vertical else 'h'}',
    opacity={options.opacity},
), col={options.col}, row={options.row}, secondary_y=False)
""")

            showLegend = False
            traceIndex += 1
            frameTraceIndex += 1
            colourIndex += 1 if args.per_trace_colours else 0
        frameIndex += 1
        colourIndex += 1 if args.per_frame_colours else 0
    inputIndex += 1
    colourIndex += 1 if args.per_input_colours else 0
    plotScript.write("\n\n")
    plotScript.write("# Subplot specific options:\n")
    plotScript.write(f"fig.update_yaxes(type='{options.y_type}', rangemode='{options.y_range_mode}', col={options.col}, row={options.row}, secondary_y=False)\n")
    plotScript.write(f"fig.update_xaxes(type='{options.x_type}', rangemode='{options.x_range_mode}', col={options.col}, row={options.row})\n")
    plotScript.write(f"fig.update_yaxes(showline=False, linewidth=0, linecolor='rgba(0,0,0,0)', col={options.col}, row={options.row}, secondary_y=False)\n")
    plotScript.write(f"{'# ' if not options.y_hide else ''}fig.update_yaxes(visible=False, showticklabels=False, showgrid=True, zeroline=False, row={options.row}, col={options.col}, secondary_y=False)\n")
    plotScript.write(f"{'# ' if not options.x_hide else ''}fig.update_xaxes(visible=False, showticklabels=False, showgrid=True, zeroline=False, row={options.row}, col={options.col})\n")
    plotScript.write(f"{'# ' if options.y_title is None else ''}fig.update_yaxes(title_text='{options.y_title}', col={options.col}, row={options.row}, secondary_y=False)\n")
    plotScript.write(f"{'# ' if options.x_title is None else ''}fig.update_xaxes(title_text='{options.x_title}', col={options.col}, row={options.row})\n")
    plotScript.write(f"fig.update_yaxes(tickformat='{options.y_tick_format}', ticksuffix='{options.y_tick_suffix}', tickprefix='{options.y_tick_prefix}', col={options.col}, row={options.row}, secondary_y=False)\n")
    plotScript.write(f"fig.update_xaxes(tickformat='{options.x_tick_format}', ticksuffix='{options.x_tick_suffix}', tickprefix='{options.x_tick_prefix}', col={options.col}, row={options.row})\n")
    plotScript.write(f"# fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', col={options.col}, row={options.row}, secondary_y=False)\n")
    plotScript.write(f"# fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', col={options.col}, row={options.row})\n")
    if options.y_range_from is not None or options.y_range_to is not None:
        options.y_range_from = options.y_range_from if options.y_range_from is not None else plotRange[1]['min']
        options.y_range_to = options.y_range_to if options.y_range_to is not None else plotRange[1]['max']
        plotScript.write(f"fig.update_yaxes(range=[{options.y_range_from}, {options.y_range_to}], col={options.col}, row={options.row}, secondary_y=False)\n")
    if options.x_range_from is not None or options.x_range_to is not None:
        options.x_range_from = options.x_range_from if options.x_range_from is not None else plotRange[0]['min']
        options.x_range_to = options.x_range_to if options.x_range_to is not None else plotRange[0]['max']
        plotScript.write(f"fig.update_xaxes(range=[{options.x_range_from}, {options.x_range_to}], col={options.col}, row={options.row})\n")
    plotScript.write(f"# fig.update_yaxes(range=[{plotRange[0]['min']}, {plotRange[0]['max']}], col={options.col}, row={options.row}, secondary_y=False)\n")
    plotScript.write(f"# fig.update_xaxes(range=[{plotRange[1]['min']}, {plotRange[1]['max']}], col={options.col}, row={options.row}, secondary_y=False)\n")
    plotScript.write("\n\n")

plotScript.write("\n\n")

if (args.violin_mode == 'halfgroup'):
    args.violin_mode = 'group'
elif (args.violin_mode[:4] == 'half'):
    args.violin_mode = 'overlay'

plotScript.write('# Global modes and paramters:\n')
plotScript.write(f"fig.update_layout(barmode='{args.bar_mode}', boxmode='{args.box_mode}', violinmode='{args.violin_mode}', violingap={args.violin_gap}, violingroupgap={args.violin_group_gap})\n")

plotScript.write(f"\n# Layout Legend\n")
plotScript.write(f"fig.update_layout(showlegend={args.legend_show})\n")
plotScript.write(f"{'# ' if args.legend_y_anchor is None else ''}fig.update_layout(legend_yanchor='{'auto' if args.legend_y_anchor is None else args.legend_y_anchor}')\n")
plotScript.write(f"{'# ' if args.legend_x_anchor is None else ''}fig.update_layout(legend_xanchor='{'auto' if args.legend_x_anchor is None else args.legend_x_anchor}')\n")
plotScript.write(f"fig.update_layout(legend=dict(x={args.legend_x}, y={args.legend_y}, orientation='{'v' if args.legend_vertical else 'h'}', bgcolor='rgba(255,255,255,0)'))\n")

plotScript.write(f"\n# Layout Plot and Background\n")
plotScript.write(f"fig.update_layout(paper_bgcolor='rgba(255, 255, 255, 0)', plot_bgcolor='rgba(255, 255, 255, 0)')\n")

args.margin_b = args.margin_b if args.margin_b is not None else args.margins if args.margins is not None else None if defaultBottomMargin else 0
args.margin_l = args.margin_l if args.margin_l is not None else args.margins if args.margins is not None else None if defaultLeftMargin else 0
args.margin_r = args.margin_r if args.margin_r is not None else args.margins if args.margins is not None else None if defaultRightMargin else 0
args.margin_t = args.margin_t if args.margin_t is not None else args.margins if args.margins is not None else None if defaultTopMargin else 0
args.margin_pad = args.margin_pad if args.margin_pad is not None else args.margins if args.margins is not None else None if defaultPadMargin else 0

plotScript.write(f"fig.update_layout(margin=dict(t={args.margin_t}, l={args.margin_l}, r={args.margin_r}, b={args.margin_b}, pad={args.margin_pad}))\n")

plotScript.write(f"\n# Plot Font\n")
plotScript.write(f"fig.update_layout(font=dict(family='{args.font_family}', size={args.font_size}, color='{args.font_colour}'))\n")


plotScript.write("""
# Execute addon file if found
import os
filename, fileext = os.path.splitext(__file__)
if (os.path.exists(f'{filename}_addon{fileext}')):
    exec(open(f'{filename}_addon.py').read())
""")

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
    plotScript.write(f"# exportFigure(fig, {args.width}, {args.height}, 'figure.pdf')\n")
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
