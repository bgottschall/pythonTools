#!/usr/bin/env python3
#
# Copyright (c) 2020 Björn Gottschall
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


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
import sys
import pickle
import copy
import textwrap


def isFloat(val):
    if val is None:
        return False
    try:
        float(val)
        return True
    except ValueError:
        return False


class ParentAction(argparse.Action):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, default=[], **kwargs)
        self.children = []

    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest)
        nspace = type(namespace)()
        for child in self.children:
            if (not child.sticky_default and child.name in ChildAction._adjusting_defaults):
                setattr(nspace, child.name, ChildAction._adjusting_defaults[child.name])
            else:
                setattr(nspace, child.name, child.default)
        items.append({'value': values, 'children': nspace})


class ChildAction(argparse.Action):
    _adjusting_defaults = {}

    def __init__(self, *args, parent, sub_action='store', sticky_default=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.dest, self.name = parent.dest, self.dest
        self.sticky_default = sticky_default
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
        ChildAction._adjusting_defaults[self.name] = True if self.action == 'store_true' else values
        items = getattr(namespace, self.dest)
        try:
            lastParent = items[-1]['children']
        except Exception:
            if (self.sticky_default):
                raise Exception(f'parameter --{self.name} can only be used after --{self.parent.dest}!') from None
                exit(1)
            return
        action = self.get_action(parser)
        action(parser, lastParent, values, option_string)


class Range(object):
    def __init__(self, start=None, end=None, orValues=None, start_inclusive=True, end_inclusive=True):
        if (start is not None and not isFloat(start)) or (end is not None and not isFloat(end)) or (orValues is not None and not isinstance(orValues, list)):
            raise Exception('invalid use of range object!')
        self.start_inclusive = start_inclusive
        self.end_inclusive = end_inclusive
        self.start = start
        self.end = end
        self.orValues = orValues

    def __eq__(self, other):
        ret = False
        if isFloat(other):
            other = float(other)
            if self.start is None and self.end is None:
                ret = True
            elif self.start is not None and self.end is not None:
                ret = (self.start <= other if self.start_inclusive else self.start < other) and (other <= self.end if self.end_inclusive else other < self.end)
            elif self.start is not None:
                ret = (self.start <= other if self.start_inclusive else self.start < other)
            elif self.end is not None:
                ret = (other <= self.end if self.end_inclusive else other < self.end)
        if not ret and self.orValues is not None:
            ret = other in self.orValues
        return ret

    def __contains__(self, item):
        return self.__eq__(item)

    def __iter__(self):
        yield self

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.start is None and self.end is None:
            ret = '-inf - +inf'
        elif self.start is not None and self.end is not None:
            ret = f'{self.start} - {self.end}'
        elif (self.start is not None):
            ret = f'{self.start} - +inf'
        else:
            ret = f'-inf - {self.end}'
        return ret + (', or ' + ', '.join(self.orValues) if self.orValues is not None and len(self.orValues) > 0 else '')


def updateRange(_range, dataList):
    if not isinstance(_range, list):
        raise Exception('updateRange needs a mutable list of min/max directories')
    for a in _range:
        if not isinstance(a, dict):
            raise Exception('updateRange needs a mutable list of directories')
        if 'min' not in a:
            a['min'] = None
        if 'max' not in a:
            a['max'] = None
    while len(_range) < len(dataList):
        _range.extend([{'min': None, 'max': None}])
    for index, data in enumerate(dataList):
        if data is not None:
            if not isinstance(data, list):
                data = [data]
            scope = [x for x in data if isFloat(x)]
            if len(scope) > 0:
                _range[index]['min'] = min(scope) if _range[index]['min'] is None else min(_range[index]['min'], min(scope))
                _range[index]['max'] = max(scope) if _range[index]['max'] is None else max(_range[index]['max'], max(scope))


considerAsNaN = ['nan', 'none', 'null', 'zero', 'nodata', '']
detectDelimiter = ['\t', ';', ' ', ',']
specialColumns = ['error', 'offset', 'label', 'colour']

parser = argparse.ArgumentParser(description="Visualize data")
# Global Arguments
parser.add_argument("--theme", help="theme to use (colour options only apply to 'gradient')", default='gradient', choices=["gradient", "plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"])
parser.add_argument("-c", "--colours", help="define explicit colours (no gradient)", default=[], nargs='+', type=colour.Color)
parser.add_argument("--colour-from", help="colour gradient start (default %(default)s)", default=colour.Color("#022752"), type=colour.Color)
parser.add_argument("--colour-to", help="colour gradient end (default %(default)s)", default=colour.Color("#CCD9FB"), type=colour.Color)
parser.add_argument("--colour-count", help="colours to use from gradient (overrides per trace, frame and input colours)", type=int, choices=Range(1,), default=None)
parser.add_argument("--per-trace-colours", help="one colour for each trace (default)", action='store_true', default=False)
parser.add_argument("--per-frame-colours", help="one colour for each dataframe", action='store_true', default=False)
parser.add_argument("--per-input-colours", help="one colour for each input file", action='store_true', default=False)

inputFileArgument = parser.add_argument('-i', '--input', type=str, help="input file to parse", nargs="+", action=ParentAction, required=True)
# Per File Parsing Arguments
parser.add_argument("--special-column-start", help="ignores lines starting with (default %(default)s)", type=str, default='_', sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--ignore-line-start", help="ignores lines starting with (default %(default)s)", type=str, default='#', sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--separator", help="data delimiter (auto detected by default)", type=str, default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--join", help="outer join input files on columns or index", default='none', choices=['none', 'index', 'columns'], sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--transpose", help="transpose data", default=False, sticky_default=True, nargs=0, sub_action="store_true", action=ChildAction, parent=inputFileArgument)
parser.add_argument("--no-columns", help="do not use a column row", default=False, sticky_default=True, nargs=0, sub_action="store_true", action=ChildAction, parent=inputFileArgument)
parser.add_argument("--no-index", help="do not use a index column", default=False, sticky_default=True, nargs=0, sub_action="store_true", action=ChildAction, parent=inputFileArgument)
parser.add_argument("--index-icolumn", help="set index column after index", type=int, sticky_default=True, choices=Range(0, None), default=None, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--index-column", help="set index column", default=None, type=str, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--split-icolumn", help="split data along column index", type=int, sticky_default=True, choices=Range(0, None), default=None, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--split-column", help="split datas along column", type=str, sticky_default=True, default=None, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--select-icolumns", help="select these column indexes", type=int, default=[], sticky_default=True, choices=Range(0, None), nargs='+', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--select-columns", help="select these columns", type=str, default=[], sticky_default=True, nargs='+', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--ignore-icolumns", help="ignore these column indexes", type=int, default=[], sticky_default=True, choices=Range(0, None), nargs='+', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--ignore-columns", help="ignore these columns", type=str, default=[], sticky_default=True, nargs='+', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--select-irows", help="select these row indexes", type=int, default=[], sticky_default=True, choices=Range(0, None), nargs='+', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--select-rows", help="select these rows", type=str, default=[], sticky_default=True, nargs='+', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--ignore-irows", help="ignore these row indexes", type=int, default=[], sticky_default=True, choices=Range(0, None), nargs='+', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--ignore-rows", help="ignore these rows", type=str, default=[], sticky_default=True, nargs='+', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--sort-columns", help="sort column (default %(default)s)", default='none', choices=['none', 'asc', 'desc'], sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--sort-columns-by", help="sort column after method or row (default %(default)s)", default='mean', choices=['mean', 'median', 'std', 'min', 'max', 'row'], sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--sort-columns-irow", help="sort column after this row index (requires sorting by 'row') (default %(default)s)", type=int, default=None, choices=Range(0, None), sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--sort-columns-row", help="sort column after this row (requires sorting by 'row') (default %(default)s)", type=str, default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--sort-rows", help="sort rows (default %(default)s)", default='none', choices=['none', 'asc', 'desc'], sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--sort-rows-by", help="sort rows after method or column (default %(default)s)", default='mean', choices=['mean', 'median', 'std', 'min', 'max', 'column'], sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--sort-rows-icolumn", help="sort rows after this column index (requires sorting by 'column') (default %(default)s)", type=int, default=None, choices=Range(0, None), sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--sort-rows-column", help="sort rows after this column (requires sorting by 'column') (default %(default)s)", type=str, default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--pickle-frames", help="pickle data frames to file (one file containing all frames)", default=None, type=str, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--file-frames", help="save data frames to text files (one file per frame)", default=None, type=str, nargs='+', sticky_default=True, action=ChildAction, parent=inputFileArgument)

# Per File Plotting Arguments:
parser.add_argument('--plot', choices=['line', 'bar', 'box', 'violin'], help='plot type', default='line', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--title", help="subplot title", type=str, default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--use-name", help="use name for traces", type=str, default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--trace-names", help="set individual trace names", default=[], sticky_default=True, type=str, nargs='+', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--trace-colours", help="define explicit trace colours", default=[], nargs='+', type=colour.Color, sticky_default=True, action=ChildAction, parent=inputFileArgument)

parser.add_argument('--row', type=int, choices=Range(1, None), help='subplot row (default %(default)s)', default=1, action=ChildAction, parent=inputFileArgument)
parser.add_argument('--rowspan', type=int, choices=Range(1, None), help='subplot rowspan (default %(default)s)', default=1, action=ChildAction, parent=inputFileArgument)
parser.add_argument('--col', type=int, choices=Range(1, None), help='subplot column (default %(default)s)', default=1, action=ChildAction, parent=inputFileArgument)
parser.add_argument('--colspan', type=int, choices=Range(1, None), help='subplot columnspan (default %(default)s)', default=1, action=ChildAction, parent=inputFileArgument)

parser.add_argument("--line-mode", choices=['lines', 'markers', 'text', 'lines+markers', 'lines+text', 'markers+text', 'lines+markers+text'], help="choose linemode (default %(default)s)", default='lines', action=ChildAction, parent=inputFileArgument)
parser.add_argument('--line-shape', choices=['linear', 'spline', 'hv', 'vh', 'hvh', 'vhv'], help='choose line shape (default %(default)s)', default='linear', action=ChildAction, parent=inputFileArgument)
parser.add_argument('--line-dash', choices=['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot'], help='choose line dash (default %(default)s)', default='solid', action=ChildAction, parent=inputFileArgument)
parser.add_argument('--line-marker', choices=['circle', 'circle-open', 'circle-dot', 'circle-open-dot', 'square', 'square-open', 'square-dot', 'square-open-dot', 'diamond', 'diamond-open', 'diamond-dot', 'diamond-open-dot', 'cross', 'cross-open', 'cross-dot', 'cross-open-dot', 'x', 'x-open', 'x-dot', 'x-open-dot', 'triangle-up', 'triangle-up-open', 'triangle-up-dot', 'triangle-up-open-dot', 'triangle-down', 'triangle-down-open', 'triangle-down-dot', 'triangle-down-open-dot', 'triangle-left', 'triangle-left-open', 'triangle-left-dot', 'triangle-left-open-dot', 'triangle-right', 'triangle-right-open', 'triangle-right-dot', 'triangle-right-open-dot', 'triangle-ne', 'triangle-ne-open', 'triangle-ne-dot', 'triangle-ne-open-dot', 'triangle-se', 'triangle-se-open', 'triangle-se-dot', 'triangle-se-open-dot', 'triangle-sw', 'triangle-sw-open', 'triangle-sw-dot', 'triangle-sw-open-dot', 'triangle-nw', 'triangle-nw-open', 'triangle-nw-dot', 'triangle-nw-open-dot', 'pentagon', 'pentagon-open', 'pentagon-dot', 'pentagon-open-dot', 'hexagon', 'hexagon-open', 'hexagon-dot', 'hexagon-open-dot', 'hexagon2', 'hexagon2-open', 'hexagon2-dot', 'hexagon2-open-dot', 'octagon', 'octagon-open', 'octagon-dot', 'octagon-open-dot', 'star', 'star-open', 'star-dot', 'star-open-dot', 'hexagram', 'hexagram-open', 'hexagram-dot', 'hexagram-open-dot', 'star-triangle-up', 'star-triangle-up-open', 'star-triangle-up-dot', 'star-triangle-up-open-dot', 'star-triangle-down', 'star-triangle-down-open', 'star-triangle-down-dot', 'star-triangle-down-open-dot', 'star-square', 'star-square-open', 'star-square-dot', 'star-square-open-dot', 'star-diamond', 'star-diamond-open', 'star-diamond-dot', 'star-diamond-open-dot', 'diamond-tall', 'diamond-tall-open', 'diamond-tall-dot', 'diamond-tall-open-dot', 'diamond-wide', 'diamond-wide-open', 'diamond-wide-dot', 'diamond-wide-open-dot', 'hourglass', 'hourglass-open', 'bowtie', 'bowtie-open', 'circle-cross', 'circle-cross-open', 'circle-x', 'circle-x-open', 'square-cross', 'square-cross-open', 'square-x', 'square-x-open', 'diamond-cross', 'diamond-cross-open', 'diamond-x', 'diamond-x-open', 'cross-thin', 'cross-thin-open', 'x-thin', 'x-thin-open', 'asterisk', 'asterisk-open', 'hash', 'hash-open', 'hash-dot', 'hash-open-dot', 'y-up', 'y-up-open', 'y-down', 'y-down-open', 'y-left', 'y-left-open', 'y-right', 'y-right-open', 'line-ew', 'line-ew-open', 'line-ns', 'line-ns-open', 'line-ne', 'line-ne-open', 'line-nw', 'line-nw-open'], help='choose line marker (default %(default)s)', default='circle', action=ChildAction, parent=inputFileArgument)
parser.add_argument('--line-marker-size', help='choose line marker size (default %(default)s)', type=int, default=6, choices=Range(0, None), action=ChildAction, parent=inputFileArgument)
parser.add_argument("--line-text-position", choices=["top left", "top center", "top right", "middle left", "middle center", "middle right", "bottom left", "bottom center", "bottom right"], help="choose line text positon (default %(default)s)", default='middle center', action=ChildAction, parent=inputFileArgument)

parser.add_argument("--bar-mode", help="choose barmode (default %(default)s)", choices=['stack', 'group', 'overlay', 'relative'], default='group')
parser.add_argument("--bar-width", help="set explicit bar width", choices=Range(0, None, ['auto']), default='auto', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--bar-shift", help="set bar shift", choices=Range(None, None, ['auto']), default='auto', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--bar-text-position", help="choose bar text position (default %(default)s)", choices=["inside", "outside", "auto", "none"], default='none', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--bar-gap", help="set bar gap (default $(default)s)", choices=Range(0, 1, ['auto']), default='auto')
parser.add_argument("--bar-group-gap", help="set bar group gap (default $(default)s)", choices=Range(0, 1), default=0)

parser.add_argument("--violin-mode", help="choose violinmode (default %(default)s)", choices=['overlay', 'group', 'halfoverlay', 'halfgroup', 'halfhalf'], default='overlay')
parser.add_argument("--violin-width", help="change violin widths (default %(default)s)", type=float, default=0, choices=Range(0,), action=ChildAction, parent=inputFileArgument)
parser.add_argument("--violin-gap", help="gap between violins (default %(default)s) (not compatible with violin-width)", type=float, default=0.3, choices=Range(0, 1))
parser.add_argument("--violin-group-gap", help="gap between violin groups (default %(default)s) (not compatible with violin-width)", type=float, default=0.3, choices=Range(0, 1))

parser.add_argument("--box-mode", choices=['overlay', 'group'], help="choose boxmode (default %(default)s)", default='overlay')
parser.add_argument("--box-mean", choices=['none', 'line', 'dot'], help="choose box mean (default %(default)s)", default='dot', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--box-width", help="box width (default %(default)s)", type=float, default=0, choices=Range(0,), action=ChildAction, parent=inputFileArgument)
parser.add_argument("--box-gap", help="gap between boxes (default %(default)s) (not compatible with box-width)", type=float, default=0.3, choices=Range(0, 1))
parser.add_argument("--box-group-gap", help="gap between box groups (default %(default)s) (not compatible with box-width)", type=float, default=0.3, choices=Range(0, 1))

parser.add_argument("--error", help="show error if supplied", default='hide', choices=['show', 'hide'], action=ChildAction, parent=inputFileArgument)

parser.add_argument("--line-width", help="set line width (default %(default)s)", type=int, default=1, choices=Range(0,), action=ChildAction, parent=inputFileArgument)
parser.add_argument("--line-colour", help="set line colour  (default %(default)s) (line charts are using just colour)", type=colour.Color, default=colour.Color('#222222'), action=ChildAction, parent=inputFileArgument)
parser.add_argument("--opacity", help="colour opacity (default 0.8 for overlay modes, else 1.0)", choices=Range(0, 1, ['auto']), action=ChildAction, parent=inputFileArgument)

parser.add_argument("--orientation", help="set plot orientation", default='auto', choices=['vertical', 'v', 'horizontal', 'h', 'auto'], action=ChildAction, parent=inputFileArgument)
parser.add_argument("--y-secondary", help="plot to secondary y-axis", default=False, sticky_default=True, nargs=0, sub_action="store_true", action=ChildAction, parent=inputFileArgument)
parser.add_argument("--x-title", help="x-axis title", default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--y-title", help="y-axis title", default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--x-type", help="choose type for x-axis (default %(default)s)", choices=['-', 'linear', 'log', 'date', 'category'], default='-', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--y-type", help="choose type for y-axis (default %(default)s)", choices=['-', 'linear', 'log', 'date', 'category'], default='-', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--x-range-mode", help="choose range mode for y-axis (default %(default)s)", choices=['normal', 'tozero', 'nonnegative'], default='normal', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--y-range-mode", help="choose range mode for x-axis (default %(default)s)", choices=['normal', 'tozero', 'nonnegative'], default='normal', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--x-range-from", help="x-axis start (default %(default)s)", default='auto', sticky_default=True, choices=Range(None, None, ['auto']), action=ChildAction, parent=inputFileArgument)
parser.add_argument("--y-range-from", help="y-axis start (default %(default)s)", default='auto', sticky_default=True, choices=Range(None, None, ['auto']), action=ChildAction, parent=inputFileArgument)
parser.add_argument("--x-range-to", help="x-axis start (default %(default)s)", default='auto', sticky_default=True, choices=Range(None, None, ['auto']), action=ChildAction, parent=inputFileArgument)
parser.add_argument("--y-range-to", help="y-axis end (default %(default)s)", default='auto', sticky_default=True, choices=Range(None, None, ['auto']), action=ChildAction, parent=inputFileArgument)
parser.add_argument("--x-tick-format", help="set format of x-axis ticks", default='', sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--y-tick-format", help="set format of y-axis ticks", default='', sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--x-tick-suffix", help="add suffix to x-axis ticks", default='', sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--y-tick-suffix", help="add suffix to y-axis ticks", default='', sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--x-tick-prefix", help="add prefix to x-axis ticks", default='', sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--y-tick-prefix", help="add prefix to y-axis ticks", default='', sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--x-ticks", help="how to draw x ticks (default '%(default)s')", choices=['', 'inside', 'outside'], default='', sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--y-ticks", help="how to draw y ticks (default '%(default)s')", choices=['', 'inside', 'outside'], default='', sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--x-tickmode", help="tick mode x-axis (default '%(default)s')", choices=['auto', 'linear', 'array'], default='auto', sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--y-tickmode", help="tick mode y-axis (default '%(default)s')", choices=['auto', 'linear', 'array'], default='auto', sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--x-nticks", help="number of ticks on x-axis (only tick mode auto) (default %(default)s)", choices=Range(0,), default=0, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--y-nticks", help="number of ticks on y-axis (only tick mode auto) (default %(default)s)", choices=Range(0,), default=0, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--x-tick0", help="first tick on x-axis (only tick mode linear) (default %(default)s)", type=str, default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--y-tick0", help="first tick on y-axis (only tick mode linear) (default %(default)s)", type=str, default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--x-dtick", help="tick step on x-axis (only tick mode linear) (default %(default)s)", type=str, default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--y-dtick", help="tick step on y-axis (only tick mode linear) (default %(default)s)", type=str, default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parser.add_argument("--x-tickvals", help="tick values on x-axis (only tick mode array) (default %(default)s)", default=[], sticky_default=True, nargs='+', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--y-tickvals", help="tick values on y-axis (only tick mode array) (default %(default)s)", default=[], sticky_default=True, nargs='+', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--x-ticktext", help="tick text on x-axis (only tick mode array) (default %(default)s)", default=[], sticky_default=True, nargs='+', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--y-ticktext", help="tick text on y-axis (only tick mode array) (default %(default)s)", default=[], sticky_default=True, nargs='+', action=ChildAction, parent=inputFileArgument)
parser.add_argument("--x-tickangle", help="tick angle on x-axis (default %(default)s)", default='auto', sticky_default=True, choices=Range(None, None, ['auto']), action=ChildAction, parent=inputFileArgument)
parser.add_argument("--y-tickangle", help="tick angle on y-axis (default %(default)s)", default='auto', sticky_default=True, choices=Range(None, None, ['auto']), action=ChildAction, parent=inputFileArgument)

parser.add_argument("--x-hide", help="hide x-axis", default=False, sticky_default=True, nargs=0, sub_action="store_true", action=ChildAction, parent=inputFileArgument)
parser.add_argument("--y-hide", help="hide y-axis", default=False, sticky_default=True, nargs=0, sub_action="store_true", action=ChildAction, parent=inputFileArgument)

parser.add_argument("--master-title", help="plot master title", type=str, default=None)
parser.add_argument("--x-master-title", help="x-axis master title", type=str, default=None)
parser.add_argument("--y-master-title", help="y-axis master title", type=str, default=None)
parser.add_argument("--x-share", help="share subplot x-axis (default %(default)s)", default=False, action="store_true")
parser.add_argument("--y-share", help="share subplot y-axis (default %(default)s)", default=False, action="store_true")
parser.add_argument("--vertical-spacing", type=float, help="vertical spacing between subplots (default %(default)s)", default=0.08, choices=Range(0, 1))
parser.add_argument("--horizontal-spacing", type=float, help="horizontal spacing between subplots (default %(default)s)", default=0.08, choices=Range(0, 1))
parser.add_argument("--font-size", help="font size (default %(default)s)", type=int, default=12)
parser.add_argument("--font-family", help="font family (default %(default)s)", type=str, default='"Open Sans", verdana, arial, sans-serif')
parser.add_argument("--font-colour", help="font colour (default %(default)s)", type=colour.Color, default=colour.Color('#000000'))

parser.add_argument("--legend", help="quick setting the legend position (default %(default)s)", type=str, choices=['topright', 'topcenter', 'topleft', 'bottomright', 'bottomcenter', 'bottomleft', 'middleleft', 'center', 'middleright', 'belowleft', 'belowcenter', 'belowright', 'aboveleft', 'abovecenter', 'aboveright', 'righttop', 'rightmiddle', 'rightbottom', 'lefttop', 'leftmiddle', 'leftbottom'], default='righttop')
parser.add_argument("--legend-entries", help="choose which entries are shown in legend", choices=['all', 'unique', 'none'], default=None)
parser.add_argument("--legend-x", help="x legend position (-2 to 3)", type=float, choices=Range(-2, 3), default=None)
parser.add_argument("--legend-y", help="y legend position (-2 to 3)", type=float, choices=Range(-2, 3), default=None)
parser.add_argument("--legend-x-anchor", help="set legend xanchor", choices=['auto', 'left', 'center', 'right'], default=None)
parser.add_argument("--legend-y-anchor", help="set legend yanchor", choices=['auto', 'top', 'bottom', 'middle'], default=None)
parser.add_argument("--legend-hide", help="hides legend", default=None, action="store_true")
parser.add_argument("--legend-show", help="forces legend to show up", default=None, action="store_true")
parser.add_argument("--legend-vertical", help="horizontal legend", default=None, action="store_true")
parser.add_argument("--legend-horizontal", help="vertical legend", default=None, action="store_true")

parser.add_argument("--margins", help="sets all margins", type=int, choices=Range(0, None), default=None)
parser.add_argument("--margin-l", help="sets left margin", type=int, choices=Range(0, None), default=None)
parser.add_argument("--margin-r", help="sets right margin", type=int, choices=Range(0, None), default=None)
parser.add_argument("--margin-t", help="sets top margin", type=int, choices=Range(0, None), default=None)
parser.add_argument("--margin-b", help="sets bottom margin", type=int, choices=Range(0, None), default=None)
parser.add_argument("--margin-pad", help="sets padding", type=int, choices=Range(0, None), default=None)

parser.add_argument("--orca", help="plotly-orca binary required to output every format except html (https://github.com/plotly/orca)", type=str, default=None)
parser.add_argument("--script", help="save self-contained plotting script", type=str, default=None)
parser.add_argument("--print", help="print plotting information and data", default=False, action="store_true")
parser.add_argument("--browser", help="open plot in the browser", default=False, action="store_true")
parser.add_argument("-o", "--output", help="export plot to file (html, pdf, svg, png,...)", default=[], nargs='+')
parser.add_argument("--width", help="output width", type=int, default=1000)
parser.add_argument("--height", help="output height", type=int)

parser.add_argument("-q", "--quiet", action="store_true", help="no warnings and don't open output file", default=False)

args = parser.parse_args()

commentColour=''

if args.theme == 'gradient':
    args.theme = 'plotly_white'
else:
    # We have chosen a theme, so just comment all colour settings out
    commentColour = '# '
    # Better to show all legend entries now if not otherwise chosen
    if args.legend_entries is None:
        args.legend_entries = 'all'

# Setting the legend entries default in case nothing was chosen
if args.legend_entries is None:
    args.legend_entries = 'unique'

if (not args.per_trace_colours and not args.per_frame_colours and not args.per_input_colours) or (args.per_trace_colours):
    args.per_trace_colours = True
    args.per_frame_colours, args.per_input_colours = False, False
elif (args.per_frame_colours):
    args.per_input_colours = False

for input in args.input:
    options = input['children']
    options.ignore_icolumns = list(set(options.ignore_icolumns))
    options.ignore_columns = list(set(options.ignore_columns))
    options.specialColumns = [options.special_column_start + x for x in specialColumns]

    if (options.opacity == 'auto' and
        ((options.plot == 'box' and 'overlay' in args.box_mode) or
         (options.plot == 'violin' and 'overlay' in args.violin_mode) or
         (options.plot == 'bar' and 'overlay' in args.bar_mode))):
        options.opacity = 0.8
    elif options.opacity == 'auto':
        options.opacity = 1.0

    if options.orientation == 'auto':
        options.vertical = options.plot != 'line'
    elif options.orientation in ['vertical', 'v']:
        options.vertical = True
    else:
        options.vertical = False
    options.horizontal = not options.vertical

    if options.error == 'show':
        options.show_error = True
    else:
        options.show_error = False
    options.hide_error = not options.show_error

    options.y_range_from = None if options.y_range_from == 'auto' else float(options.y_range_from)
    options.x_range_from = None if options.x_range_from == 'auto' else float(options.x_range_from)
    options.y_range_to = None if options.y_range_to == 'auto' else float(options.y_range_to)
    options.x_range_to = None if options.x_range_to == 'auto' else float(options.x_range_to)
    options.bar_width = None if options.bar_width == 'auto' else float(options.bar_width)
    options.bar_shift = None if options.bar_shift == 'auto' else float(options.bar_shift)
    options.y_tickangle = None if options.y_tickangle == 'auto' else float(options.y_tickangle)
    options.x_tickangle = None if options.x_tickangle == 'auto' else float(options.x_tickangle)
   
args.bar_gap = None if args.bar_gap == 'auto' else float(args.bar_gap)
args.master_title = f"'{args.master_title}'" if args.master_title is not None else None
args.y_master_title = f"'{args.y_master_title}'" if args.y_master_title is not None else None
args.x_master_title = f"'{args.x_master_title}'" if args.x_master_title is not None else None

if (args.legend_show is not None or args.legend_hide is not None):
    args.legend_show = not args.legend_hide
    args.legend_hide = not args.legend_show

# Setting the legend orientation if it was explicitly set
if args.legend_vertical is not None or args.legend_horizontal is not None:
    args.legend_vertical = not args.legend_horizontal
    args.legend_horizontal = not args.legend_vertical

if args.legend is not None:
    # If not legend orientation is set, set the default depending on the position
    if args.legend_horizontal is None:
        if args.legend.startswith('top') or args.legend.startswith('bottom') or args.legend.startswith('above') or args.legend.startswith('below'):
            args.legend_horizontal = True
        else:
            args.legend_horizontal = False
    args.legend_vertical = not args.legend_horizontal

    if (args.legend_y_anchor is None):
        if args.legend.startswith('middle') or args.legend.endswith('middle') or args.legend == 'center':
            args.legend_y_anchor = 'middle'
        elif args.legend.startswith('top') or args.legend.startswith('below'):
            args.legend_y_anchor = 'top'
        elif args.legend.startswith('bottom') or args.legend.startswith('above'):
            args.legend_y_anchor = 'bottom'

    if (args.legend_x_anchor is None):
        if args.legend.endswith('center') or args.legend == 'center':
            args.legend_x_anchor = 'center'
        elif args.legend.endswith('right') or args.legend.startswith('left'):
            args.legend_x_anchor = 'right'
        elif args.legend.endswith('left') or args.legend.startswith('right'):
            args.legend_x_anchor = 'left'

    if (args.legend_y is None):
        if args.legend.startswith('middle') or args.legend.endswith('middle') or args.legend == 'center':
            args.legend_y = 0.5
        elif args.legend.startswith('top') or args.legend.endswith('top'):
            args.legend_y = 1.0
        elif args.legend.startswith('bottom') or args.legend.endswith('bottom'):
            args.legend_y = 0.0
        elif args.legend.startswith('above'):
            args.legend_y = 1.0
        elif args.legend.startswith('below'):
            args.legend_y = -0.05

    if (args.legend_x is None):
        if args.legend.endswith('center') or args.legend == 'center':
            args.legend_x = 0.5
        elif args.legend.endswith('left'):
            args.legend_x = 0.0
        elif args.legend.endswith('right'):
            args.legend_x = 1.0
        elif args.legend.startswith('right'):
            args.legend_x = 1.02
        elif args.legend.startswith('left'):
            args.legend_x = -0.05

totalTraceCount = 0
totalFrameCount = 0
totalInputCount = 0
subplotGrid = [{'min': 1, 'max': 1}, {'min': 1, 'max': 1}]
subplotGridDefinition = {}
data = []

# None means it will be set automatically to True/False
defaultBottomMargin = True if args.x_master_title is not None else None
defaultLeftMargin = True if args.y_master_title is not None else None
defaultRightMargin = None
# Those are never set automatically so either use True or False
defaultTopMargin = True if args.master_title is not None else None
defaultPadMargin = False

doneSomething = False

for input in args.input:
    inputOptions = input['children']
    inputOptions.traceCount = 0
    inputOptions.frameCount = 0
    masterFrame = None
    masterFrames = []
    for filename in input['value']:
        if (not os.path.isfile(filename)):
            raise Exception(f'Could not find input file {filename}!')

        if (filename.endswith('.bz2')):
            rawFile = bz2.BZ2File(filename, mode='rb')
        else:
            rawFile = open(filename, mode='rb')

        try:
            frame = pickle.load(rawFile)
        except Exception:
            frame = None
            rawFile.seek(0)

        if frame is not None:
            if (isinstance(frame, list)):
                for f in frame:
                    if (not isinstance(f, pandas.DataFrame)):
                        raise Exception(f'pickle file {filename} is not a list of pandas dataframes!')
                    masterFrames.append((copy.deepcopy(inputOptions), f))
            elif (not isinstance(frame, pandas.DataFrame)):
                raise Exception(f'pickle file {filename} is not a pandas data frame!')
            else:
                masterFrames.append((copy.deepcopy(inputOptions), frame))

            if not args.quiet and inputOptions.no_columns:
                print("WARNING: ignoring --no-columns for {filename}", file=sys.stderr)

            for _index, (options, frame) in enumerate(masterFrames):
                if (options.transpose):
                    frame = frame.transpose()
                    # Restore an set index as first column:
                if (not isinstance(frame.index, pandas.RangeIndex)):
                    frame.reset_index(inplace=True)

                masterFrames[_index] = (options, frame)
        else:
            fFile = rawFile.read().decode('utf-8').replace('\r\n', '\n')
            options = copy.deepcopy(inputOptions)

            # Check if we can detect the data delimiter if it was not passed in manually
            if options.separator is None:
                # Try to find delimiters
                for tryDelimiter in detectDelimiter:
                    if sum([x.count(tryDelimiter) for x in fFile.split('\n')]) > 0:
                        options.separator = tryDelimiter
                        break
                # Fallback if there is just one column and no index column
                options.separator = ' ' if options.separator is None and options.no_index else options.separator
                if (options.separator is None):
                    raise Exception('Could not identify data separator, please specify it manually')

            # Data delimiters clean up, remove multiple separators and separators from the end
            reDelimiter = re.escape(options.separator)
            fFile = re.sub(reDelimiter + '{1,}\n', '\n', fFile)
            # Tab and space delimiters, replace multiple occurences
            if options.separator == ' ' or options.separator == '\t':
                fFile = re.sub(reDelimiter + '{2,}', options.separator, fFile)
            # Parse the file
            fData = [
                [None if val.lower() in considerAsNaN else val for val in x.split(options.separator)]
                for x in fFile.split('\n')
                if (len(x) > 0) and  # Ignore empty lines
                (len(options.ignore_line_start) > 0 and not x.startswith(options.ignore_line_start)) and  # Ignore lines starting with
                (options.no_index or x.count(options.separator) > 0)  # Ignore lines which contain no data
            ]
            fData = [[float(val) if isFloat(val) else val for val in row] for row in fData]
            if len(fData) < 1 or len(fData[0]) == 0 or (len(fData[0]) < 2 and not options.no_index):
                raise Exception(f'Could not extract any data from file {filename}')

            fData = numpy.array(fData, dtype=object)

            if (options.transpose):
                fData = numpy.transpose(fData)

            if (options.no_columns):
                frame = pandas.DataFrame(fData)
            else:
                frame = pandas.DataFrame(fData[1:])

            if (not options.no_columns):
                frame.columns = fData[0]

            masterFrames.append((options, frame))

    for _index, (options, frame) in enumerate(masterFrames):
        if (not options.no_index):
            iIndexColumn = 0
            if (options.index_icolumn is not None):
                if (options.index_icolumn >= frame.shape[1]):
                    raise Exception(f"Index column index {options.index_icolumn} out of bounds in {', '.join(input['value'])}!")
                else:
                    iIndexColumn = options.index_icolumn
            elif (options.index_column is not None):
                if (options.index_column not in frame.columns):
                    raise Exception(f"Index column {options.index_column} not found in {', '.join(input['value'])}!")
                else:
                    iIndexColumn = frame.columns.tolist().index(options.index_column)
            options.index_icolumn = iIndexColumn
        else:
            options.index_icolumn = None

        masterFrames[_index] = (options, frame)

    if options.join != 'none':
        joinedFrame = None
        newIndex = None
        revisedOptions = copy.deepcopy(inputOptions)
        for options, frame in masterFrames:
            if (inputOptions.join == 'index' and not inputOptions.no_index):
                if newIndex is None:
                    newIndex = frame.columns[options.index_icolumn]
                frame.set_index(frame.columns[options.index_icolumn], inplace=True)
            elif not inputOptions.no_index and joinedFrame is None:
                revisedOptions.index_icolumn = options.index_icolumn
            joinedFrame = frame if joinedFrame is None else pandas.concat([joinedFrame, frame], axis=1 if inputOptions.join == 'index' else 0, join='outer', verify_integrity=False, copy=True)

        if (inputOptions.join == 'index' and not inputOptions.no_index):
            joinedFrame.index.name = newIndex
            joinedFrame.reset_index(inplace=True)
            revisedOptions.index_icolumn = 0

        masterFrames = [(revisedOptions, joinedFrame)]

    newMasterFrames = []
    for _index, (options, frame) in enumerate(masterFrames):
        if options.sort_columns != 'none':
            if len(options.select_icolumns) > 0:
                if not args.quiet:
                    print("WARNING: --select-columns and --select-icolumns column order overrides --sort-columns!")
            else:
                sortKey = None
                if options.sort_columns_by == 'row':
                    if options.sort_columns_irow is None and options.sort_columns_row is None:
                        options.sort_columns_irow = 0
                    elif options.sort_columns_irow is None:
                        if (options.index_icolumn is not None):
                            lIndex = frame.iloc[:, options.index_icolumn].tolist()
                        else:
                            lIndex = frame.index.tolist()
                        if (options.sort_columns_row not in lIndex):
                            raise Exception(f"Sort row {options.sort_columns_row} not found in files {', '.join(input['value'])}")
                        options.sort_columns_irow = lIndex.index(options.sort_columns_row)
                    if (options.sort_columns_irow >= frame.shape[0]):
                        raise Exception("Sort row is out of bounds in files {', '.join(input['value'])}")
                    sortKey = frame.iloc[options.sort_columns_irow].apply(pandas.to_numeric, errors='coerce')
                    # sortKey = frame.iloc[options.sort_columns_irow]
                else:
                    sortKey = getattr(frame.apply(pandas.to_numeric, errors='coerce'), options.sort_columns_by)(axis=0)
                frame = frame[sortKey.sort_values(ascending=options.sort_columns == 'asc').index]
                if (options.index_icolumn is not None):
                    options.index_icolumn = frame.columns.tolist().index(sortKey.index.tolist()[options.index_icolumn])

        if options.sort_rows != 'none':
                sortKey = None
                if options.sort_rows_by == 'column':
                    if options.sort_rows_icolumn is None and options.sort_rows_column is None:
                        options.sort_rows_icolumn = 0
                    elif options.sort_rows_icolumn is None:
                        lColumns = frame.columns.tolist()
                        if (options.sort_rows_column not in lColumns):
                            raise Exception(f"Sort column {options.sort_rows_column} not found in files {', '.join(input['value'])}")
                        options.sort_rows_icolumn = lColumns.index(options.sort_rows_column)
                    if (options.sort_rows_icolumn >= frame.shape[1]):
                        raise Exception("Sort column is out of bounds in files {', '.join(input['value'])}")
                    # sortKey = frame.iloc[:, options.sort_rows_icolumn].apply(pandas.to_numeric, errors='coerce')
                    sortKey = frame.iloc[:, options.sort_rows_icolumn]
                else:
                    filterColumns = numpy.array([True] * frame.shape[1])
                    if options.index_icolumn is not False:
                        filterColumns[options.index_icolumn] = False
                    sortKey = getattr(frame.loc[:, filterColumns].apply(pandas.to_numeric, errors='coerce'), options.sort_rows_by)(axis=1)
                frame = frame.reindex(sortKey.sort_values(ascending=options.sort_rows == 'asc').index)

        # Column Selection
        if len(options.ignore_icolumns) > 0:
            options.ignore_icolumns = [i for i in options.ignore_icolumns if i >= 0 and i < frame.shape[1]]

        if len(options.ignore_columns) > 0:
            for i, c in enumerate(frame.columns):
                if c in options.ignore_columns:
                    options.ignore_icolumns.append(i)
            options.ignore_icolumns = list(set(options.ignore_icolumns))

        selectColumns = len(options.select_icolumns) > 0 or len(options.select_columns) > 0

        if len(options.select_icolumns) > 0:
            selectCount = len(options.select_icolumns)
            options.select_icolumns = [i for i in options.select_icolumns if i >= 0 and i < frame.shape[1]]
            if not args.quiet and (selectCount != len(options.select_icolumns)):
                print(f"WARNING: some selected column indexes where not found in files {', '.join(input['value'])}", file=sys.stderr)

        if len(options.select_columns) > 0:
            selectedColumns = []
            for sc in options.select_columns:
                for i, c in enumerate(frame.columns):
                    if (sc == c):
                        options.select_icolumns.append(i)
                        selectedColumns.append(c)
            if not args.quiet and len(options.select_columns) != len(selectedColumns):
                print(f"WARNING: some selected columns where not found in files {', '.join(input['value'])}", file=sys.stderr)

        if len(options.select_icolumns) > 0 and len(options.ignore_icolumns) > 0:
            options.select_icolumns = [i for i in options.select_icolumns if i not in options.ignore_icolumns]

        if (len(options.select_icolumns) == 0) and selectColumns:
            raise Exception(f"No selected columns found or all are ignored in files {', '.join(input['value'])}!")

        # Row Selection
        if len(options.ignore_irows) > 0:
            options.ignore_irows = [i for i in options.ignore_irows if i >= 0 and i < frame.shape[0]]

        if len(options.ignore_rows) > 0:
            for i, r in enumerate(frame.iloc[:, options.index_icolumn].tolist() if options.index_icolumn is not None else frame.index.tolist()):
                if r in options.ignore_rows:
                    options.ignore_irows.append(i)
            options.ignore_irows = list(set(options.ignore_irows))

        selectRows = len(options.select_irows) > 0 or len(options.select_rows) > 0

        if len(options.select_irows) > 0:
            selectCount = len(options.select_irows)
            options.select_irows = [i for i in options.select_irows if i >= 0 and i < frame.shape[0]]
            if not args.quiet and (selectCount != len(options.select_irows)):
                print(f"WARNING: some selected row indexes where not found in files {', '.join(input['value'])}", file=sys.stderr)

        if len(options.select_rows) > 0:
            selectedRows = []
            for sr in options.select_rows:
                for i, r in enumerate(frame.iloc[:, options.index_icolumn].tolist() if options.index_icolumn is not None else frame.index.tolist()):
                    if (sr == r):
                        options.select_irows.append(i)
                        selectedRows.append(r)
            if not args.quiet and len(options.select_rows) != len(selectedRows):
                print(f"WARNING: some selected rows where not found in files {', '.join(input['value'])}", file=sys.stderr)

        if len(options.select_irows) > 0 and len(options.ignore_irows) > 0:
            options.select_irows = [i for i in options.select_irows if i not in options.ignore_irows]

        if (len(options.select_irows) == 0) and selectRows:
            raise Exception(f"No selected rows found or all are ignored in files {', '.join(input['value'])}!")

        # Frame splitting
        if (options.split_icolumn is not None) or (options.split_column is not None):
            iSplitColumn = None
            if (options.split_icolumn is not None):
                if (options.split_icolumn >= frame.shape[1]):
                    raise Exception(f"Split column index {options.split_icolumn} out of bounds in files {', '.join(input['value'])}!")
                else:
                    iSplitColumn = options.split_icolumn
            elif (options.split_column is not None):
                if (options.split_column not in frame.columns):
                    raise Exception(f"Split column {options.split_column} not found in files {', '.join(input['value'])}!")
                else:
                    iSplitColumn = frame.columns.tolist().index(options.split_column)
            for v in frame.iloc[:, iSplitColumn].unique():
                newFrame = frame[frame.iloc[:, iSplitColumn] == v].reset_index(drop=True)
                newMasterFrames.append((options, newFrame))
        else:
            masterFrames[_index] = (options, frame)
            newMasterFrames.append(masterFrames[_index])

    masterFrames = newMasterFrames

    inputOptions.traceCount = 0
    for _index, (options, frame) in enumerate(masterFrames):

        # Filter selected/ignored rows
        if len(options.select_irows) > 0:
            frame = frame.iloc[options.select_irows, :]
        elif len(options.ignore_irows) > 0:
            filterRows = numpy.array([False if i in options.ignore_irows else True for i in range(frame.shape[0])])
            frame = frame.loc[filterRows, :]

        # Filter selcted/ignored columns
        if (options.index_icolumn is not None):
            frame.set_index(frame.iloc[:, options.index_icolumn], inplace=True)
            # If the index column was explicitly selected, do not remove it
            if options.index_icolumn not in options.select_icolumns:
                filterColumns = numpy.array([True] * frame.shape[1])
                filterColumns[options.index_icolumn] = False
                frame = frame.loc[:, filterColumns]
                if len(options.select_icolumns) > 0:
                    options.select_icolumns = [i - 1 if i >= options.index_icolumn else i for i in options.select_icolumns]
                elif len(options.ignore_icolumns) > 0:
                    options.ignore_icolumns = [i - 1 if i >= options.index_icolumn else i for i in options.ignore_icolumns]
        else:
            frame = frame.reset_index(drop=True)

        if len(options.select_icolumns) > 0:
            newFrame = None
            for i in options.select_icolumns:
                column = frame.iloc[:, i].to_frame()
                newFrame = column if newFrame is None else pandas.concat([newFrame, column], axis=1, verify_integrity=False, copy=True)
            frame = newFrame
        elif len(options.ignore_icolumns) > 0:
            filterColumns = numpy.array([False if i in options.ignore_icolumns else True for i in range(frame.shape[1])])
            frame = frame.loc[:, filterColumns]
        inputOptions.traceCount += len([x for x in frame.columns if not str(x) in options.specialColumns])
        masterFrames[_index] = (options, frame)
        totalFrameCount += 1

        if options.file_frames is not None and _index < len(options.file_frames):
            doneSomething = True
            sFile = options.file_frames[_index]
            sep = '\t' if options.separator is None or len(options.separator) > 1 else options.separator
            if sFile.endswith('.tsv'):
                sep = '\t'
            elif sFile.endswith('.csv'):
                sep = ';'
            else:
                if not args.quiet and options.separator is not None and len(options.separator) > 1:
                    print(f"WARNING: cannot use separator '{options.separator}' (length > 1) for exporting the data frame, default to '\\t'", file=sys.stderr)
            frame.to_csv(sFile, sep=sep, na_rep='NaN')
            if not args.quiet:
                if (len(masterFrames) == 1):
                    print(f'Frame saved to {options.file_frames[_index]}')
                else:
                    print(f'Frame {_index + 1}/{len(masterFrames)} saved to {options.file_frames[_index]}')

    inputOptions.inputIndex = totalInputCount
    totalTraceCount += inputOptions.traceCount
    totalInputCount += 1

    updateRange(subplotGrid, [inputOptions.col + (inputOptions.colspan - 1), inputOptions.row + (inputOptions.rowspan - 1)])
    if (inputOptions.row not in subplotGridDefinition):
        subplotGridDefinition[inputOptions.row] = {}
    if (inputOptions.col not in subplotGridDefinition[inputOptions.row]):
        subplotGridDefinition[inputOptions.row][inputOptions.col] = {'rowspan': inputOptions.rowspan, 'colspan': inputOptions.colspan, 'secondary_y': inputOptions.y_secondary, 'title': inputOptions.title}

    subplotGridDefinition[inputOptions.row][inputOptions.col]['rowspan'] = max(inputOptions.rowspan, subplotGridDefinition[inputOptions.row][inputOptions.col]['rowspan'])
    subplotGridDefinition[inputOptions.row][inputOptions.col]['colspan'] = max(inputOptions.colspan, subplotGridDefinition[inputOptions.row][inputOptions.col]['colspan'])
    subplotGridDefinition[inputOptions.row][inputOptions.col]['secondary_y'] = inputOptions.y_secondary or subplotGridDefinition[inputOptions.row][inputOptions.col]['secondary_y']
    if inputOptions.title is not None:
        subplotGridDefinition[inputOptions.row][inputOptions.col]['title'] = inputOptions.title

    if (args.print):
        doneSomething = True
        pFiles = f"Files: {', '.join(input['value'])}"
        pGrid = f"Frames: {len(masterFrames)}, Plot: {inputOptions.plot}  Grid: [ {inputOptions.col}{' - ' + str(inputOptions.col + inputOptions.colspan - 1) if inputOptions.colspan > 1 else ''}, {inputOptions.row}{' - ' + str(inputOptions.row + inputOptions.rowspan - 1) if inputOptions.rowspan > 1 else ''} ]"
        pSep = '-' * min(80, max(len(pGrid), len(pFiles)))
        print(pSep + '\n' + textwrap.fill(pFiles, width=80, subsequent_indent=' ') + '\n' + textwrap.fill(pGrid, width=80, subsequent_indent=' ') + '\n' + pSep)
        for _, f in masterFrames:
            with pandas.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.max_columns', None):
                print(f)
            print(pSep)

    if options.pickle_frames is not None:
        if options.pickle_frames.endswith(".bz2"):
            fDataframe = bz2.BZ2File(options.pickle_frames, mode='wb')
        else:
            fDataframe = open(options.pickle_frames, mode="wb")
        pickle.dump([f for _, f in masterFrames], fDataframe, pickle.HIGHEST_PROTOCOL)
        fDataframe.close()
        doneSomething = True
        if not args.quiet:
            print(f'Dataframe saved to {options.pickle_frames}')

    data.append({'options': options, 'frames': [f for _, f in masterFrames]})


if doneSomething and not args.browser and len(args.output) == 0 and not args.script:
    exit(0)
elif len(args.output) == 0 and not args.script:
    args.browser = True

# Plotting script will be executed in this path context
if args.script:
    scriptContext = os.path.abspath(os.path.dirname(args.script))
else:
    scriptContext = os.path.abspath(os.getcwd())

# Converting paths to new relative paths to the plotting script
# In case a script is saved, those paths are still valid no matter from where its called
for i, p in enumerate(args.output):
    if not os.path.isabs(p):
        args.output[i] = os.path.relpath(p, scriptContext)

# Building up the colour array
requiredColours = args.colour_count if args.colour_count is not None else totalTraceCount if args.per_trace_colours else totalFrameCount if args.per_frame_colours else totalInputCount
colours = args.colours if args.colours else list(args.colour_from.range_to(args.colour_to, requiredColours))
colourIndex = 0

legendEntries = []

plotFd = None
if (args.script is None):
    args.script_only = False
    plotFd, plotScriptName = tempfile.mkstemp()
    plotScript = open(plotScriptName, 'w+')
else:
    plotScriptName = os.path.abspath(args.script)
    plotScript = open(plotScriptName, 'w+')

plotScript.write(f"""#!/usr/bin/env python3
#
# Generated by plot.py from https://github.com/bgottschall/pythonTools
#
# Copyright (c) 2020 Björn Gottschall
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
import shutil
import subprocess
import tempfile
import argparse
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

parser = argparse.ArgumentParser(description="plots the contained figure")
parser.add_argument("--font-size", help="font size (default %(default)s)", type=int, default={args.font_size})
parser.add_argument("--orca", help="path to plotly orca (https://github.com/plotly/orca)", type=str, default=None)
parser.add_argument("--width", help="width of output file (default %(default)s)", type=int, default={args.width})
parser.add_argument("--height", help="height of output (default %(default)s)", type=int, default={args.height})
parser.add_argument("--output", help="output file (html, png, jpeg, pdf...) (default %(default)s)", type=str, nargs="+", default={args.output})
parser.add_argument("--browser", help="open plot in browser", action="store_true")
parser.add_argument("--quiet", help="no warnings and don't open output file", action="store_true")

args = parser.parse_args()

if len(args.output) == 0:
    args.browser = True
""")

plotScript.write("""


def checkOrca(orca = 'orca'):
    if orca is not None:
        orca = shutil.which(orca)
    if orca is None:
        raise Exception('Could not find plotly orca please provide it via --orca (https://github.com/plotly/orca)')
    orcaOutput = subprocess.run([orca, '--help'], stdout=subprocess.PIPE, check=True)
    if 'Plotly\\'s image-exporting utilities' not in orcaOutput.stdout.decode():
       raise Exception(f'Invalid orca version {orca}. Please provide the correct version via --orca (https://github.com/plotly/orca)')
    return orca


def exportFigure(fig, width, height, exportFile, orca = 'orca'):
    if exportFile.endswith('.html'):
        plotly.offline.plot(fig, filename=exportFile, auto_open=False)
        return
    else:
        tmpFd, tmpFile = tempfile.mkstemp()
        try:
            exportFile = os.path.abspath(exportFile)
            exportDir = os.path.dirname(exportFile)
            exportFilename = os.path.basename(exportFile)
            _, fileExtension = os.path.splitext(exportFilename)
            fileExtension = fileExtension.lstrip('.')

            go.Figure(fig).write_json(tmpFile)
            cmd = [orca, 'graph', tmpFile, '--output-dir', exportDir, '--output', exportFilename, '--format', fileExtension, '--mathjax', 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js']
            if width is not None:
                cmd.extend(['--width', f'{width}'])
            if height is not None:
                cmd.extend(['--height', f'{height}'])
            exportRun = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if exportRun.returncode != 0:
                print(f'ERROR: failed to export figure to {exportFile}! Unsupported file format?')
                print(exportRun.stderr.decode('utf-8'))
                exit(1)
        finally:
            os.remove(tmpFile)

""")

subplotTitles = []
print(subplotGridDefinition)

plotScript.write(f"""\n\nplotly.io.templates.default = '{args.theme}'

fig = make_subplots(
    cols={subplotGrid[0]['max']},
    rows={subplotGrid[1]['max']},
    shared_xaxes={args.x_share},
    shared_yaxes={args.y_share},
    y_title={args.y_master_title},
    x_title={args.x_master_title},
    vertical_spacing={args.vertical_spacing},
    horizontal_spacing={args.horizontal_spacing},
    specs=[""")
for r in range(1, subplotGrid[1]['max'] + 1):
    plotScript.write("\n    [")
    for c in range(1, subplotGrid[0]['max'] + 1):
        if (r in subplotGridDefinition and c in subplotGridDefinition[r]):
            plotScript.write(f"{{'rowspan': {subplotGridDefinition[r][c]['rowspan']}, 'colspan': {subplotGridDefinition[r][c]['colspan']}, 'secondary_y': {subplotGridDefinition[r][c]['secondary_y']}}},")
            subplotTitles.append('' if subplotGridDefinition[r][c]['title'] is None else subplotGridDefinition[r][c]['title'])
        else:
            plotScript.write("None,")
    plotScript.write("],")
plotScript.write(f"""
    ],
    subplot_titles={subplotTitles}
)""")

currentInputIndex = None
frameIndex = 0
traceIndex = 0
for input in data:
    options = input['options']
    frames = input['frames']
    plotRange = []
    inputTraceIndex = 0
    inputFrameIndex = 0
    for frame in frames:
        # NaN cannot be plotted or used, cast it to None
        # Drop only columns/rows NaN values and replace NaN with None
        frame = frame.dropna(how='all', axis=0)
        frame = frame.where((pandas.notnull(frame)), None)

        frameTraceIndex = 0

        for colIndex, _ in enumerate(frame.columns):
            if options.trace_colours and frameTraceIndex < len(options.trace_colours):
                fillcolour = options.trace_colours[frameTraceIndex]
            else:
                fillcolour = colours[colourIndex % len(colours)]
            markercolour = colour.Color(options.line_colour)
            col = str(frame.columns[colIndex])
            specialColumnCount = len(options.specialColumns)
            _errors = None
            _bases = None
            _labels = None
            _colours = None
            if (col in options.specialColumns):
                continue
            for nextColIndex in range(colIndex + 1, colIndex + 1 + specialColumnCount if colIndex + 1 + specialColumnCount <= len(frame.columns) else len(frame.columns)):
                nextCol = str(frame.columns[nextColIndex])
                if (nextCol not in options.specialColumns):
                    continue
                if (nextCol == options.special_column_start + 'error') and (_errors is None):
                    _errors = [x if (x is not None) else 0 for x in frame.iloc[:, nextColIndex].values.tolist()]
                elif (nextCol == options.special_column_start + 'offset') and (_bases is None):
                    _bases = [x if (x is not None) else 0 for x in frame.iloc[:, nextColIndex].values.tolist()]
                elif (nextCol == options.special_column_start + 'label') and (_labels is None):
                    _labels = frame.iloc[:, nextColIndex].values.tolist()
                elif (nextCol == options.special_column_start + 'colour') and (_colours is None) and (frameTraceIndex >= len(options.trace_colours)):
                    _colours = frame.iloc[:, nextColIndex].values.tolist()
                    _colours = [c if c is not None else fillcolour.hex for c in _colours]

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

            traceName = col

            if (inputTraceIndex < len(options.trace_names)):
                traceName = options.trace_names[inputTraceIndex]
            elif (options.use_name is not None):
                traceName = options.use_name

            showInLegend = args.legend_entries == 'all'
            if traceName not in legendEntries:
                if args.legend_entries == 'unique':
                    showInLegend = True
                legendEntries.append(traceName)

            if options.plot == 'line':
                plotScript.write(f"""
fig.add_trace(go.Scatter(
    name='{traceName}',
    legendgroup='{traceName}',
    showlegend={showInLegend},
    mode='{options.line_mode}',""")
                if (_colours is not None):
                    plotScript.write(f"""
    marker_color={_colours},
    line_color='{_colours[0]}',""")
                else:
                    plotScript.write(f"""
{commentColour}    marker_color='{fillcolour.hex}',
{commentColour}    line_color='{fillcolour.hex}',""")
                plotScript.write(f"""
    marker_symbol='{options.line_marker}',
    marker_size={options.line_marker_size},
    line_dash='{options.line_dash}',
    line_shape='{options.line_shape}',
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
        visible={options.show_error},
        type='data',
        symmetric=True,
        array={_errors},
    ),""")
                plotScript.write(f"""
    opacity={options.opacity},
), col={options.col}, row={options.row}, secondary_y={options.y_secondary})
""")
            elif options.plot == 'bar':
                plotScript.write(f"""
fig.add_trace(go.Bar(
    name='{traceName}',
    legendgroup='{traceName}',
    showlegend={showInLegend},
    orientation='{'v' if options.vertical else 'h'}',""")
                if (_colours is not None):
                    plotScript.write(f"""
    marker_color={_colours},""")
                else:
                    plotScript.write(f"""
{commentColour}    marker_color='{fillcolour.hex}',""")
                plotScript.write(f"""
{commentColour}    marker_line_color='{markercolour.hex}',
    marker_line_width={options.line_width},
    width={options.bar_width},
    offset={options.bar_shift},
    y={ydata},
    x={xdata},""")
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
        visible={options.show_error},
        type='data',
        symmetric=True,
        array={_errors},
    ),""")
                plotScript.write(f"""
    opacity={options.opacity},
), col={options.col}, row={options.row}, secondary_y={options.y_secondary})
""")
            elif options.plot == 'box':
                markercolour = options.line_colour
                plotScript.write(f"""
fig.add_trace(go.Box(
    name='{traceName}',
    legendgroup='{traceName}',
    showlegend={showInLegend},
    y={ydata},
    x={xdata},
    boxpoints=False,
    boxmean={True if options.box_mean == 'line' else False},
    width={options.box_width},
{commentColour}    fillcolor='{fillcolour.hex}',
{commentColour}    line_color='{markercolour.hex}',
    line_width={options.line_width},
    orientation='{'v' if options.vertical else 'h'}',
    opacity={options.opacity},
), col={options.col}, row={options.row}, secondary_y={options.y_secondary})
""")
                if options.box_mean == 'dot':
                    plotScript.write(f"""
fig.add_trace(go.Scatter(
    name='mean_{traceName}',
    legendgroup='{traceName}',
    showlegend=False,
    x={xdata if options.vertical else [statistics.mean(xdata)]},
    y={ydata if not options.vertical else [statistics.mean(ydata)]},
{commentColour}    fillcolor='{fillcolour.hex}',
{commentColour}    line_color='{markercolour.hex}',
    line_width={options.line_width},
    opacity={options.opacity},
), col={options.col}, row={options.row}, secondary_y={options.y_secondary})
""")
            elif options.plot == 'violin':
                if args.violin_mode == 'halfhalf':
                    side = 'negative' if inputTraceIndex % 2 == 0 else 'positive'
                elif args.violin_mode[:4] == 'half':
                    side = 'positive'
                else:
                    side = 'both'
                markercolour = options.line_colour
                plotScript.write(f"""
fig.add_trace(go.Violin(
    name='{traceName}',
    legendgroup='{traceName}',
    showlegend={showInLegend},
    scalegroup='trace{inputTraceIndex}',
    y={ydata},
    x={xdata},
{commentColour}    fillcolor='{fillcolour.hex}',
{commentColour}    line_color='{options.line_colour.hex}',
    line_width={options.line_width},
    side='{side}',
    width={options.violin_width},
    scalemode='width',
    points=False,
    orientation='{'v' if options.vertical else 'h'}',
    opacity={options.opacity},
), col={options.col}, row={options.row}, secondary_y={options.y_secondary})
""")

            traceIndex += 1
            frameTraceIndex += 1
            inputTraceIndex += 1
            colourIndex += 1 if args.per_trace_colours else 0
        inputFrameIndex += 1
        frameIndex += 1
        colourIndex += 1 if args.per_frame_colours else 0
    colourIndex += 1 if args.per_input_colours else 0
    # Find out if we need left, right and bottom margin:
    if defaultLeftMargin is None and options.col == 1 and options.y_title and not options.y_secondary:
        defaultLeftMargin = True
    if defaultTopMargin is None and options.row == 1 and options.title:
        defaultTopMargin = True
    if defaultRightMargin is None and options.col + options.colspan - 1 == subplotGrid[0]['max'] and options.y_title is not None and options.y_secondary:
        defaultRightMargin = True
    if defaultBottomMargin is None and options.row + options.rowspan - 1 == subplotGrid[1]['max'] and options.x_title is not None:
        defaultBottomMargin = True

    plotScript.write("\n\n")
    plotScript.write("# Subplot specific options:\n")
    plotScript.write(f"fig.update_yaxes(type='{options.y_type}', rangemode='{options.y_range_mode}', col={options.col}, row={options.row}, secondary_y={options.y_secondary})\n")
    plotScript.write(f"fig.update_xaxes(type='{options.x_type}', rangemode='{options.x_range_mode}', col={options.col}, row={options.row})\n")
    plotScript.write(f"fig.update_yaxes(showline=False, linewidth=0, linecolor='rgba(0,0,0,0)', col={options.col}, row={options.row}, secondary_y={options.y_secondary})\n")
    plotScript.write(f"{'# ' if not options.y_hide else ''}fig.update_yaxes(visible=False, showticklabels=False, showgrid=True, zeroline=False, row={options.row}, col={options.col}, secondary_y={options.y_secondary})\n")
    plotScript.write(f"{'# ' if not options.x_hide else ''}fig.update_xaxes(visible=False, showticklabels=False, showgrid=True, zeroline=False, row={options.row}, col={options.col})\n")
    plotScript.write(f"{'# ' if options.y_title is None else ''}fig.update_yaxes(title_text='{options.y_title}', col={options.col}, row={options.row}, secondary_y={options.y_secondary})\n")
    plotScript.write(f"{'# ' if options.x_title is None else ''}fig.update_xaxes(title_text='{options.x_title}', col={options.col}, row={options.row})\n")
    plotScript.write(f"fig.update_yaxes(tickformat='{options.y_tick_format}', ticksuffix='{options.y_tick_suffix}', tickprefix='{options.y_tick_prefix}', col={options.col}, row={options.row}, secondary_y={options.y_secondary})\n")
    plotScript.write(f"fig.update_xaxes(tickformat='{options.x_tick_format}', ticksuffix='{options.x_tick_suffix}', tickprefix='{options.x_tick_prefix}', col={options.col}, row={options.row})\n")
    plotScript.write(f"# fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', col={options.col}, row={options.row}, secondary_y={options.y_secondary})\n")
    plotScript.write(f"# fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', col={options.col}, row={options.row})\n")
    if options.y_range_from is not None or options.y_range_to is not None:
        options.y_range_from = options.y_range_from if options.y_range_from is not None else plotRange[1]['min']
        options.y_range_to = options.y_range_to if options.y_range_to is not None else plotRange[1]['max']
        plotScript.write(f"fig.update_yaxes(range=[{options.y_range_from}, {options.y_range_to}], col={options.col}, row={options.row}, secondary_y={options.y_secondary})\n")
    if options.x_range_from is not None or options.x_range_to is not None:
        options.x_range_from = options.x_range_from if options.x_range_from is not None else plotRange[0]['min']
        options.x_range_to = options.x_range_to if options.x_range_to is not None else plotRange[0]['max']
        plotScript.write(f"fig.update_xaxes(range=[{options.x_range_from}, {options.x_range_to}], col={options.col}, row={options.row})\n")
    plotScript.write(f"# fig.update_yaxes(range=[{plotRange[0]['min']}, {plotRange[0]['max']}], col={options.col}, row={options.row}, secondary_y={options.y_secondary})\n")
    plotScript.write(f"# fig.update_xaxes(range=[{plotRange[1]['min']}, {plotRange[1]['max']}], col={options.col}, row={options.row}, secondary_y={options.y_secondary})\n")
    plotScript.write(f"fig.update_xaxes(tickmode='{options.x_tickmode}', ticks='{options.x_ticks}', nticks={options.x_nticks}, tick0='{options.x_tick0}', dtick='{options.x_dtick}', tickvals={options.x_tickvals}, ticktext={options.x_ticktext}, tickangle={options.x_tickangle}, col={options.col}, row={options.row})\n")
    plotScript.write(f"fig.update_yaxes(tickmode='{options.y_tickmode}', ticks='{options.y_ticks}', nticks={options.y_nticks}, tick0='{options.y_tick0}', dtick='{options.y_dtick}', tickvals={options.y_tickvals}, ticktext={options.y_ticktext}, tickangle={options.y_tickangle}, col={options.col}, row={options.row}, secondary_y={options.y_secondary})\n")

    plotScript.write("\n\n")

plotScript.write("\n\n")

if (args.violin_mode == 'halfgroup'):
    args.violin_mode = 'group'
elif (args.violin_mode[:4] == 'half'):
    args.violin_mode = 'overlay'

plotScript.write('# Global modes and paramters:\n')
plotScript.write(f"fig.update_layout(title={args.master_title})\n")
plotScript.write(f"fig.update_layout(barmode='{args.bar_mode}', boxmode='{args.box_mode}', violinmode='{args.violin_mode}')\n")
plotScript.write(f"fig.update_layout(bargap={args.bar_gap}, bargroupgap={args.bar_group_gap}, boxgap={args.box_gap}, boxgroupgap={args.box_group_gap}, violingap={args.violin_gap}, violingroupgap={args.violin_group_gap})\n")

plotScript.write(f"\n# Layout Legend\n")
plotScript.write(f"fig.update_layout(showlegend={args.legend_show})\n")
plotScript.write(f"{'# ' if args.legend_y_anchor is None else ''}fig.update_layout(legend_yanchor='{'auto' if args.legend_y_anchor is None else args.legend_y_anchor}')\n")
plotScript.write(f"{'# ' if args.legend_x_anchor is None else ''}fig.update_layout(legend_xanchor='{'auto' if args.legend_x_anchor is None else args.legend_x_anchor}')\n")
plotScript.write(f"fig.update_layout(legend=dict(x={args.legend_x}, y={args.legend_y}, orientation='{'v' if args.legend_vertical else 'h'}', bgcolor='rgba(255,255,255,0)'))\n")

plotScript.write(f"\n# Layout Plot and Background\n")
plotScript.write(f"{commentColour}fig.update_layout(paper_bgcolor='rgba(255, 255, 255, 0)', plot_bgcolor='rgba(255, 255, 255, 0)')\n")

args.margin_b = args.margin_b if args.margin_b is not None else args.margins if args.margins is not None else None if defaultBottomMargin else 0
args.margin_l = args.margin_l if args.margin_l is not None else args.margins if args.margins is not None else None if defaultLeftMargin else 0
args.margin_r = args.margin_r if args.margin_r is not None else args.margins if args.margins is not None else None if defaultRightMargin else 0
args.margin_t = args.margin_t if args.margin_t is not None else args.margins if args.margins is not None else None if defaultTopMargin else 0
args.margin_pad = args.margin_pad if args.margin_pad is not None else args.margins if args.margins is not None else None if defaultPadMargin else 0

plotScript.write(f"fig.update_layout(margin=dict(t={args.margin_t}, l={args.margin_l}, r={args.margin_r}, b={args.margin_b}, pad={args.margin_pad}))\n")

plotScript.write(f"\n# Plot Font\n")
plotScript.write(f"fig.update_layout(font=dict(family='{args.font_family}', size=args.font_size))\n")
plotScript.write(f"{commentColour}fig.update_layout(font=dict(color='{args.font_colour.hex}'))\n")


plotScript.write("""
# Execute addon file if found
filename, fileext = os.path.splitext(__file__)
if (os.path.exists(f'{filename}_addon{fileext}')):
    exec(open(f'{filename}_addon{fileext}').read())""")

if args.orca is None:
    orcaSearchPath = ['/opt/plotly-orca/orca', '/opt/plotly/orca', 'orca']
    for executable in orcaSearchPath:
        args.orca = shutil.which(executable)
        if args.orca is not None:
            break

plotScript.write(f"""

if args.orca is None and os.getenv('PLOTLY_ORCA') is not None:
    args.orca = os.getenv('PLOTLY_ORCA')""")

if args.orca is not None:
    plotScript.write(f"""

# An initial orca version is provided by the plot author
if args.orca is None:
    args.orca = '{args.orca}'""")

plotScript.write(f"""

if args.browser:
    fig.show()
if len(args.output) > 0:
    if not args.quiet:
        openWith = None
        for app in ['xdg-open', 'open', 'start']:
            if shutil.which(app) is not None:
                openWith = app
                break
        if openWith is None:
            args.quiet = True

    orca = checkOrca(args.orca);
    for output in args.output:
        exportFigure(fig, args.width, args.height, output, args.orca)
        print(f'Saved to {{output}}')
        if not args.quiet:
            try:
                subprocess.check_call([openWith, output], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            except Exception:
                print(f'Could not open {{output}}!')
""")

plotScript.close()
if args.browser or len(args.output) > 0:
    cmdLine = ['python3', plotScriptName]
    if args.browser:
        cmdLine.append('--browser')
    if args.quiet:
        cmdLine.append('--quiet')
    subprocess.check_call(cmdLine, cwd=scriptContext)

if not args.script:
    os.close(plotFd)
    os.remove(plotScriptName)
elif not args.quiet:
    print(f"Plot script saved to {plotScriptName}")
