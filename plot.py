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
import subprocess
import statistics
import sys
import pickle
import copy
import textwrap
import shutil
import xopen
import seaborn


def isFloat(val):
    if val is None:
        return False
    try:
        float(val)
        return True
    except ValueError:
        return False


class OrderedAction(argparse.Action):
    def __init__(self, *args, ordered=True, sub_action='store', **kwargs):
        super().__init__(*args, **kwargs)
        self.action = sub_action
        self.ordered = ordered

    def __call__(self, parser, namespace, values, option_string=None):
        _action = parser._registry_get('action', self.action, self.action)(self.option_strings, self.dest)
        _action(parser, namespace, values, option_string)
        if 'ordered_args' not in namespace:
            setattr(namespace, 'ordered_args', [])
        if self.ordered:
            previous = namespace.ordered_args
            if (self.action == 'append'):
                for i, (k, v) in enumerate(previous):
                    if k == self.dest:
                        previous[i] = (k, getattr(namespace, self.dest))
                        break
            else:
                previous.append((self.dest, getattr(namespace, self.dest)))
            setattr(namespace, 'ordered_args', previous)


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
            setattr(nspace, 'ordered_args', [])
        items.append({'value': values, 'args': nspace})


class ChildAction(argparse.Action):
    _adjusting_defaults = {}

    def __init__(self, *args, parent, sub_action='store', sticky_default=False, ordered=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.dest, self.name = parent.dest, self.dest
        self.sticky_default = sticky_default

        self.sub_action = sub_action
        self.action = OrderedAction
        self.ordered = ordered

        self.parent = parent
        parent.children.append(self)

    def __call__(self, parser, namespace, values, option_string=None):
        ChildAction._adjusting_defaults[self.name] = True if self.action == 'store_true' else values
        items = getattr(namespace, self.dest)
        try:
            lastParentNamespace = items[-1]['args']
        except Exception:
            if (self.sticky_default):
                raise Exception(f'parameter --{self.name} can only be used after --{self.parent.dest}!') from None
                exit(1)
            return
        _action = parser._registry_get('action', self.action, self.action)(self.option_strings, self.name, sub_action=self.sub_action, ordered=self.ordered)
        _action(parser, lastParentNamespace, values, option_string)


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


class DataframeActions:
    def transpose(dataframe):
        return dataframe.transpose()

    def dropNaN(dataframe, dropAny=False):
        return dataframe.dropna(how='any' if dropAny else 'all', axis=0).dropna(how='any' if dropAny else 'all', axis=1)

    def dropColumnsByIdx(dataframe, columns):
        if not isinstance(columns, list):
            columns = [columns]
        columns = [x if x >= 0 else x + dataframe.shape[1] for x in columns]
        filterColumns = numpy.array([False if i in columns else True for i in range(dataframe.shape[1])])
        return dataframe.loc[:, filterColumns]

    def filterColumnsByIdx(dataframe, columns):
        if not isinstance(columns, list):
            columns = [columns]
        validColumns = range(-dataframe.shape[1], dataframe.shape[1])
        filterColumns = numpy.array([i for i in columns if i in validColumns])
        return dataframe.iloc[:, filterColumns]

    def dropRowsByIdx(dataframe, rows):
        if not isinstance(rows, list):
            rows = [rows]
        rows = [x if x >= 0 else x + dataframe.shape[0] for x in rows]
        filterRows = numpy.array([False if i in rows else True for i in range(dataframe.shape[0])])
        return dataframe.iloc[filterRows, :]

    def filterRowsByIdx(dataframe, rows):
        if not isinstance(rows, list):
            rows = [rows]
        validRows = range(-dataframe.shape[0], dataframe.shape[0])
        filterRows = numpy.array([i for i in rows if i in validRows])
        return dataframe.iloc[filterRows, :]

    def getColumnIdx(dataframe, columns, mode='all', ignore_errors=False):
        if not isinstance(columns, list):
            columns = [columns]
        columnIdx = []
        for col in columns:
            if col not in dataframe.columns:
                if not ignore_errors:
                    raise Exception(f'Could not find column name {col}')
            else:
                selection = reversed(list(enumerate(dataframe.columns.tolist()))) if mode == 'last' else enumerate(dataframe.columns.tolist())
                for i, fcol in selection:
                    if fcol == col:
                        columnIdx.append(i)
                        if mode != 'all':
                            break
        return columnIdx

    def getRowIdx(dataframe, rows, mode='all', ignore_errors=False):
        if not isinstance(rows, list):
            rows = [rows]
        rowIdx = []
        for row in rows:
            if row not in dataframe.index:
                if not ignore_errors:
                    raise Exception(f'Could not find row name {row}')
            else:
                selection = reversed(list(enumerate(dataframe.index.tolist()))) if mode == 'last' else enumerate(dataframe.index.tolist())
                for i, frow in selection:
                    if frow == row:
                        rowIdx.append(i)
                        if mode != 'all':
                            break
        return rowIdx

    def setIndexColumnByIDx(dataframe, colIdx):
        return dataframe.set_index(dataframe.iloc[:, colIdx])

    def renameColumns(dataframe, names):
        if not isinstance(names, list):
            names = [names]
        names = [float(x) if isFloat(x) else x for x in names]
        dataframe.columns = (names + dataframe.columns.to_list()[len(names):])[:len(dataframe.columns)]
        return dataframe

    def renameRows(dataframe, names):
        if not isinstance(names, list):
            names = [names]
        names = [float(x) if isFloat(x) else x for x in names]
        dataframe.index = (names + dataframe.index.to_list()[len(names):])[:len(dataframe.index)]
        return dataframe

    def sortColumns(dataframe, function='mean', order='asc'):
        sortKey = getattr(dataframe, function)(axis=0)
        sortKey.reset_index(drop=True, inplace=True)
        return dataframe.iloc[:, sortKey.sort_values(ascending=(order == 'asc')).index]

    def sortRows(dataframe, function='mean', order='asc'):
        sortKey = getattr(dataframe, function)(axis=1)
        return dataframe.reindex(sortKey.sort_values(ascending=(order == 'asc')).index)

    def reverseColumns(dataframe):
        return dataframe.iloc[::, ::-1]

    def reverseRows(dataframe):
        return dataframe.iloc[::-1]

    def sortColumnsByRowIdx(dataframe, rowIdx, order='asc'):
        if rowIdx == 'columns':
            sortKey = sorted(range(len(dataframe.columns)), key=lambda k: dataframe.columns[k], reverse=(order != 'asc'))
            return dataframe.iloc[:, sortKey]
        else:
            sortKey = dataframe.iloc[int(rowIdx)]
            sortKey.reset_index(drop=True, inplace=True)
            return dataframe.iloc[:, sortKey.sort_values(ascending=(order == 'asc')).index]

    def sortRowsByColumnIdx(dataframe, colIdx, order='asc'):
        if colIdx == 'index':
            return dataframe.reindex(dataframe.index.sort_values(ascending=(order == 'asc')))
        else:
            return dataframe.reindex(dataframe.iloc[:, int(colIdx)].sort_values(ascending=(order == 'asc')).index)

    def addConstant(dataframe, constant):
        return dataframe + constant

    def scaleConstant(dataframe, constant):
        return dataframe * constant

    def normaliseToConstant(dataframe, constant):
        return dataframe / constant

    def abs(dataframe):
        return dataframe.abs()

    def applyOnRowIdx(dataframe, rowIdx, function='abs', inclusive=True):
        with pandas.option_context('mode.chained_assignment', None):
            functionMap = {'zero': 0, 'one': 1, 'nan': numpy.nan}
            if function in ['abs', 'cumsum', 'cummax', 'cummin', 'cumprod', 'rank']:
                dataframe.iloc[rowIdx, :] = getattr(dataframe.iloc[rowIdx, :], function)()
            elif function in functionMap:
                dataframe.iloc[rowIdx, :] = functionMap[function]
            elif function == 'set':
                dataframe = dataframe.apply(lambda _: dataframe.iloc[rowIdx, :], axis=1)
            else:
                applyRow = dataframe.iloc[rowIdx, :].apply(pandas.to_numeric, errors='coerce')
                dataframe = dataframe.apply(lambda row: getattr(row, function)(applyRow), axis=1)
                if not inclusive:
                    dataframe.iloc[rowIdx, :] = applyRow
            return dataframe

    def applyOnColumnIdx(dataframe, columnIdx, function='abs', inclusive=True):
        with pandas.option_context('mode.chained_assignment', None):
            functionMap = {'zero': 0, 'one': 1, 'nan': numpy.nan}
            if function in ['abs', 'cumsum', 'cummax', 'cummin', 'cumprod', 'rank']:
                dataframe.iloc[:, columnIdx] = getattr(dataframe.iloc[:, columnIdx], function)()
            elif function in functionMap:
                dataframe.iloc[:, columnIdx] = functionMap[function]
            elif function == 'set':
                dataframe = dataframe.apply(lambda _: dataframe.iloc[:, colIdx], axis=1)
            else:
                applyColumn = dataframe.iloc[:, columnIdx].apply(pandas.to_numeric, errors='coerce')
                dataframe = dataframe.apply(lambda col: getattr(col, function)(applyColumn), axis=0)
                if not inclusive:
                    dataframe.iloc[:, columnIdx] = applyColumn
            return dataframe

    def addRow(dataframe, name, function='mean', where='back'):
        if function in ['nan', 'zero', 'one']:
            element = numpy.nan if function == 'nan' else 0 if function == 'zero' else 1
            newRow = pandas.DataFrame([[element] * dataframe.shape[1]], index=[name], columns=dataframe.columns)
        else:
            newRow = getattr(dataframe.apply(pandas.to_numeric, errors='coerce'), function)(axis=0)
            newRow = newRow.to_frame(name).transpose()
        return pandas.concat([dataframe, newRow], axis=0) if where == 'back' else pandas.concat([newRow, dataframe], axis=0)

    def addColumn(dataframe, name, function='mean', where='back'):
        if function in ['nan', 'zero', 'one']:
            element = numpy.nan if function == 'nan' else 0 if function == 'zero' else 1
            newCol = pandas.Series(data=[element] * dataframe.shape[0], name=name, index=dataframe.index)
        else:
            newCol = getattr(dataframe.apply(pandas.to_numeric, errors='coerce'), function)(axis=1)
            newCol.name = name
        return pandas.concat([dataframe, newCol], axis=1) if where == 'back' else pandas.concat([newCol, dataframe], axis=1)

    def groupByColumnIdx(dataframe, columnIdx, function='sum'):
        if columnIdx == 'index':
            return getattr(dataframe.groupby(dataframe.index, axis=0), function)()
        else:
            return getattr(dataframe.groupby(dataframe.iloc[:, int(columnIdx)], as_index=False, axis=0), function)()

    def groupByRowIdx(dataframe, rowIdx, function='sum'):
        if rowIdx == 'columns':
            return getattr(dataframe.groupby(dataframe.columns, axis=1), function)()
        else:
            return getattr(dataframe.groupby(dataframe.iloc[int(rowIdx), :], axis=0), function)()

    def joinFrames(dataframes, function='index'):
        joinedFrame = None
        for frame in dataframes:
            joinedFrame = frame if joinedFrame is None else pandas.concat([joinedFrame, frame], axis=(1 if function == 'index' else 0), join='outer', verify_integrity=False, copy=True)
        return joinedFrame

    def splitFramesByRowIdx(dataframes, rowIdx):
        newFrames = []
        for frame in dataframes:
            if rowIdx == 'columns':
                for v in frame.columns.unique():
                    columnIdx = DataframeActions.getColumnIdx(frame, v, 'all', False)
                    newFrames.append(DataframeActions.filterColumnsByIdx(frame, columnIdx))
            else:
                rowIdx = int(rowIdx)
                for v in frame.iloc[rowIdx, :].unique():
                    newFrames.append(frame[frame.iloc[rowIdx, :] == v])
        return newFrames

    def splitFramesByColumnIdx(dataframes, columnIdx):
        newFrames = []
        for frame in dataframes:
            if columnIdx == 'index':
                for v in frame.index.unique():
                    rowIdx = DataframeActions.getRowIdx(frame, v, 'all', False)
                    newFrames.append(DataframeActions.filterRowsByIdx(frame, rowIdx))
            else:
                columnIdx = int(columnIdx)
                for v in frame.iloc[:, columnIdx].unique():
                    if v != v:
                        newFrames.append(frame[frame.iloc[:, columnIdx].isna()])
                    else:
                        newFrames.append(frame[frame.iloc[:, columnIdx] == v])
        return newFrames

    def printFrames(filenames, dataframe, frameIndex, frameCount, precision=None):
        if not isinstance(filenames, list):
            filenames = [filenames]
        consoleWidth = shutil.get_terminal_size((80, 40))
        pSep = '---'
        if len(filenames) > 0:
            pFiles = f"File: {', '.join(filenames)}"
            pSep = '-' * min(consoleWidth.columns, len(pFiles))
        print(pSep + f'\nFrame: {frameIndex+1}/{frameCount}')
        if len(filenames) > 0:
            print(textwrap.fill(pFiles, width=consoleWidth.columns, subsequent_indent=' '))
        print(pSep)
        with pandas.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', consoleWidth.columns, 'display.max_columns', None, 'display.float_format', None if precision is None else f'{{:.{precision}f}}'.format):
            print(dataframe)
        print(pSep)

    def framesToCSV(dataframes, filenames=['stdout'], separator=None, quiet=False, precision=None):
        _index = 1
        for (frame, filename) in zip(dataframes, filenames):
            sep = ';' if separator is None else separator
            if filename.endswith('.tsv'):
                sep = '\t'
            elif filename.endswith('.csv'):
                sep = ';'
            fFile = sys.stdout if filename == 'stdout' else sys.stderr if filename == 'stderr' else xopen.xopen(filename, 'w')
            frame.to_csv(fFile, sep=sep, na_rep='NaN', float_format=None if precision is None else f'%.{precision}f')
            if (fFile != sys.stdout and fFile != sys.stdout):
                fFile.close()
            if not quiet and not fFile == sys.stdout:
                print(f'Frame {_index + 1}/{len(dataframes)} saved to {filename}')
            _index += 1

    def framesToPickle(dataframes, filename, quiet=False):
        fFile = sys.stdout.buffer if filename == 'stdout' else sys.stderr.buffer if filename == 'stderr' else xopen.xopen(filename, 'wb')
        pickle.dump(dataframes, fFile, pickle.HIGHEST_PROTOCOL)
        if not quiet and not fFile == sys.stdout.buffer:
            print(f'Dataframes saved to {filename}')
        if (fFile != sys.stdout.buffer and fFile != sys.stdout.buffer):
            fFile.close()


considerAsNaN = ['nan', 'none', 'null', 'zero', 'nodata', '']
detectDelimiter = ['\t', ';', ' ', ',']

traceSpecialColumns = ['error', 'error-', 'error+', 'offset', 'label', 'colour']
frameSpecialColumns = ['category']
allSpecialColumns = traceSpecialColumns + frameSpecialColumns

parser = argparse.ArgumentParser(description="Visualize your data the easy way")
# Global Arguments

parserFileOptions = parser.add_argument_group('file parsing options')

inputFileArgument = parser.add_argument('-i', '--input', type=str, help="input file to parse", nargs="+", action=ParentAction, required=True)
# Per File Parsing Arguments
parserFileOptions.add_argument("--special-column-start", help="special columns (or ignore columns) starting with (default %(default)s)", type=str, default='_', sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--ignore-line-start", help="ignores lines starting with (default %(default)s)", type=str, default='#', sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--separator", help="data delimiter (auto detected by default)", type=str, default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--no-columns", help="do not use a column row", default=False, sticky_default=True, nargs=0, sub_action="store_true", action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--no-index", help="do not use a index column", default=False, sticky_default=True, nargs=0, sub_action="store_true", action=ChildAction, parent=inputFileArgument)


parserFileOptions.add_argument("--index-icolumn", help="set index column after index", type=int, sticky_default=True, choices=Range(None, None), default=None, action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--index-column", help="set index column", default=None, type=str, sticky_default=True, action=ChildAction, parent=inputFileArgument)

parserFileOptions.add_argument("--select-mode", help="select row/columns after policy (default %(default)s)", type=str, default='all', choices=['all', 'first', 'last'], action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--ignore-icolumns", help="ignore these column indexes", type=int, default=[], sticky_default=True, choices=Range(None, None), nargs='+', action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--ignore-columns", help="ignore these columns", type=str, default=[], sticky_default=True, nargs='+', action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--ignore-irows", help="ignore these row indexes", type=int, default=[], sticky_default=True, choices=Range(None, None), nargs='+', action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--ignore-rows", help="ignore these rows", type=str, default=[], sticky_default=True, nargs='+', action=ChildAction, parent=inputFileArgument)

parserFileOptions.add_argument("--select-irows", help="select these row indexes", type=int, default=[], sticky_default=True, choices=Range(None, None), nargs='+', action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--select-rows", help="select these rows", type=str, default=[], sticky_default=True, nargs='+', action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--select-icolumns", help="select these column indexes", type=int, default=[], sticky_default=True, choices=Range(None, None), nargs='+', action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--select-columns", help="select these columns", type=str, default=[], sticky_default=True, nargs='+', action=ChildAction, parent=inputFileArgument)

parserFileOptions.add_argument("--sort-order", help="sort rows after or column in this order (default %(default)s)", default='asc', choices=['asc', 'desc'], action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--sort-function", help="sort rows after function or column (default %(default)s)", default='mean', choices=['mean', 'median', 'std', 'min', 'max'], action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--sort-columns", help="sort columns", default=False, sub_action="store_true", nargs=0, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--sort-by-irow", help="sort column after this row index", type=str, default=None, choices=Range(None, None, ['columns']), sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--sort-by-row", help="sort column after this row", type=str, default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--sort-rows", help="sort rows", default=False, sub_action="store_true", nargs=0, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--sort-by-icolumn", help="sort rows after this column index", type=str, default=None, choices=Range(None, None, ['index']), sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--sort-by-column", help="sort rows after this column", type=str, default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--reverse-columns", help="reverse columns order", default=False, sub_action="store_true", nargs=0, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--reverse-rows", help="reverse row order", default=False, sub_action="store_true", nargs=0, sticky_default=True, action=ChildAction, parent=inputFileArgument)

parserFileOptions.add_argument("--data-scale", help="scales data (default %(default)s)", type=float, default=1, choices=Range(None, None), sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--data-offset", help="offsets data (default %(default)s)", type=float, default=0, choices=Range(None, None), sticky_default=True, action=ChildAction, parent=inputFileArgument)

parserFileOptions.add_argument("--normalise-to", help="normalise data to (default %(default)s)", type=float, default=0, choices=Range(None, None), sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--normalise-to-icolumn", help="normalise to this column index", type=int, default=None, choices=Range(None, None), sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--normalise-to-column", help="normalise to this column", type=str, default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--normalise-to-irow", help="normalise to this row index", type=int, default=None, choices=Range(None, None), sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--normalise-to-row", help="normalise to this row", type=str, default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)

parserFileOptions.add_argument("--add-at", help="add at the front or back of the dataframe", type=str, default='back', choices=['front', 'back'], action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--add-function", help="use this function to compute new row/column", type=str, default='mean', choices=['sum', 'mean', 'median', 'std', 'var', 'sum', 'count', 'skew', 'mad', 'min', 'max', 'nan', 'zero', 'one'], action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--add-column", help="add a new column with name", type=str, default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--add-row", help="add a new row with name", type=str, default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)

parserFileOptions.add_argument("--group-function", help="use this function to compute grouped dataframe", type=str, default='sum', choices=['sum', 'mean', 'median', 'std', 'var', 'sum', 'count', 'skew', 'mad', 'min', 'max'], action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--group-by-icolumn", help="group by this column index", type=str, default=0, choices=Range(None, None, ['index']), sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--group-by-column", help="group by this column", type=str, default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--group-by-irow", help="group by this row index", type=str, default=0, choices=Range(None, None, ['columns']), sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--group-by-row", help="group by this row", type=str, default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)

parserFileOptions.add_argument("--abs", help="convert all values to absolute values", type=str, default=False, nargs=0, sub_action="store_true", sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--apply-mode", help="convert all values to absolute values", type=str, default='inclusive', choices=['exclusive', 'inclusive'], action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--apply-function", help="use this function to compute new row/column", type=str, default='mean', choices=['add', 'radd', 'sub', 'rsub', 'mul', 'rmul', 'div', 'rdiv', 'mod', 'rmod', 'pow', 'rpow', 'cumsum', 'cummax', 'cummin', 'cumprod', 'rank', 'nan', 'zero', 'one', 'abs', 'set'], action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--apply-icolumns", help="compute function on multiple column indexes", type=str, default=None, nargs='+', sticky_default=True, choices=Range(None, None, ['all']), action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--apply-irows", help="compute function on multiple row indexes", type=str, default=None, nargs='+', sticky_default=True, choices=Range(None, None, ['all']), action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--apply-columns", help="compute function on multiple columns", type=str, default=None, nargs='+', sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--apply-rows", help="compute function on multiple rows", type=str, default=None, nargs='+', sticky_default=True, action=ChildAction, parent=inputFileArgument)

parserFileOptions.add_argument("--column-names", help="rename columns", type=str, sticky_default=True, default=[], nargs='+', action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--row-names", help="rename rows", type=str, sticky_default=True, default=[], nargs='+', action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--drop-nan", help="dropping rows/columns that are completely empty", sticky_default=True, default=False, nargs=0, sub_action="store_true", action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--drop-any-nan", help="dropping rows/columns that contain empty values", sticky_default=True, default=False, nargs=0, sub_action="store_true", action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--transpose", help="transpose data", default=False, sticky_default=True, nargs=0, sub_action="store_true", action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--print", help="print out each parsed input file", default=False, nargs=0, sub_action="store_true", action=ChildAction, parent=inputFileArgument)

parserFileOptions.add_argument("--join", help="outer join input files on columns or index", default='none', choices=['none', 'index', 'columns'], sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--split-icolumn", help="split frame along this column index ('_index' splits by index)", type=str, sticky_default=True, choices=Range(None, None, ['index']), default=None, action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--split-column", help="split frame along this column", type=str, sticky_default=True, default=None, action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--split-irow", help="split frame along this row index ('_columns' splits by columns)", type=str, sticky_default=True, choices=Range(None, None, ['columns']), default=None, action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--split-row", help="split frame along this row", type=str, sticky_default=True, default=None, action=ChildAction, parent=inputFileArgument)

parserFileOptions.add_argument("--focus-frames", help="set the frame focus for file options (default %(default)s)", type=str, default='all', nargs='+', choices=Range(None, None, ['all']), sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--defocus-frames", help="remove frames from focus for file options", type=int, default=None, nargs='+', choices=Range(None, None), sticky_default=True, action=ChildAction, parent=inputFileArgument)

parserFileOptions.add_argument("--output-precision", help="set explicit output prevision for text and console output (default %(default)s)", type=str, default='default', choices=Range(0, None, ['default']), action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--file", help="save data frames to text files (one file per frame)", default=None, type=str, nargs='+', sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserFileOptions.add_argument("--pickle", help="pickle data frames to file (one file containing all frames)", default=None, type=str, sticky_default=True, action=ChildAction, parent=inputFileArgument)

# Per File Plotting Arguments:
parserPlotOptions = parser.add_argument_group('plot options')
parserPlotOptions.add_argument('--plot', choices=['line', 'bar', 'box', 'violin'], help='plot type', default='line', action=ChildAction, parent=inputFileArgument)
parserPlotOptions.add_argument("--orientation", help="set plot orientation", default='auto', choices=['vertical', 'v', 'horizontal', 'h', 'auto'], action=ChildAction, parent=inputFileArgument)
parserPlotOptions.add_argument("--title", help="subplot title", type=str, default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserPlotOptions.add_argument("--use-name", help="use name for traces", type=str, default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserPlotOptions.add_argument('--row', type=int, choices=Range(1, None), help='subplot row (default %(default)s)', default=1, action=ChildAction, parent=inputFileArgument)
parserPlotOptions.add_argument('--rowspan', type=int, choices=Range(1, None), help='subplot rowspan (default %(default)s)', default=1, action=ChildAction, parent=inputFileArgument)
parserPlotOptions.add_argument('--col', type=int, choices=Range(1, None), help='subplot column (default %(default)s)', default=1, action=ChildAction, parent=inputFileArgument)
parserPlotOptions.add_argument('--colspan', type=int, choices=Range(1, None), help='subplot columnspan (default %(default)s)', default=1, action=ChildAction, parent=inputFileArgument)

parserPlotOptions.add_argument("--error", help="show error markers in plot (need to be supplied by data)", default='hide', choices=['show', 'hide'], action=ChildAction, parent=inputFileArgument)
parserPlotOptions.add_argument("--trace-names", help="set individual trace names", default=[], sticky_default=True, type=str, nargs='+', action=ChildAction, parent=inputFileArgument)
parserPlotOptions.add_argument("--trace-colours", help="define explicit trace colours", default=[], nargs='+', type=str, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserPlotOptions.add_argument("--line-width", help="set line width (default %(default)s)", type=int, default=1, choices=Range(0,), action=ChildAction, parent=inputFileArgument)
parserPlotOptions.add_argument("--line-colour", help="set line colour  (default %(default)s) (line charts are using just colour)", type=str, default='#222222', action=ChildAction, parent=inputFileArgument)
parserPlotOptions.add_argument("--opacity", help="colour opacity (default 0.8 for overlay modes, else 1.0)", choices=Range(0, 1, ['auto']), action=ChildAction, parent=inputFileArgument)
parserPlotOptions.add_argument("--offsetgroups", help="set explicit offsetgroups for e.g. bar charts", type=int, default='auto', nargs='+', choices=Range(0, None, ['auto']), sticky_default=True, action=ChildAction, parent=inputFileArgument)

parserLinePlotOptions = parser.add_argument_group('line plot options')
parserLinePlotOptions.add_argument("--line-mode", choices=['none', 'lines', 'markers', 'text', 'lines+markers', 'lines+text', 'markers+text', 'lines+markers+text'], help="choose linemode (default %(default)s)", default='lines', action=ChildAction, parent=inputFileArgument)
parserLinePlotOptions.add_argument('--line-fill', choices=['none', 'tozeroy', 'tozerox', 'tonexty', 'tonextx', 'toself', 'tonext'], help='fill line area (default %(default)s)', default='none', action=ChildAction, parent=inputFileArgument)
parserLinePlotOptions.add_argument('--line-stack', help='stack line input traces (default %(default)s)', default=False, sticky_default=True, nargs=0, sub_action="store_true", action=ChildAction, parent=inputFileArgument)
parserLinePlotOptions.add_argument('--line-shape', choices=['linear', 'spline', 'hv', 'vh', 'hvh', 'vhv'], help='choose line shape (default %(default)s)', default='linear', action=ChildAction, parent=inputFileArgument)
parserLinePlotOptions.add_argument('--line-dash', choices=['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot'], help='choose line dash (default %(default)s)', default='solid', action=ChildAction, parent=inputFileArgument)
parserLinePlotOptions.add_argument('--line-markers', choices=['circle', 'circle-open', 'circle-dot', 'circle-open-dot', 'square', 'square-open', 'square-dot', 'square-open-dot', 'diamond', 'diamond-open', 'diamond-dot', 'diamond-open-dot', 'cross', 'cross-open', 'cross-dot', 'cross-open-dot', 'x', 'x-open', 'x-dot', 'x-open-dot', 'triangle-up', 'triangle-up-open', 'triangle-up-dot', 'triangle-up-open-dot', 'triangle-down', 'triangle-down-open', 'triangle-down-dot', 'triangle-down-open-dot', 'triangle-left', 'triangle-left-open', 'triangle-left-dot', 'triangle-left-open-dot', 'triangle-right', 'triangle-right-open', 'triangle-right-dot', 'triangle-right-open-dot', 'triangle-ne', 'triangle-ne-open', 'triangle-ne-dot', 'triangle-ne-open-dot', 'triangle-se', 'triangle-se-open', 'triangle-se-dot', 'triangle-se-open-dot', 'triangle-sw', 'triangle-sw-open', 'triangle-sw-dot', 'triangle-sw-open-dot', 'triangle-nw', 'triangle-nw-open', 'triangle-nw-dot', 'triangle-nw-open-dot', 'pentagon', 'pentagon-open', 'pentagon-dot', 'pentagon-open-dot', 'hexagon', 'hexagon-open', 'hexagon-dot', 'hexagon-open-dot', 'hexagon2', 'hexagon2-open', 'hexagon2-dot', 'hexagon2-open-dot', 'octagon', 'octagon-open', 'octagon-dot', 'octagon-open-dot', 'star', 'star-open', 'star-dot', 'star-open-dot', 'hexagram', 'hexagram-open', 'hexagram-dot', 'hexagram-open-dot', 'star-triangle-up', 'star-triangle-up-open', 'star-triangle-up-dot', 'star-triangle-up-open-dot', 'star-triangle-down', 'star-triangle-down-open', 'star-triangle-down-dot', 'star-triangle-down-open-dot', 'star-square', 'star-square-open', 'star-square-dot', 'star-square-open-dot', 'star-diamond', 'star-diamond-open', 'star-diamond-dot', 'star-diamond-open-dot', 'diamond-tall', 'diamond-tall-open', 'diamond-tall-dot', 'diamond-tall-open-dot', 'diamond-wide', 'diamond-wide-open', 'diamond-wide-dot', 'diamond-wide-open-dot', 'hourglass', 'hourglass-open', 'bowtie', 'bowtie-open', 'circle-cross', 'circle-cross-open', 'circle-x', 'circle-x-open', 'square-cross', 'square-cross-open', 'square-x', 'square-x-open', 'diamond-cross', 'diamond-cross-open', 'diamond-x', 'diamond-x-open', 'cross-thin', 'cross-thin-open', 'x-thin', 'x-thin-open', 'asterisk', 'asterisk-open', 'hash', 'hash-open', 'hash-dot', 'hash-open-dot', 'y-up', 'y-up-open', 'y-down', 'y-down-open', 'y-left', 'y-left-open', 'y-right', 'y-right-open', 'line-ew', 'line-ew-open', 'line-ns', 'line-ns-open', 'line-ne', 'line-ne-open', 'line-nw', 'line-nw-open'], help='choose line marker (default circle)', default=[], nargs='+', action=ChildAction, parent=inputFileArgument)
parserLinePlotOptions.add_argument('--line-marker-size', help='choose line marker size (default %(default)s)', type=int, default=6, choices=Range(0, None), action=ChildAction, parent=inputFileArgument)
parserLinePlotOptions.add_argument("--line-text-position", choices=["top left", "top center", "top right", "middle left", "middle center", "middle right", "bottom left", "bottom center", "bottom right"], help="choose line text positon (default %(default)s)", default='middle center', action=ChildAction, parent=inputFileArgument)

parserBarPlotOptions = parser.add_argument_group('bar plot options')
parserBarPlotOptions.add_argument("--bar-mode", help="choose barmode (default %(default)s)", choices=['stack', 'group', 'unique_group', 'overlay', 'relative'], default='group')
parserBarPlotOptions.add_argument("--bar-width", help="set explicit bar width", choices=Range(0, None, ['auto']), default='auto', action=ChildAction, parent=inputFileArgument)
parserBarPlotOptions.add_argument("--bar-shift", help="set bar shift", choices=Range(None, None, ['auto']), default='auto', action=ChildAction, parent=inputFileArgument)
parserBarPlotOptions.add_argument("--bar-text-position", help="choose bar text position (default %(default)s)", choices=["inside", "outside", "auto", "none"], default='none', action=ChildAction, parent=inputFileArgument)
parserBarPlotOptions.add_argument("--bar-gap", help="set bar gap (default $(default)s)", choices=Range(0, 1, ['auto']), default='auto')
parserBarPlotOptions.add_argument("--bar-group-gap", help="set bar group gap (default $(default)s)", choices=Range(0, 1), default=0)

parserViolinPlotOptions = parser.add_argument_group('violin plot options')
parserViolinPlotOptions.add_argument("--violin-mode", help="choose violinmode (default %(default)s)", choices=['overlay', 'group', 'halfoverlay', 'halfgroup', 'halfhalf'], default='overlay')
parserViolinPlotOptions.add_argument("--violin-mean", help="choose violin mean (default %(default)s)", choices=['none', 'line', 'box'], default='none', action=ChildAction, parent=inputFileArgument)
parserViolinPlotOptions.add_argument("--violin-points", help="set points mode for (default %(default)s)", type=str, default='none', choices=['all', 'outliers', 'suspectedoutliers', 'none'], action=ChildAction, parent=inputFileArgument)
parserViolinPlotOptions.add_argument("--violin-jitter", help="set jitter for violin points (default %(default)s)", type=float, default=0, choices=Range(0, 1), action=ChildAction, parent=inputFileArgument)
parserViolinPlotOptions.add_argument("--violin-width", help="change violin widths (default %(default)s)", type=float, default=0, choices=Range(0,), action=ChildAction, parent=inputFileArgument)
parserViolinPlotOptions.add_argument("--violin-gap", help="gap between violins (default %(default)s) (not compatible with violin-width)", type=float, default=0.3, choices=Range(0, 1))
parserViolinPlotOptions.add_argument("--violin-group-gap", help="gap between violin groups (default %(default)s) (not compatible with violin-width)", type=float, default=0.3, choices=Range(0, 1))

parserBoxPlotOptions = parser.add_argument_group('box plot options')
parserBoxPlotOptions.add_argument("--box-mode", choices=['overlay', 'group'], help="choose boxmode (default %(default)s)", default='overlay')
parserBoxPlotOptions.add_argument("--box-mean", choices=['none', 'line', 'dot'], help="choose box mean (default %(default)s)", default='dot', action=ChildAction, parent=inputFileArgument)
parserBoxPlotOptions.add_argument("--box-width", help="box width (default %(default)s)", type=float, default=0, choices=Range(0,), action=ChildAction, parent=inputFileArgument)
parserBoxPlotOptions.add_argument("--box-gap", help="gap between boxes (default %(default)s) (not compatible with box-width)", type=float, default=0.3, choices=Range(0, 1))
parserBoxPlotOptions.add_argument("--box-group-gap", help="gap between box groups (default %(default)s) (not compatible with box-width)", type=float, default=0.3, choices=Range(0, 1))

parserPlotAxisOptions = parser.add_argument_group('plot axis options')
parserPlotAxisOptions.add_argument("--y-secondary", help="plot to secondary y-axis", default=False, sticky_default=True, nargs=0, sub_action="store_true", action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--y-title", help="y-axis title", default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--x-title", help="x-axis title", default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--y-title-standoff", help="added margin between tick labels and y-title in px", choices=Range(0,), default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--x-title-standoff", help="added margin between tick labels and x-title in px", choices=Range(0,), default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--y-type", help="choose type for y-axis (default %(default)s)", choices=['-', 'linear', 'log', 'date', 'category'], default='-', action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--x-type", help="choose type for x-axis (default %(default)s)", choices=['-', 'linear', 'log', 'date', 'category'], default='-', action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--y-hide", help="hide y-axis", default=False, sticky_default=True, nargs=0, sub_action="store_true", action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--x-hide", help="hide x-axis", default=False, sticky_default=True, nargs=0, sub_action="store_true", action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--y-range-mode", help="choose range mode for x-axis (default %(default)s)", choices=['normal', 'tozero', 'nonnegative'], default='normal', action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--x-range-mode", help="choose range mode for y-axis (default %(default)s)", choices=['normal', 'tozero', 'nonnegative'], default='normal', action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--y-range-from", help="y-axis start (default %(default)s)", default='auto', sticky_default=True, choices=Range(None, None, ['auto']), action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--x-range-from", help="x-axis start (default %(default)s)", default='auto', sticky_default=True, choices=Range(None, None, ['auto']), action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--y-range-to", help="y-axis end (default %(default)s)", default='auto', sticky_default=True, choices=Range(None, None, ['auto']), action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--x-range-to", help="x-axis start (default %(default)s)", default='auto', sticky_default=True, choices=Range(None, None, ['auto']), action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--y-tick-format", help="set format of y-axis ticks", default='', sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--x-tick-format", help="set format of x-axis ticks", default='', sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--y-tick-suffix", help="add suffix to y-axis ticks", default='', sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--x-tick-suffix", help="add suffix to x-axis ticks", default='', sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--y-tick-prefix", help="add prefix to y-axis ticks", default='', sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--x-tick-prefix", help="add prefix to x-axis ticks", default='', sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--y-ticks", help="how to draw y ticks (default '%(default)s')", choices=['', 'inside', 'outside'], default='', sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--x-ticks", help="how to draw x ticks (default '%(default)s')", choices=['', 'inside', 'outside'], default='', sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--y-tickmode", help="tick mode y-axis (default '%(default)s')", choices=['auto', 'linear', 'array'], default='auto', sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--x-tickmode", help="tick mode x-axis (default '%(default)s')", choices=['auto', 'linear', 'array'], default='auto', sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--y-nticks", help="number of ticks on y-axis (only tick mode auto) (default %(default)s)", choices=Range(0,), default=0, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--x-nticks", help="number of ticks on x-axis (only tick mode auto) (default %(default)s)", choices=Range(0,), default=0, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--y-tick0", help="first tick on y-axis (only tick mode linear) (default %(default)s)", type=str, default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--x-tick0", help="first tick on x-axis (only tick mode linear) (default %(default)s)", type=str, default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--y-dtick", help="tick step on y-axis (only tick mode linear) (default %(default)s)", type=str, default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--x-dtick", help="tick step on x-axis (only tick mode linear) (default %(default)s)", type=str, default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--y-tickvals", help="tick values on y-axis (only tick mode array) (default %(default)s)", default=[], sticky_default=True, nargs='+', action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--x-tickvals", help="tick values on x-axis (only tick mode array) (default %(default)s)", default=[], sticky_default=True, nargs='+', action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--y-ticktext", help="tick text on y-axis (only tick mode array) (default %(default)s)", default=[], sticky_default=True, nargs='+', action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--x-ticktext", help="tick text on x-axis (only tick mode array) (default %(default)s)", default=[], sticky_default=True, nargs='+', action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--y-tickangle", help="tick angle on y-axis (default %(default)s)", default='auto', sticky_default=True, choices=Range(None, None, ['auto']), action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--x-tickangle", help="tick angle on x-axis (default %(default)s)", default='auto', sticky_default=True, choices=Range(None, None, ['auto']), action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--y-grid-colour", help="set y-grid colour", type=str, default=None, action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--x-grid-colour", help="set x-grid colour", type=str, default=None, action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--y-colour", help="set y-axis colour", type=str, default=None, action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--x-colour", help="set x-axis colour", type=str, default=None, action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--y-line-width", help="set y-axis line width (default %(default)s)", type=float, choices=Range(0, None), default=0, action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--x-line-width", help="set x-axis line width (default %(default)s)", type=float, choices=Range(0, None), default=0, action=ChildAction, parent=inputFileArgument)
parserPlotAxisOptions.add_argument("--grid-colour", help="set grid colour", type=str, default=None, action=ChildAction, parent=inputFileArgument)

parserColourOptions = parser.add_argument_group('colour options')
parserColourOptions.add_argument("--theme", help="theme to use (all colour options only apply to 'palette')", default='palette', choices=["palette", "plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"])
parserColourOptions.add_argument("--colours", help="define explicit colours (filled up by palette)", default=[], nargs='+', type=str)
parserColourOptions.add_argument("--palette", help="valid seaborn colour palette (default %(default)s)", type=str, default='ch:s=2.8,rot=0.1,d=0.85,l=0.15')
parserColourOptions.add_argument("--palette-count", help="manually set the number of colours to generate from the palette", type=int, choices=Range(1, None), default=None)
parserColourOptions.add_argument("--subplot-colours", help="specify explicit subplot colours (sets default colour cycle to subplot)", type=str, default=[], nargs='+', sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserColourOptions.add_argument("--subplot-palette", help="valid seaborn colour palette used for this subplot (sets default colour cycle to subplot)", type=str, default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserColourOptions.add_argument("--subplot-palette-count", help="manually set the number of colours to generate from the subplot palette (set default colour cycle to subplot)", type=int, choices=Range(1, None), default=None, sticky_default=True, action=ChildAction, parent=inputFileArgument)
parserColourOptions.add_argument("--colour-cycle", help="cycle through colours globally or per subplot (default global)", choices=['subplot', 'global'], default=None, action=ChildAction, parent=inputFileArgument)
parserColourOptions.add_argument("--per-trace-colours", help="one colour for each trace (default)", action='store_true', default=False)
parserColourOptions.add_argument("--per-frame-colours", help="one colour for each dataframe", action='store_true', default=False)
parserColourOptions.add_argument("--per-input-colours", help="one colour for each input file", action='store_true', default=False)
parserColourOptions.add_argument("--font-colour", help="font colour (default %(default)s)", type=str, default='#000000')
parserColourOptions.add_argument("--background-colour", help="set background colour  (default 'rgba(255, 255, 255, 0)')", type=str, default=None)

parserPlotGlobalOptions = parser.add_argument_group('plot global options')

parserPlotGlobalOptions.add_argument("--master-title", help="plot master title", type=str, default=None)
parserPlotGlobalOptions.add_argument("--x-master-title", help="x-axis master title", type=str, default=None)
parserPlotGlobalOptions.add_argument("--y-master-title", help="y-axis master title", type=str, default=None)
parserPlotGlobalOptions.add_argument("--x-share", help="share subplot x-axis (default %(default)s)", default=False, action="store_true")
parserPlotGlobalOptions.add_argument("--y-share", help="share subplot y-axis (default %(default)s)", default=False, action="store_true")
parserPlotGlobalOptions.add_argument("--vertical-spacing", type=float, help="vertical spacing between subplots", default=None, choices=Range(0, 1))
parserPlotGlobalOptions.add_argument("--horizontal-spacing", type=float, help="horizontal spacing between subplots", default=None, choices=Range(0, 1))
parserPlotGlobalOptions.add_argument("--font-size", help="font size (default %(default)s)", type=int, default=12)
parserPlotGlobalOptions.add_argument("--font-family", help="font family (default %(default)s)", type=str, default='"Open Sans", verdana, arial, sans-serif')

parserPlotGlobalOptions.add_argument("--legend", help="quick setting the legend position (default %(default)s)", type=str, choices=['topright', 'topcenter', 'topleft', 'bottomright', 'bottomcenter', 'bottomleft', 'middleleft', 'center', 'middleright', 'belowleft', 'belowcenter', 'belowright', 'aboveleft', 'abovecenter', 'aboveright', 'righttop', 'rightmiddle', 'rightbottom', 'lefttop', 'leftmiddle', 'leftbottom'], default='righttop')
parserPlotGlobalOptions.add_argument("--legend-entries", help="choose which entries are shown in legend", choices=['all', 'unique', 'none'], default=None)
parserPlotGlobalOptions.add_argument("--legend-x", help="x legend position (-2 to 3)", type=float, choices=Range(-2, 3), default=None)
parserPlotGlobalOptions.add_argument("--legend-y", help="y legend position (-2 to 3)", type=float, choices=Range(-2, 3), default=None)
parserPlotGlobalOptions.add_argument("--legend-x-anchor", help="set legend xanchor", choices=['auto', 'left', 'center', 'right'], default=None)
parserPlotGlobalOptions.add_argument("--legend-y-anchor", help="set legend yanchor", choices=['auto', 'top', 'bottom', 'middle'], default=None)
parserPlotGlobalOptions.add_argument("--legend-hide", help="hides legend", default=None, action="store_true")
parserPlotGlobalOptions.add_argument("--legend-show", help="forces legend to show up", default=None, action="store_true")
parserPlotGlobalOptions.add_argument("--legend-vertical", help="horizontal legend", default=None, action="store_true")
parserPlotGlobalOptions.add_argument("--legend-horizontal", help="vertical legend", default=None, action="store_true")

parserPlotGlobalOptions.add_argument("--margins", help="sets all margins", type=int, choices=Range(0, None), default=None)
parserPlotGlobalOptions.add_argument("--margin-l", help="sets left margin", type=int, choices=Range(0, None), default=None)
parserPlotGlobalOptions.add_argument("--margin-r", help="sets right margin", type=int, choices=Range(0, None), default=None)
parserPlotGlobalOptions.add_argument("--margin-t", help="sets top margin", type=int, choices=Range(0, None), default=None)
parserPlotGlobalOptions.add_argument("--margin-b", help="sets bottom margin", type=int, choices=Range(0, None), default=None)
parserPlotGlobalOptions.add_argument("--margin-pad", help="sets padding", type=int, choices=Range(0, None), default=None)

parserPlotGlobalOptions.add_argument("--width", help="plot width", type=int, default=1000)
parserPlotGlobalOptions.add_argument("--height", help="plot height", type=int)

parserOutputOptions = parser.add_argument_group('output options')
parserOutputOptions.add_argument("--orca", help="path to plotly orca (https://github.com/plotly/orca)", type=str, default=None)
parserOutputOptions.add_argument("--script", help="save self-contained plotting script", type=str, default=None)
parserOutputOptions.add_argument("--browser", help="open plot in the browser", default=False, action="store_true")
parserOutputOptions.add_argument("-o", "--output", help="export plot to file (html, pdf, svg, png, py, ...)", default=[], nargs='+')
parserOutputOptions.add_argument("-q", "--quiet", action="store_true", help="no warnings and don't open output file", default=False)

args = parser.parse_args()

commentColour = ''
commentBackgroundColour = ''
uniqueBarMode = False

if args.theme == 'palette':
    args.theme = 'plotly_white'
else:
    # We have chosen a theme, so just comment all colour settings out
    commentColour = '# '
    commentBackgroundColour = '' if args.background_colour else '# '
    # Better to show all legend entries now if not otherwise chosen
    if args.legend_entries is None:
        args.legend_entries = 'all'

if not args.background_colour:
    args.background_colour = 'rgba(255, 255, 255, 0)'

# Setting the legend entries default in case nothing was chosen
if args.legend_entries is None:
    args.legend_entries = 'unique'

if (args.bar_mode == 'unique_group'):
    args.bar_mode = 'group'
    uniqueBarMode = True

if (not args.per_trace_colours and not args.per_frame_colours and not args.per_input_colours) or (args.per_trace_colours):
    args.per_trace_colours = True
    args.per_frame_colours, args.per_input_colours = False, False
elif (args.per_frame_colours):
    args.per_input_colours = False

for input in args.input:
    options = input['args']
    options.ignore_icolumns = list(set(options.ignore_icolumns))
    options.ignore_columns = list(set(options.ignore_columns))

    options.traceSpecialColumns = [options.special_column_start + x for x in traceSpecialColumns]
    options.frameSpecialColumns = [options.special_column_start + x for x in frameSpecialColumns]

    if (options.opacity == 'auto' and ((options.plot == 'box' and 'overlay' in args.box_mode) or
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

    if len(options.line_markers) == 0:
        options.line_markers = ['circle']

    if options.error == 'show':
        options.show_error = True
    else:
        options.show_error = False
    options.hide_error = not options.show_error

    if options.colour_cycle is None and (len(options.subplot_colours) > 0 or options.subplot_palette is not None or options.subplot_palette_count is not None):
        options.colour_cycle = 'subplot'

    if options.colour_cycle is None:
        options.colour_cycle = 'global'

    options.y_grid_colour = f"'{options.y_grid_colour}'" if options.y_grid_colour is not None else f"'{options.grid_colour}'" if options.grid_colour is not None else None
    options.x_grid_colour = f"'{options.x_grid_colour}'" if options.x_grid_colour is not None else f"'{options.grid_colour}'" if options.grid_colour is not None else None
    options.x_colour = f"'{options.y_colour}'" if options.y_colour is not None else None
    options.y_colour = f"'{options.x_colour}'" if options.x_colour is not None else None
    options.y_line_width_forced = 'y_line_width' in [i for (i, _) in options.ordered_args]
    options.x_line_width_forced = 'x_line_width' in [i for (i, _) in options.ordered_args]

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

# Default values for the sticky default mode selection arguments

for input in args.input:
    inputOptions = input['args']
    inputFileNames = input['value']
    inputOptions.filenames = inputFileNames
    inputOptions.traceCount = 0
    inputOptions.frameCount = 0
    inputOptions.frameIndex = 0
    inputFrames = []
    for filename in inputFileNames:
        try:
            rawFile = xopen.xopen(filename, mode='rb')
        except Exception:
            raise Exception(f'Could not open file {filename}!')

        rawData = rawFile.read()
        rawFile.close()
        try:
            frame = pickle.loads(rawData)
        except Exception:
            frame = None

        if frame is not None:
            options = copy.deepcopy(inputOptions)
            options.filenames = [filename]

            if not args.quiet and inputOptions.no_index:
                print("WARNING: ignoring --no-index for {filename}", file=sys.stderr)
            if not args.quiet and inputOptions.no_columns:
                print("WARNING: ignoring --no-columns for {filename}", file=sys.stderr)

            if (isinstance(frame, list)):
                for f in frame:
                    if (not isinstance(f, pandas.DataFrame)):
                        raise Exception(f'pickle file {filename} is not a list of pandas dataframes!')
                    inputFrames.append((copy.deepcopy(options), f))
            elif (not isinstance(frame, pandas.DataFrame)):
                raise Exception(f'pickle file {filename} is not a pandas data frame!')
            else:
                inputFrames.append((options, frame))
        else:
            fFile = rawData.decode('utf-8').replace('\r\n', '\n')
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

            frame = pandas.DataFrame(fData)

            if (not options.no_columns):
                frame.columns = frame.iloc[0]
                frame.columns.name = ''
                frame = DataframeActions.dropRowsByIdx(frame, 0)

            if (not options.no_index):
                frame.index = frame.iloc[:, 0]
                frame = DataframeActions.dropColumnsByIdx(frame, 0)

            options.filenames = [filename]
            options.frameCount = 1
            options.frameIndex = inputOptions.frameCount
            frame = frame.fillna(value=numpy.nan)
            inputFrames.append((options, frame))
            inputOptions.frameCount += 1

    selectMode = input['args'].select_mode
    sortFunction = input['args'].sort_function
    addAt = input['args'].add_at
    addFunction = input['args'].add_function
    applyFunction = input['args'].apply_function
    groupFunction = input['args'].group_function
    applyInclusive = input['args'].apply_mode == 'inclusive'
    sortOrder = input['args'].sort_order
    outputPrecision = None if input['args'].output_precision == 'default' else int(input['args'].output_precision)

    focusedFrames = list(range(len(inputFrames)))

    for (optionName, optionValue) in input['args'].ordered_args:
        multiFrameActions = ['output_precision', 'select_mode', 'sort_function', 'sort_order', 'add_at', 'add_function', 'apply_function', 'group_function',
                             'apply_mode', 'join', 'file', 'pickle', 'split_column', 'split_icolumn', 'split_row', 'split_irow', 'focus_frames', 'defocus_frames']
        if optionName not in multiFrameActions:
            for _index, (frameOptions, frame) in enumerate(inputFrames):
                if (_index) not in focusedFrames:
                    continue
                if optionName == 'transpose':
                    frame = DataframeActions.transpose(frame)
                elif optionName == 'index_column':
                    columnIdx = DataframeActions.getColumnIdx(frame, optionValue, selectMode)[0]
                    frame = DataframeActions.setIndexColumnByIDx(frame, columnIdx)
                elif optionName == 'index_icolumn':
                    frame = DataframeActions.setIndexColumnByIDx(frame, optionValue)
                elif optionName == 'ignore_columns':
                    columnIdx = DataframeActions.getColumnIdx(frame, optionValue, selectMode, True)
                    frame = DataframeActions.dropColumnsByIdx(frame, columnIdx)
                elif optionName == 'ignore_icolumns':
                    frame = DataframeActions.dropColumnsByIdx(frame, optionValue)
                elif optionName == 'ignore_rows':
                    rowIdx = DataframeActions.getRowIdx(frame, optionValue, selectMode, True)
                    frame = DataframeActions.dropRowsByIdx(frame, rowIdx)
                elif optionName == 'ignore_irows':
                    frame = DataframeActions.dropRowsByIdx(frame, optionValue)
                elif optionName == 'select_columns':
                    columnIdx = DataframeActions.getColumnIdx(frame, optionValue, selectMode, True)
                    frame = DataframeActions.filterColumnsByIdx(frame, columnIdx)
                elif optionName == 'select_icolumns':
                    frame = DataframeActions.filterColumnsByIdx(frame, optionValue)
                elif optionName == 'select_rows':
                    rowIdx = DataframeActions.getRowIdx(frame, optionValue, selectMode, True)
                    frame = DataframeActions.filterRowsByIdx(frame, rowIdx)
                elif optionName == 'select_irows':
                    frame = DataframeActions.filterRowsByIdx(frame, optionValue)
                elif optionName == 'reverse_columns':
                    frame = DataframeActions.reverseColumns(frame)
                elif optionName == 'reverse_rows':
                    frame = DataframeActions.reverseRows(frame)
                elif optionName == 'sort_columns':
                    frame = DataframeActions.sortColumns(frame, sortFunction, sortOrder)
                elif optionName == 'sort_rows':
                    frame = DataframeActions.sortRows(frame, sortFunction, sortOrder)
                elif optionName == 'sort_by_column':
                    columnIdx = DataframeActions.getColumnIdx(frame, optionValue, selectMode)[0]
                    frame = DataframeActions.sortRowsByColumnIdx(frame, columnIdx, sortOrder)
                elif optionName == 'sort_by_icolumn':
                    frame = DataframeActions.sortRowsByColumnIdx(frame, optionValue, sortOrder)
                elif optionName == 'sort_by_row':
                    rowIdx = DataframeActions.getRowIdx(frame, optionValue, selectMode)[0]
                    frame = DataframeActions.sortColumnsByRowIdx(frame, rowIdx, sortOrder)
                elif optionName == 'sort_by_irow':
                    frame = DataframeActions.sortColumnsByRowIdx(frame, optionValue, sortOrder)
                elif optionName == 'normalise_to':
                    frame = DataframeActions.normaliseToConstant(frame, optionValue)
                elif optionName == 'normalise_to_column':
                    columnIdx = DataframeActions.getColumnIdx(frame, optionValue, selectMode)[0]
                    frame = DataframeActions.applyOnColumnIdx(frame, columnIdx, 'div', True)
                elif optionName == 'normalise_to_icolumn':
                    frame = DataframeActions.applyOnColumnIdx(frame, optionValue, 'div', True)
                elif optionName == 'normalise_to_row':
                    rowIdx = DataframeActions.getRowIdx(frame, optionValue, selectMode)[0]
                    frame = DataframeActions.applyOnRowIdx(frame, rowIdx, 'div', True)
                elif optionName == 'normalise_to_irow':
                    frame = DataframeActions.applyOnRowIdx(frame, optionValue, 'div', True)
                elif optionName == 'abs':
                    frame = DataframeActions.abs(frame)
                elif optionName == 'apply_irow':
                    frame = DataframeActions.applyOnRowIdx(frame, optionValue, applyFunction, applyInclusive)
                elif optionName == 'apply_row':
                    rowIdx = DataframeActions.getRowIdx(frame, optionValue, selectMode)[0]
                    frame = DataframeActions.applyOnRowIdx(frame, rowIdx, applyFunction, applyInclusive)
                elif optionName == 'apply_icolumn':
                    frame = DataframeActions.applyOnColumnIdx(frame, optionValue, applyFunction, applyInclusive)
                elif optionName == 'apply_column':
                    columnIdx = DataframeActions.getColumnIdx(frame, optionValue, selectMode)[0]
                    frame = DataframeActions.applyOnColumnIdx(frame, columnIdx, applyFunction, applyInclusive)
                elif optionName == 'apply_irows':
                    if 'all' in optionValue:
                        optionValue = range(frame.shape[0])
                    for rowIdx in optionValue:
                        frame = DataframeActions.applyOnRowIdx(frame, int(rowIdx), applyFunction, applyInclusive)
                elif optionName == 'apply_rows':
                    for rowName in optionValue:
                        rowIdx = DataframeActions.getRowIdx(frame, rowName, selectMode)[0]
                        frame = DataframeActions.applyOnRowIdx(frame, rowIdx, applyFunction, applyInclusive)
                elif optionName == 'apply_icolumns':
                    if 'all' in optionValue:
                        optionValue = range(frame.shape[1])
                    for colIdx in optionValue:
                        frame = DataframeActions.applyOnColumnIdx(frame, int(colIdx), applyFunction, applyInclusive)
                elif optionName == 'apply_columns':
                    for colName in optionValue:
                        columnIdx = DataframeActions.getColumnIdx(frame, colName, selectMode)[0]
                        frame = DataframeActions.applyOnColumnIdx(frame, columnIdx, applyFunction, applyInclusive)
                elif optionName == 'group_by_irow':
                    frame = DataframeActions.groupByRowIdx(frame, optionValue, groupFunction)
                elif optionName == 'group_by_row':
                    rowIdx = DataframeActions.getRowIdx(frame, optionValue, selectMode)[0]
                    frame = DataframeActions.groupByRowIdx(frame, rowIdx, groupFunction)
                elif optionName == 'group_by_icolumn':
                    frame = DataframeActions.groupByColumnIdx(frame, optionValue, groupFunction)
                elif optionName == 'group_by_column':
                    columnIdx = DataframeActions.getColumnIdx(frame, optionValue, selectMode)[0]
                    frame = DataframeActions.groupByColumnIdx(frame, columnIdx, groupFunction)
                elif optionName == 'add_column':
                    frame = DataframeActions.addColumn(frame, optionValue, addFunction, addAt)
                elif optionName == 'add_row':
                    frame = DataframeActions.addRow(frame, optionValue, addFunction, addAt)
                elif optionName == 'data_offset':
                    frame = DataframeActions.addConstant(frame, optionValue)
                elif optionName == 'data_scale':
                    frame = DataframeActions.scaleConstant(frame, optionValue)
                elif optionName == 'column_names':
                    frame = DataframeActions.renameColumns(frame, optionValue)
                elif optionName == 'row_names':
                    frame = DataframeActions.renameRows(frame, optionValue)
                elif optionName == 'drop_nan':
                    frame = DataframeActions.dropNaN(frame)
                elif optionName == 'drop_any_nan':
                    frame = DataframeActions.dropNaN(frame, True)
                elif optionName == 'print':
                    DataframeActions.printFrames(frameOptions.filenames, frame, _index, len(inputFrames), outputPrecision)
                    doneSomething = True

                inputFrames[_index] = (frameOptions, frame)
        else:
            if optionName == 'focus_frames':
                if 'all' in optionValue:
                    focusedFrames = list(range(len(inputFrames)))
                else:
                    focusedFrames = []
                    for _index in [int(x) for x in optionValue]:
                        nindex = _index if _index >= 0 else _index + len(inputFrames)
                        if nindex not in focusedFrames:
                            if nindex < len(inputFrames):
                                focusedFrames.append(nindex)
                            else:
                                raise Exception(f"frame index {_index} out of bounds")
                    focusedFrames = sorted(focusedFrames)
            elif optionName == 'defocus_frames':
                for _index in optionValue:
                    nindex = _index if _index >= 0 else _index + len(inputFrames)
                    if nindex in focusedFrames:
                        focusedFrames.remove(nindex)
            elif optionName == 'output_precision':
                outputPrecision = None if optionValue == 'default' else int(optionValue)
            elif optionName == 'select_mode':
                selectMode = optionValue
            elif optionName == 'sort_function':
                sortFunction = optionValue
            elif optionName == 'sort_order':
                sortOrder = optionValue
            elif optionName == 'add_at':
                addAt = optionValue
            elif optionName == 'add_function':
                addFunction = optionValue
            elif optionName == 'apply_function':
                applyFunction = optionValue
            elif optionName == 'group_function':
                groupFunction = optionValue
            elif optionName == 'apply_mode':
                applyInclusive = optionValue == 'inclusive'
            elif optionName == 'join':
                newOptions = copy.deepcopy(inputOptions)
                newOptions.filenames = []
                newOptions.frameCount = 1
                frontDefocusedFrames = [inputFrames[x] for x in range(focusedFrames[0]) if x not in focusedFrames]
                backDefocusedFrames = [inputFrames[x] for x in range(focusedFrames[0], len(inputFrames)) if x not in focusedFrames]
                joinedFrame = DataframeActions.joinFrames([frame for (_, frame) in [inputFrames[x] for x in focusedFrames]], optionValue)
                inputFrames = frontDefocusedFrames + [(newOptions, joinedFrame)] + backDefocusedFrames
                focusedFrames = [len(frontDefocusedFrames)]
            elif optionName == 'file':
                DataframeActions.framesToCSV([frame for (_, frame) in [inputFrames[x] for x in focusedFrames]], optionValue, inputOptions.separator, args.quiet, outputPrecision)
                doneSomething = True
            elif optionName == 'pickle':
                DataframeActions.framesToPickle([frame for (_, frame) in [inputFrames[x] for x in focusedFrames]], optionValue, args.quiet)
                doneSomething = True
            elif optionName in ['split_column', 'split_icolumn', 'split_row', 'split_irow']:
                newInputFrames = []
                newFocusedFrames = []
                for _index, (frameOptions, frame) in enumerate(inputFrames):
                    if (_index) not in focusedFrames:
                        newInputFrames.append((frameOptions, frame))
                        continue
                    newFrames = []
                    if optionName == 'split_icolumn':
                        newFrames = DataframeActions.splitFramesByColumnIdx([frame], optionValue)
                    elif optionName == 'split_column':
                        columnIdx = DataframeActions.getColumnIdx(frame, optionValue, selectMode)[0]
                        newFrames = DataframeActions.splitFramesByColumnIdx([frame], columnIdx)
                    elif optionName == 'split_irow':
                        newFrames = DataframeActions.splitFramesByRowIdx([frame], optionValue)
                    elif optionName == 'split_row':
                        rowIdx = DataframeActions.getRowIdx(frame, optionValue, selectMode)[0]
                        newFrames = DataframeActions.splitFramesByRowIdx([frame], rowIdx)
                    newFocusedFrames.extend(range(len(newInputFrames), len(newInputFrames) + len(newFrames)))
                    for newFrame in newFrames:
                        newOptions = copy.deepcopy(inputOptions)
                        newOptions.filenames = []
                        newOptions.frameCount = 1
                        newInputFrames.append((newOptions, newFrame))
                focusedFrames = newFocusedFrames
                inputFrames = newInputFrames

    defocusedFrames = [i for i in range(len(inputFrames)) if i not in focusedFrames]
    if not args.quiet and len(defocusedFrames) > 0:
        print(f'WARNING: {len(defocusedFrames)} frames are defocused and will be ignored', file=sys.stderr)

    inputOptions.traceCount = 0
    inputOptions.frameCount = 0
    for _index, (options, frame) in enumerate(inputFrames):
        if _index not in focusedFrames:
            continue
        inputOptions.frameCount += 1
        inputOptions.traceCount += len([x for x in frame.columns if str(x) not in inputOptions.traceSpecialColumns and str(x) not in inputOptions.frameSpecialColumns])
        frame = frame.replace([numpy.inf, -numpy.inf], numpy.nan)
        frame = frame.where(pandas.notnull(frame), None)
        inputFrames[_index] = (options, frame)

    if inputOptions.frameCount == 0:
        if not args.quiet:
            print(f'WARNING: files {", ".join(inputFileNames)} did turn into any valid dataframes', file=sys.stderr)
        continue

    inputOptions.inputIndex = totalInputCount
    totalTraceCount += inputOptions.traceCount
    totalFrameCount += inputOptions.frameCount
    totalInputCount += 1

    updateRange(subplotGrid, [inputOptions.col + (inputOptions.colspan - 1), inputOptions.row + (inputOptions.rowspan - 1)])
    if (inputOptions.row not in subplotGridDefinition):
        subplotGridDefinition[inputOptions.row] = {}
    if (inputOptions.col not in subplotGridDefinition[inputOptions.row]):
        subplotGridDefinition[inputOptions.row][inputOptions.col] = copy.deepcopy({
            'rowspan': inputOptions.rowspan,
            'colspan': inputOptions.colspan,
            'secondary_y': inputOptions.y_secondary,
            'title': inputOptions.title,
            'traces': 0,
            'frames': 0,
            'colours': args.colours,
            'palette': args.palette,
            'palette_count': args.palette_count,
            'palette_local': inputOptions.colour_cycle == 'subplot',
            'palette_index': 0
        })

    subplotGridDefinition[inputOptions.row][inputOptions.col]['rowspan'] = max(inputOptions.rowspan, subplotGridDefinition[inputOptions.row][inputOptions.col]['rowspan'])
    subplotGridDefinition[inputOptions.row][inputOptions.col]['colspan'] = max(inputOptions.colspan, subplotGridDefinition[inputOptions.row][inputOptions.col]['colspan'])
    subplotGridDefinition[inputOptions.row][inputOptions.col]['secondary_y'] = inputOptions.y_secondary or subplotGridDefinition[inputOptions.row][inputOptions.col]['secondary_y']
    if inputOptions.title is not None:
        subplotGridDefinition[inputOptions.row][inputOptions.col]['title'] = inputOptions.title

    if len(inputOptions.subplot_colours) > 0:
        subplotGridDefinition[inputOptions.row][inputOptions.col]['colours'] = copy.copy(inputOptions.subplot_colours)
    if inputOptions.subplot_palette is not None:
        subplotGridDefinition[inputOptions.row][inputOptions.col]['palette'] = inputOptions.subplot_palette
    if inputOptions.subplot_palette_count is not None:
        subplotGridDefinition[inputOptions.row][inputOptions.col]['palette_count'] = inputOptions.subplot_palette_count

    inputOptions.subplotTraceIndex = subplotGridDefinition[inputOptions.row][inputOptions.col]['traces']
    subplotGridDefinition[inputOptions.row][inputOptions.col]['traces'] += inputOptions.traceCount
    subplotGridDefinition[inputOptions.row][inputOptions.col]['frames'] += inputOptions.frameCount

    data.append({'options': copy.deepcopy(inputOptions), 'frames': [f.where(pandas.notnull(f), None) for f in [f for i, (_, f) in enumerate(inputFrames) if i in focusedFrames]]})


# Separate python outputs from actual output put the first script
# into args.script and all others into scriptOutputs, args.script is
# out master all others are secondary
secondaryScripts = []
for x in args.output:
    if x.lower().endswith('.py'):
        if not args.script:
            args.script = x
        else:
            secondaryScripts.append(x)

# Remove scripts from output
args.output = [x for x in args.output if not x.lower().endswith('.py')]


if doneSomething and not args.browser and len(args.output) == 0 and not args.script:
    exit(0)
elif len(args.output) == 0 and not args.script:
    args.browser = True

if totalTraceCount == 0:
    if not args.quiet:
        print('No input data available for plotting.')
    exit(0)

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


for r in range(1, subplotGrid[1]['max'] + 1):
    for c in range(1, subplotGrid[0]['max'] + 1):
        if r in subplotGridDefinition and c in subplotGridDefinition[r]:
            subplot = subplotGridDefinition[r][c]
            if subplot['palette_count'] is None:
                if subplot['palette_local']:
                    subplot['palette_count'] = subplot['traces'] if args.per_trace_colours else subplot['frames'] if args.per_frame_colour else 1
                else:
                    subplot['palette_count'] = totalTraceCount if args.per_trace_colours else totalFrameCount if args.per_frame_colour else totalInputCount
                subplot['palette_count'] = max(0, subplot['palette_count'] - len(subplot['colours']))
            subplot['colours'].extend([f'rgb({int(255*r)}, {int(255*g)}, {int(255*b)})' for (r, g, b) in seaborn.color_palette(subplot['palette'], subplot['palette_count'])])
globalPaletteIndex = 0

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

import urllib.request
import glob
import re
import platform

parser = argparse.ArgumentParser(description="plots the contained figure")
parser.add_argument("--font-size", help="font size (default %(default)s)", type=int, default={args.font_size})
parser.add_argument("--font-colour", help="font colour (default %(default)s)", default='{args.font_colour}')
parser.add_argument("--font-family", help="font family (default %(default)s)", default='{args.font_family}')
parser.add_argument("--orca", help="path to plotly orca (https://github.com/plotly/orca)", type=str, default=None)
parser.add_argument("--width", help="width of output file (default %(default)s)", type=int, default={args.width})
parser.add_argument("--height", help="height of output (default %(default)s)", type=int, default={args.height})
parser.add_argument("--output", "-o", help="output file (html, png, jpeg, pdf...) (default %(default)s)", type=str, nargs="+", default={args.output})
parser.add_argument("--browser", help="open plot in browser", action="store_true")
parser.add_argument("--quiet", "-q", help="no warnings and don't open output file", action="store_true")

args = parser.parse_args()

if len(args.output) == 0:
    args.browser = True
""")


subplotTitles = []

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
    plotScript.write("\n        [")
    for c in range(1, subplotGrid[0]['max'] + 1):
        if (r in subplotGridDefinition and c in subplotGridDefinition[r]):
            plotScript.write(f"{{'rowspan': {subplotGridDefinition[r][c]['rowspan']}, 'colspan': {subplotGridDefinition[r][c]['colspan']}, 'secondary_y': {subplotGridDefinition[r][c]['secondary_y']}}}, ")
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
inputIndex = 0
for input in data:
    options = input['options']
    frames = input['frames']
    subplot = subplotGridDefinition[options.row][options.col]
    plotRange = []
    inputTraceIndex = 0
    inputFrameIndex = 0
    multiCategory = False
    for frame in frames:
        # NaN cannot be plotted or used, cast it to None
        # Drop only columns/rows NaN values and replace NaN with None
        frame = frame.dropna(how='all', axis=0)
        frame = frame.where((pandas.notnull(frame)), None)

        frameTraceIndex = 0

        _categories = None
        for specialFrameColumn in options.frameSpecialColumns:
            if specialFrameColumn not in frame.columns:
                continue
            for colIndex in range(len(frame.columns)):
                colName = str(frame.columns[colIndex])
                if (colName == options.special_column_start + 'category') and _categories is None:
                    _categories = ['' if x is None else x for x in frame.iloc[:, colIndex].values.tolist()]

        for colIndex, _ in enumerate(frame.columns):
            col = str(frame.columns[colIndex])
            if col in options.traceSpecialColumns or col in options.frameSpecialColumns:
                continue

            if options.trace_colours and frameTraceIndex < len(options.trace_colours):
                fillcolour = options.trace_colours[frameTraceIndex]
            else:
                colourIndex = (subplot['palette_index'] if subplot['palette_local'] else globalPaletteIndex) % len(subplot['colours'])
                fillcolour = subplot['colours'][colourIndex]
            markercolour = options.line_colour

            _errors_symmetric = True
            _errors_pos = None
            _errors_neg = None
            _bases = None
            _labels = None
            _colours = None
            for nextColIndex in range(colIndex + 1, colIndex + 1 + len(options.traceSpecialColumns) if colIndex + 1 + len(options.traceSpecialColumns) <= len(frame.columns) else len(frame.columns)):
                nextCol = str(frame.columns[nextColIndex])
                if (nextCol not in options.traceSpecialColumns):
                    break
                if (nextCol == options.special_column_start + 'error') and (_errors_pos is None):
                    _errors_pos = [x if (x is not None) else 0 for x in frame.iloc[:, nextColIndex].values.tolist()]
                elif (nextCol == options.special_column_start + 'error+') and (_errors_pos is None):
                    _errors_symmetric = False
                    _errors_pos = [x if (x is not None) else 0 for x in frame.iloc[:, nextColIndex].values.tolist()]
                elif (nextCol == options.special_column_start + 'error-') and (_errors_neg is None):
                    _errors_symmetric = False
                    _errors_neg = [x if (x is not None) else 0 for x in frame.iloc[:, nextColIndex].values.tolist()]
                elif (nextCol == options.special_column_start + 'offset') and (_bases is None):
                    _bases = [x if (x is not None) else 0 for x in frame.iloc[:, nextColIndex].values.tolist()]
                elif (nextCol == options.special_column_start + 'label') and (_labels is None):
                    _labels = frame.iloc[:, nextColIndex].values.tolist()
                elif (nextCol == options.special_column_start + 'colour') and (_colours is None) and (frameTraceIndex >= len(options.trace_colours)):
                    _colours = frame.iloc[:, nextColIndex].values.tolist()
                    _colours = [c if c is not None else fillcolour for c in _colours]

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
                if _categories is not None:
                    multiCategory = True
                    if options.horizontal:
                        ydata = [_categories, ydata]
                    else:
                        xdata = [_categories, xdata]
            else:  # Box and Violin
                data = [x for x in frame.iloc[:, colIndex].values.tolist() if x is not None]
                index = f"['{col}'] * {len(data)}"
                ydata = index if not options.vertical else data
                xdata = index if options.vertical else data
                updateRange(plotRange, [xdata, ydata])

            if options.offsetgroups == 'auto':
                offsetgroup = (0 if (uniqueBarMode) else options.subplotTraceIndex) + inputFrameIndex + frameTraceIndex + 1 if args.bar_mode == 'group' else None
            else:
                if inputTraceIndex < len(options.offsetgroups):
                    offsetgroup = options.offsetgroups[inputTraceIndex]
                else:
                    offsetgroup = options.offsetgroups[-1]

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
                lineMarker = options.line_markers[-1] if len(options.line_markers) <= inputTraceIndex else options.line_markers[inputTraceIndex]
                plotScript.write(f"""
fig.add_trace(go.Scatter(
    name='{traceName}',
    legendgroup='{traceName}',
    showlegend={showInLegend},
    mode='{options.line_mode}',""")
                if (_colours is not None):
                    plotScript.write(f"""
    marker_color={_colours},""")
                else:
                    plotScript.write(f"""
{commentColour}    marker_color='{fillcolour}',""")
                plotScript.write(f"""
{commentColour}    line_color='{fillcolour}',
{commentColour}    fillcolor='{fillcolour}', # Currently not supported through script, using default
    stackgroup='{'stackgroup-' + str(inputIndex) if options.line_stack else ''}',
    marker_symbol='{lineMarker}',
    marker_size={options.line_marker_size},
    fill='{options.line_fill}',
    line_dash='{options.line_dash}',
    line_shape='{options.line_shape}',
    line_width={options.line_width},
    y={ydata},
    x={xdata},""")
                if (_labels is not None):
                    plotScript.write(f"""
    text={_labels},
    textposition='{options.line_text_position}',""")
                if (_errors_pos is not None or _errors_neg is not None):
                    plotScript.write(f"""
    error_{'y' if options.horizontal else 'x'}=dict(
        visible={options.show_error},
        type='data',
        symmetric={_errors_symmetric},
        array={_errors_pos},
        arrayminus={_errors_neg},
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
{commentColour}    marker_color='{fillcolour}',""")
                plotScript.write(f"""
{commentColour}    marker_line_color='{markercolour}',
    marker_line_width={options.line_width},
    width={options.bar_width},
    offset={options.bar_shift},
    offsetgroup={offsetgroup},
    y={ydata},
    x={xdata},""")
                if (_labels is not None):
                    plotScript.write(f"""
    text={_labels},
    textposition='{options.bar_text_position}',""")
                if (_bases is not None):
                    plotScript.write(f"""
    base={_bases},""")
                if (_errors_pos is not None or _errors_neg is not None):
                    plotScript.write(f"""
    error_{'x' if options.horizontal else 'y'}=dict(
        visible={options.show_error},
        type='data',
        symmetric={_errors_symmetric},
        array={_errors_pos},
        arrayminus={_errors_neg},
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
{commentColour}    fillcolor='{fillcolour}',
{commentColour}    line_color='{markercolour}',
{commentColour}    marker_color='{markercolour}',
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
{commentColour}    fillcolor='{fillcolour}',
{commentColour}    line_color='{markercolour}',
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
{commentColour}    fillcolor='{fillcolour}',
{commentColour}    line_color='{options.line_colour}',
{commentColour}    marker_color='{markercolour}',
    line_width={options.line_width},
    side='{side}',
    width={options.violin_width},
    scalemode='width',
    points={False if options.violin_points == 'none' else options.violin_points},
    jitter={options.violin_jitter},
    orientation='{'v' if options.vertical else 'h'}',
    meanline_visible={True if options.violin_mean == 'line' else False},
    box_visible={True if options.violin_mean == 'box' else False},
    opacity={options.opacity},
), col={options.col}, row={options.row}, secondary_y={options.y_secondary})
""")

            traceIndex += 1
            frameTraceIndex += 1
            inputTraceIndex += 1
            if subplot['palette_local']:
                subplot['palette_index'] += 1 if args.per_trace_colours else 0
            else:
                globalPaletteIndex += 1 if args.per_trace_colours else 0
        inputFrameIndex += 1
        frameIndex += 1
        if subplot['palette_local']:
            subplot['palette_index'] += 1 if args.per_frame_colours else 0
        else:
            globalPaletteIndex += 1 if args.per_frame_colours else 0
    inputIndex += 1
    if subplot['palette_local']:
        subplot['palette_index'] += 1 if args.per_input_colours else 0
    else:
        globalPaletteIndex += 1 if args.per_input_colours else 0

    # Find out if we need left, right and bottom margin:
    if defaultLeftMargin is None and options.col == 1 and options.y_title and not options.y_secondary:
        defaultLeftMargin = True
    if defaultTopMargin is None and options.row == 1 and options.title:
        # default top margin is 100 which is a bit much for just having subplot titles:
        defaultTopMargin = True
        if args.margin_t is None and args.margins is None:
            args.margin_t = 40
    if defaultRightMargin is None and options.col + options.colspan - 1 == subplotGrid[0]['max'] and options.y_title is not None and options.y_secondary:
        defaultRightMargin = True
    if defaultBottomMargin is None and options.row + options.rowspan - 1 == subplotGrid[1]['max'] and options.x_title is not None:
        defaultBottomMargin = True

    # If line width was not explicitly set, set the axis line width for the multi category axis to one
    if multiCategory and options.horizontal and not options.y_line_width_forced:
        options.y_line_width = 1
    if multiCategory and options.vertical and not options.x_line_width_forced:
        options.x_line_width = 1

    plotScript.write("\n\n")
    plotScript.write("# Subplot specific options:\n")
    plotScript.write(f"fig.update_yaxes(type='{options.y_type}', rangemode='{options.y_range_mode}', automargin={True if options.y_title_standoff is None else False}, title_standoff={options.y_title_standoff}, col={options.col}, row={options.row}, secondary_y={options.y_secondary})\n")
    plotScript.write(f"fig.update_xaxes(type='{options.x_type}', rangemode='{options.x_range_mode}', automargin={True if options.x_title_standoff is None else False}, title_standoff={options.x_title_standoff}, col={options.col}, row={options.row})\n")
    plotScript.write(f"fig.update_yaxes(showline={options.y_line_width > 0}, linewidth={options.y_line_width}, linecolor={options.y_colour}, gridcolor={options.y_grid_colour}, col={options.col}, row={options.row}, secondary_y={options.y_secondary})\n")
    plotScript.write(f"fig.update_xaxes(showline={options.x_line_width > 0}, linewidth={options.x_line_width}, linecolor={options.x_colour}, gridcolor={options.x_grid_colour}, col={options.col}, row={options.row})\n")
    plotScript.write("# Multi-category options:\n")
    plotScript.write(f"fig.update_yaxes(showdividers={options.y_line_width > 0}, dividercolor={options.y_colour}, dividerwidth={options.y_line_width}, col={options.col}, row={options.row}, secondary_y={options.y_secondary})\n")
    plotScript.write(f"fig.update_xaxes(showdividers={options.x_line_width > 0}, dividercolor={options.x_colour}, dividerwidth={options.x_line_width}, col={options.col}, row={options.row})\n")
    plotScript.write(f"{'# ' if not options.y_hide else ''}fig.update_yaxes(visible=False, showticklabels=False, showgrid=True, zeroline=False, row={options.row}, col={options.col}, secondary_y={options.y_secondary})\n")
    plotScript.write(f"{'# ' if not options.x_hide else ''}fig.update_xaxes(visible=False, showticklabels=False, showgrid=True, zeroline=False, row={options.row}, col={options.col})\n")
    plotScript.write(f"{'# ' if options.y_title is None else ''}fig.update_yaxes(title_text='{options.y_title}', col={options.col}, row={options.row}, secondary_y={options.y_secondary})\n")
    plotScript.write(f"{'# ' if options.x_title is None else ''}fig.update_xaxes(title_text='{options.x_title}', col={options.col}, row={options.row})\n")
    plotScript.write(f"fig.update_yaxes(tickcolor={options.y_colour}, tickformat='{options.y_tick_format}', ticksuffix='{options.y_tick_suffix}', tickprefix='{options.y_tick_prefix}', col={options.col}, row={options.row}, secondary_y={options.y_secondary})\n")
    plotScript.write(f"fig.update_xaxes(tickcolor={options.x_colour}, tickformat='{options.x_tick_format}', ticksuffix='{options.x_tick_suffix}', tickprefix='{options.x_tick_prefix}', col={options.col}, row={options.row})\n")
    if options.y_range_from is not None or options.y_range_to is not None:
        options.y_range_from = options.y_range_from if options.y_range_from is not None else plotRange[1]['min']
        options.y_range_to = options.y_range_to if options.y_range_to is not None else plotRange[1]['max']
        plotScript.write(f"fig.update_yaxes(range=[{options.y_range_from}, {options.y_range_to}], col={options.col}, row={options.row}, secondary_y={options.y_secondary})\n")
    if options.x_range_from is not None or options.x_range_to is not None:
        options.x_range_from = options.x_range_from if options.x_range_from is not None else plotRange[0]['min']
        options.x_range_to = options.x_range_to if options.x_range_to is not None else plotRange[0]['max']
        plotScript.write(f"fig.update_xaxes(range=[{options.x_range_from}, {options.x_range_to}], col={options.col}, row={options.row})\n")
    plotScript.write(f"# fig.update_yaxes(range=[{plotRange[0]['min']}, {plotRange[0]['max']}], col={options.col}, row={options.row}, secondary_y={options.y_secondary})\n")
    plotScript.write(f"# fig.update_xaxes(range=[{plotRange[1]['min']}, {plotRange[1]['max']}], col={options.col}, row={options.row})\n")
    plotScript.write(f"fig.update_xaxes(tickmode='{options.x_tickmode}', ticks='{options.x_ticks}', nticks={options.x_nticks}, tick0='{options.x_tick0}', dtick='{options.x_dtick}', tickvals={options.x_tickvals}, ticktext={options.x_ticktext}, tickangle={options.x_tickangle}, col={options.col}, row={options.row})\n")
    plotScript.write(f"fig.update_yaxes(tickmode='{options.y_tickmode}', ticks='{options.y_ticks}', nticks={options.y_nticks}, tick0='{options.y_tick0}', dtick='{options.y_dtick}', tickvals={options.y_tickvals}, ticktext={options.y_ticktext}, tickangle={options.y_tickangle}, col={options.col}, row={options.row}, secondary_y={options.y_secondary})\n")
    plotScript.write("\n")

if (args.violin_mode == 'halfgroup'):
    args.violin_mode = 'group'
elif (args.violin_mode[:4] == 'half'):
    args.violin_mode = 'overlay'

plotScript.write('# Global modes and paramters:\n')
plotScript.write(f"fig.update_layout(title={args.master_title})\n")
plotScript.write(f"fig.update_layout(barmode='{args.bar_mode}', boxmode='{args.box_mode}', violinmode='{args.violin_mode}')\n")
plotScript.write(f"fig.update_layout(bargap={args.bar_gap}, bargroupgap={args.bar_group_gap}, boxgap={args.box_gap}, boxgroupgap={args.box_group_gap}, violingap={args.violin_gap}, violingroupgap={args.violin_group_gap})\n")

plotScript.write("\n# Layout Legend\n")
plotScript.write(f"fig.update_layout(showlegend={args.legend_show})\n")
plotScript.write(f"{'# ' if args.legend_y_anchor is None else ''}fig.update_layout(legend_yanchor='{'auto' if args.legend_y_anchor is None else args.legend_y_anchor}')\n")
plotScript.write(f"{'# ' if args.legend_x_anchor is None else ''}fig.update_layout(legend_xanchor='{'auto' if args.legend_x_anchor is None else args.legend_x_anchor}')\n")
plotScript.write(f"fig.update_layout(legend=dict(x={args.legend_x}, y={args.legend_y}, orientation='{'v' if args.legend_vertical else 'h'}', bgcolor='rgba(255,255,255,0)'))\n")

plotScript.write("\n# Layout Plot and Background\n")
plotScript.write(f"{commentBackgroundColour}fig.update_layout(paper_bgcolor='{args.background_colour}', plot_bgcolor='{args.background_colour}')\n")

args.margin_b = args.margin_b if args.margin_b is not None else args.margins if args.margins is not None else None if defaultBottomMargin else 0
args.margin_l = args.margin_l if args.margin_l is not None else args.margins if args.margins is not None else None if defaultLeftMargin else 0
args.margin_r = args.margin_r if args.margin_r is not None else args.margins if args.margins is not None else None if defaultRightMargin else 0
args.margin_t = args.margin_t if args.margin_t is not None else args.margins if args.margins is not None else None if defaultTopMargin else 0
args.margin_pad = args.margin_pad if args.margin_pad is not None else args.margins if args.margins is not None else None if defaultPadMargin else 0

plotScript.write(f"fig.update_layout(margin=dict(t={args.margin_t}, l={args.margin_l}, r={args.margin_r}, b={args.margin_b}, pad={args.margin_pad}))\n")

plotScript.write("\n# Plot Font\n")
plotScript.write(f"""fig.update_layout(font=dict(
    family=args.font_family,
    size=args.font_size,
{commentColour}    color=args.font_colour
))
""")

plotScript.write("""
# Execute addon file if found
filename, fileext = os.path.splitext(__file__)
if (os.path.exists(f'{filename}_addon{fileext}')):
    exec(open(f'{filename}_addon{fileext}').read())

if args.orca is None and os.getenv('PLOTLY_ORCA') is not None:
    args.orca = os.getenv('PLOTLY_ORCA')
""")

if args.orca is not None:
    plotScript.write(f"""
# An initial orca version is provided by the plot author
if args.orca is None:
    args.orca = '{args.orca}'
""")


plotScript.write(f"""
# Output and export below this line
plotPyMaster = "{os.path.realpath(__file__)}"
""")

plotScript.write("""

def getLatestPlotlyOrca(destination=".", quiet=False):
    fetchUrl = "https://github.com/plotly/orca/releases/latest"
    if not quiet:
        print(f"Fetching latest plotly orca release from {fetchUrl}", file=sys.stderr)
    lastReleases = urllib.request.urlopen(fetchUrl).read().decode()
    appImage = re.search(r'a href="(.+\\.AppImage)"', lastReleases)
    if not appImage:
        raise Exception('Could not locate latest plotly orca AppImage release at {fetchUrl}')
    fileAppImage = os.path.realpath(destination) + '/' + os.path.basename(appImage.group(1))
    urlAppImage = "https://github.com" + appImage.group(1)
    if not quiet:
        print(f"Downloading {urlAppImage} to {fileAppImage}", file=sys.stderr)
    urllib.request.urlretrieve(urlAppImage, fileAppImage)
    os.chmod(fileAppImage, 0o755)
    return fileAppImage


def getValidOrca(orcas=['orca']):
    if not isinstance(orcas, list):
        orcas = [orcas]
    norcas = []
    for orca in orcas:
        if '*' in orca:
            norcas += glob.glob(orca)
        else:
            norcas.append(orca)
    for orca in norcas:
        fBin = shutil.which(orca)
        if fBin is not None:
            fRun = subprocess.run([orca, '--help'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            if fRun.returncode == 0 and "Plotly's image-exporting utilities" in fRun.stdout.decode():
                return fBin


def exportFigure(fig, width, height, exportFile, orca='orca'):
    if exportFile.lower().endswith('.html'):
        fig.write_html(exportFile)
        return
    else:
        tmpFd, tmpFile = tempfile.mkstemp()
        try:
            exportFile = os.path.abspath(exportFile)
            exportDir = os.path.dirname(exportFile)
            exportFilename = os.path.basename(exportFile)
            fileName, fileExtension = os.path.splitext(exportFilename)
            fileExtension = fileExtension.lstrip('.').lower()
            fileExtension = 'jpeg' if fileExtension == 'jpg' else fileExtension
            go.Figure(fig).write_json(tmpFile)
            cmd = [orca, 'graph', tmpFile, '--output-dir', exportDir, '--output', fileName, '--format', fileExtension]
            if width is not None:
                cmd.extend(['--width', f'{width}'])
            if height is not None:
                cmd.extend(['--height', f'{height}'])
            exportRun = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if exportRun.returncode != 0:
                print(f'ERROR: failed to export figure to {exportFile}! Unsupported file format?')
                print(exportRun.stderr.decode('utf-8'))
                exit(1)
            if f'{fileName}.{fileExtension}' != exportFilename:
                os.rename(f'{exportDir}/{fileName}.{fileExtension}', f'{exportDir}/{exportFilename}')
        finally:
            os.remove(tmpFile)


if len(args.output) > 0 and not all([x.lower().endswith('.html') for x in args.output]):
    plotPyDir = os.path.dirname(plotPyMaster)
    searchSpace = ([args.orca] if args.orca is not None else [])
    searchSpace.extend(['orca', 'plotly-orca', plotPyDir + '/orca*.AppImage', './orca*.AppImage'])
    args.orca = getValidOrca(searchSpace)

    if args.orca is None:
        if platform.system() != 'Linux':
            print("Automatic installation of plotly orca not supported for your platform which is required to export the requested output format. Please manually install plotly orca from https://github.com/plotly/orca and make it available in your environment.")
            exit(0)

        if args.quiet or not (input("Download latest plotly orca for output format support? [Y/n]: ").lower() in ['y', 'yes', '']):
            print("Requested output format requires plotly orca, please provide it manually from https://github.com/plotly/orca!")
            exit(0)

        args.orca = getLatestPlotlyOrca(plotPyDir if os.path.isdir(plotPyDir) else ".")

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

    for output in args.output:
        exportFigure(fig, args.width, args.height, output, args.orca)
        print(f'Saved to {output}')
        if not args.quiet:
            try:
                subprocess.check_call([openWith, output], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            except Exception:
                print(f'Could not open {output}!')
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
else:
    for s in secondaryScripts:
        shutil.copy(args.script, s)
    if not args.quiet:
        for s in [args.script] + secondaryScripts:
            print(f"Script saved to {s}")
