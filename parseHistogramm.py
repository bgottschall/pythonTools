#!/usr/bin/env python3
import argparse
import csv
import sys
import os
import xopen
import fcntl

F_SETPIPE_SZ = 1031 if not hasattr(fcntl, "F_SETPIPE_SZ") else fcntl.F_SETPIPE_SZ
F_GETPIPE_SZ = 1032 if not hasattr(fcntl, "F_GETPIPE_SZ") else fcntl.F_GETPIPE_SZ


def isFloat(val):
    if val is None:
        return False
    try:
        float(val)
        return True
    except ValueError:
        return False


defaultSliceTypeTranslator = {'all': slice(None)}


def SliceType(translator=defaultSliceTypeTranslator):
    def str2slice(value):
        if value in translator:
            return translator[value]
        try:
            return int(value)
        except ValueError:
            tSection = [int(s) if s else None for s in value.split(':')]
            if len(tSection) > 3:
                raise ValueError(f'{value} is not a valid slice notation')
            return slice(*tSection)
    return str2slice


def isSliceType(value, translator=defaultSliceTypeTranslator):
    if value is None:
        return False
    try:
        SliceType(translator)(value)
        return True
    except Exception:
        return False


parser = argparse.ArgumentParser(description="Expand compressed histogramm notation")
parser.add_argument("input", nargs="?", help="compressed histogramm csv")
parser.add_argument("--filter-columns", default=[], type=SliceType(), nargs='*', help='filter based on these columns')
parser.add_argument("--filter-mode", choices=['any', 'all'], default='any', help='either a value must match in any of the columns of all columns must contain a filter value')
parser.add_argument("--filter-data", default=[], type=str, nargs='*', help='filter based on this data')
parser.add_argument("--slice", type=str, default='1:', help="slice histogram (default '1:')",)
parser.add_argument("--delimiter", help="csv delimiter (default '%(default)s')", default=';')
parser.add_argument("--flatten", choices=['buckets', 'counts', 'items'], help="output flat histogramm", default=[], nargs="*")
parser.add_argument("--item-columns", nargs="*", help="set item column (default %(default)s)", default=None)
parser.add_argument("--select-buckets", nargs="*", help="select buckets", default=False)
parser.add_argument("-o", "--output", help="output file (default stdout)", default=None)

args = parser.parse_args()

if args.input and not os.path.exists(args.input):
    print("ERROR: csv input file not found!")
    parser.print_help()
    sys.exit(1)

if (len(args.filter_columns) > 0 and len(args.filter_data) == 0) or (len(args.filter_columns) == 0 and len(args.filter_data) > 0):
    raise Exception('Filtering requires --filter-columns and --filter-data!')

if args.item_columns is not None:
    itemColumns = []
    for x in args.item_columns:
        if x.isnumeric():
            itemColumns.append(slice(int(x), int(x) + 1))
        else:
            x = [int(y) if y.isnumeric() else None for y in x.split(':')]
            if len(x) > 3 or all(y is None for y in x):
                raise Exception('invalid item columns parameter')
            itemColumns.append(slice(*x))
    args.item_columns = itemColumns

if args.slice.isnumeric():
    args.slice = [int(args.slice), int(args.slice) + 1]
else:
    args.slice = [int(x) if x.isnumeric() else None for x in args.slice.split(':')]
    if len(args.slice) != 2:
        raise Exception('Invalid histogram slice')
    args.slice[0] = args.slice[0] if args.slice[0] is not None else 0

if not args.input:
    try:
        fcntl.fcntl(sys.stdin.fileno(), F_SETPIPE_SZ, int(open("/proc/sys/fs/pipe-max-size", 'r').read()))
    except Exception:
        pass
    fInput = sys.stdin
else:
    fInput = xopen.xopen(args.input, 'r')

csvFile = csv.reader(fInput, delimiter=args.delimiter)

header = None
for header in csvFile:
    if header[0].startswith('#'):
        continue
    break

if header is None:
    raise Exception('Could not find a histogram header!')

args.slice[1] = args.slice[1] if args.slice[1] is not None else len(header)
if args.slice[0] == args.slice[1]:
    raise Exception('Invalid histogram slice range')

itemsHeader = None if args.item_columns is None else [x for sx in [header[s] for s in args.item_columns] for x in sx]
print(itemsHeader)
if 'buckets' in args.flatten and not all(isFloat(x) for x in header[slice(*args.slice)]):
    raise Exception('Flatten buckets only works with numeric header')


selector = None
if args.select_buckets:
    selector = [i for i, x in enumerate(header[slice(*args.slice)]) if x in args.select_buckets]

outputFile = sys.stdout if not args.output else xopen.xopen(args.output, 'w')

if len(args.flatten) == 0:
    outputFile.write(args.delimiter.join(header) + '\n')
elif 'items' not in args.flatten and any(x in args.flatten for x in ['counts', 'buckets']):
    outputFile.write(args.delimiter.join(header[:args.slice[0]] + [x for x in ['counts', 'buckets'] if x in args.flatten]) + '\n')
elif 'items' in args.flatten and any(x in args.flatten for x in ['counts', 'buckets']):
    outputFile.write(args.delimiter.join(((itemsHeader if itemsHeader is not None else ['all']) + [x for x in ['counts', 'buckets'] if x in args.flatten])) + '\n')
else:
    outputFile.write(args.delimiter.join((itemsHeader if itemsHeader is not None else ['all']) + header[slice(*args.slice)]) + '\n')

flatHist = {}
flat = {}


def parseNormal(line, itemIndex, itemHeaders, itemValues):
    outputFile.write(args.delimiter.join(line[:args.slice[0]] + itemValues + line[args.slice[1]:]) + '\n')


def parseCounts(line, itemIndex, itemHeaders, itemValues):
    outputFile.write(args.delimiter.join(line[:args.slice[0]] + [str(sum([float(i) for i in itemValues if len(i) > 0]))] + line[args.slice[1]:]) + '\n')


def parseBuckets(line, itemIndex, itemHeaders, itemValues):
    outputFile.write(args.delimiter.join(line[:args.slice[0]] + [str(sum([float(h) * float(i) for (h, i) in zip(itemHeaders, itemValues) if len(i) > 0]))] + line[args.slice[1]:]) + '\n')


def parseCountsBuckets(line, itemIndex, itemHeaders, itemValues):
    outputFile.write(args.delimiter.join(line[:args.slice[0]] + [str(sum([float(i) for i in itemValues if len(i) > 0])), str(sum([float(h) * float(i) for (h, i) in zip(itemHeaders, itemValues) if len(i) > 0]))] + line[args.slice[1]:]) + '\n')


def parseItems(line, itemIndex, itemHeaders, itemValues):
    global flatHist
    if itemIndex not in flatHist:
        flatHist[itemIndex] = [0] * len(itemValues)
    flatHist[itemIndex] = [(p + float(v) if len(v) > 0 else p) for (p, v) in zip(flatHist[itemIndex], itemValues)]


def parseItemsBuckets(line, itemIndex, itemHeaders, itemValues):
    global flat
    if itemIndex not in flat:
        flat[itemIndex] = {'counts': 0, 'buckets': 0}
    flat[itemIndex]['buckets'] += sum([float(h) * float(i) for (h, i) in zip(itemHeaders, itemValues) if len(i) > 0])


def parseItemsCounts(line, itemIndex, itemHeaders, itemValues):
    global flat
    if itemIndex not in flat:
        flat[itemIndex] = {'counts': 0, 'buckets': 0}
    flat[itemIndex]['counts'] += sum([float(i) for i in itemValues if len(i) > 0])


def parseItemsCountsBuckets(line, itemIndex, itemHeaders, itemValues):
    global flat
    if itemIndex not in flat:
        flat[itemIndex] = {'counts': 0, 'buckets': 0}
    flat[itemIndex]['counts'] += sum([float(i) for i in itemValues if len(i) > 0])
    flat[itemIndex]['buckets'] += sum([float(h) * float(i) for (h, i) in zip(itemHeaders, itemValues) if len(i) > 0])


if 'items' in args.flatten:
    if all(x in args.flatten for x in ['counts', 'buckets']):
        parser = parseItemsCountsBuckets
    elif 'counts' in args.flatten:
        parser = parseItemsCounts
    elif 'buckets' in args.flatten:
        parser = parseItemsBuckets
    else:
        parser = parseItems
else:
    if all(x in args.flatten for x in ['counts', 'buckets']):
        parser = parseCountsBuckets
    elif 'counts' in args.flatten:
        parser = parseCounts
    elif 'buckets' in args.flatten:
        parser = parseBuckets
    else:
        parser = parseNormal


applyFilter = len(args.filter_columns) > 0
filterSlices = [s if isinstance(s, slice) else slice(s, s + 1) for s in args.filter_columns]
filterFunc = any if args.filter_mode == 'any' else all
itemHeaders = header[slice(*args.slice)]
if selector is not None:
    itemHeaders = [itemHeaders[i] for i in selector]

for i, line in enumerate(csvFile):
    if line[0].startswith('#'):
        continue
    if len(line) < 2:
        continue

    if applyFilter and not filterFunc(v in args.filter_data for lv in [line[slc] for slc in filterSlices] for v in lv):
        continue

    itemValues = line[slice(*args.slice)]
    if selector is not None:
        itemValues = [itemValues[i] for i in selector]

    itemsIndex = 'all' if args.item_columns is None else args.delimiter.join([x for sx in [line[s] for s in args.item_columns] for x in sx])

    parser(line, itemsIndex, itemHeaders, itemValues)

if 'items' in args.flatten:
    if not any(x in args.flatten for x in ['counts', 'buckets']):
        for itemIndex, itemFlat in flatHist.items():
            outputFile.write(args.delimiter.join([itemIndex] + [str(f) for f in itemFlat]) + '\n')
    else:
        if all(x in args.flatten for x in ['counts', 'buckets']):
            for itemIndex, itemFlat in flat.items():
                outputFile.write(args.delimiter.join([itemIndex, str(itemFlat['counts']), str(itemFlat['buckets'])]) + '\n')
        elif 'counts' in args.flatten:
            for itemIndex, itemFlat in flat.items():
                outputFile.write(args.delimiter.join([itemIndex, str(itemFlat['counts'])]) + '\n')
        else:
            for itemIndex, itemFlat in flat.items():
                outputFile.write(args.delimiter.join([itemIndex, str(itemFlat['buckets'])]) + '\n')

if (args.output):
    outputFile.close()
