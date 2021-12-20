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
parser.add_argument("--slice", type=SliceType(), default=SliceType()('1:'), help="slice histogram (default '1:')",)
parser.add_argument("--delimiter", help="csv delimiter (default '%(default)s')", default=';')
parser.add_argument("--flatten", choices=['buckets', 'counts', 'items'], help="output flat histogramm", default=None, nargs="*")
parser.add_argument("--items", nargs="*", help="select items", default=False)
parser.add_argument("--buckets", nargs="*", help="select buckets", default=False)
parser.add_argument("-o", "--output", help="output file (default stdout)", default=None)

args = parser.parse_args()

if args.input and not os.path.exists(args.input):
    print("ERROR: csv input file not found!")
    parser.print_help()
    sys.exit(1)

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

selector = None

if args.buckets:
    selector = [0] + [i for i, x in enumerate(header) if i > 0 and x in args.buckets]

if (args.flatten == 'buckets' or args.flatten == 'items+buckets') and not all(isFloat(x) for x in header[args.slice]):
    raise Exception('Flatten buckets only works with numeric header')

hasItems = 0 not in (args.slice.indices(1) if isinstance(args.slice, slice) else [args.slice])

outputFile = sys.stdout if not args.output else xopen.xopen(args.output, 'w')

if args.flatten is not None and 'items' in args.flatten and not any(x in args.flatten for x in ['buckets', 'counts']):
    outputFile.write(args.delimiter.join([header[0] if hasItems else ''] + header[args.slice]) + '\n')
else:
    outputFile.write(args.delimiter.join([header[0] if hasItems else ''] + [x for x in ['counts', 'buckets'] if x in args.flatten]) + '\n')

flatHist = [0] * (len(header[args.slice]) - 1)
flat = {
    'buckets': 0,
    'counts': 0
}


def parseNormal(line):
    outputFile.write(args.delimiter.join([line[0] if hasItems else str(i)] + line[args.slice]) + '\n')


def parseCounts(line):
    outputFile.write(args.delimiter.join([line[0] if hasItems else str(i), str(sum([float(i) for i in line[args.slice] if len(i) > 0]))]) + '\n')


def parseBuckets(line):
    outputFile.write(args.delimiter.join([line[0] if hasItems else str(i), str(sum([float(h) * float(i) for (h, i) in zip(header[args.slice], line[args.slice]) if len(i) > 0]))]) + '\n')


def parseCountsBuckets(line):
    outputFile.write(args.delimiter.join([line[0] if hasItems else str(i), str(sum([float(i) for i in line[args.slice] if len(i) > 0])), str(sum([float(h) * float(i) for (h, i) in zip(header[args.slice], line[args.slice]) if len(i) > 0]))]) + '\n')


def parseItems(line):
    global flatHist
    flatHist = [(p + float(v) if len(v) > 0 else p) for (p, v) in zip(flatHist, line[args.slice])]


def parseItemsBuckets(line):
    global flat
    flat['buckets'] += sum([float(h) * float(i) for (h, i) in zip(header[args.slice], line[args.slice]) if len(i) > 0])


def parseItemsCounts(line):
    global flat
    flat['counts'] += sum([float(i) for i in line[args.slice] if len(i) > 0])


def parseItemsCountsBuckets(line):
    global flat
    flat['counts'] += sum([float(i) for i in line[args.slice] if len(i) > 0])
    flat['buckets'] += sum([float(h) * float(i) for (h, i) in zip(header[args.slice], line[args.slice]) if len(i) > 0])


if 'items' in args.flatten:
    if all(x in args.flatten for x in ['counts', 'buckets']):
        parser = parseItemsCountsBuckets
    elif 'counts' in args.flatten:
        parser = parseItemsCounts
    elif 'buckets' in args.flatten:
        parser = parseItemsCounts
    else:
        parser = parseItems
else:
    if all(x in args.flatten for x in ['counts', 'buckets']):
        parser = parseCountsBuckets
    elif 'buckets' in args.flatten:
        parser = parseBuckets
    elif 'counts' in args.flatten:
        parser = parseCounts
    else:
        parser = parseNormal

for i, line in enumerate(csvFile):
    if line[0].startswith('#'):
        continue
    if len(line) < 2:
        continue
    if args.items and line[0] not in args.items:
        continue

    if selector is not None:
        line = [line[i] for i in selector]

    parser(line)

if 'items' in args.flatten:
    if not any(x in args.flatten for x in ['counts', 'buckets']):
        outputFile.write(args.delimiter.join(["items"] + [str(f) for f in flatHist]) + '\n')
    else:
        if all(x in args.flatten for x in ['counts', 'buckets']):
            outputFile.write(args.delimiter.join(["items", str(flat['counts']), str(flat['buckets'])]) + '\n')
        elif 'counts' in args.flatten:
            outputFile.write(args.delimiter.join(["items", str(flat['counts'])]) + '\n')
        else:
            outputFile.write(args.delimiter.join(["items", str(flat['buckets'])]) + '\n')

if (args.output):
    outputFile.close()
