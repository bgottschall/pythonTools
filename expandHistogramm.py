#!/usr/bin/env python3
#
import argparse
import csv
import sys
import os
import xopen
import collections
import functools
import itertools
import operator
import fcntl

F_SETPIPE_SZ = 1031 if not hasattr(fcntl, "F_SETPIPE_SZ") else fcntl.F_SETPIPE_SZ
F_GETPIPE_SZ = 1032 if not hasattr(fcntl, "F_GETPIPE_SZ") else fcntl.F_GETPIPE_SZ

parser = argparse.ArgumentParser(description="Expand compressed histogramm notation")
parser.add_argument("input", nargs="?", help="compressed histogramm csv")
parser.add_argument("-c", "--column", help="parse this column (default %(default)s)", type=int, default=1)
parser.add_argument("-vd", "--value-delimiter", help="histogramm delimiter between category and value (default '%(default)s')", default='/')
parser.add_argument("-bd", "--bucket-delimiter", help="histogramm delimiter between category and value (default '%(default)s')", default=':')
parser.add_argument("--prebins", type=str, help="parse each line with prepared bins", default=None, nargs='*')
parser.add_argument("--force-prebins", default=False, action="store_true", help="force prebins instead of failing")
parser.add_argument("--delimiter", help="csv delimiter (default '%(default)s')", default=';')
parser.add_argument("--flatten", action="store_true", help="output flat histogramm", default=False)
parser.add_argument("--no-header", action="store_true", help="input file does not contain a header", default=False)
parser.add_argument("-o", "--output", help="output file (default stdout)", default=None)

args = parser.parse_args()

if args.input and not os.path.exists(args.input):
    print("ERROR: csv input file not found!")
    parser.print_help()
    sys.exit(1)

if args.column < 0:
    print("ERROR: process column cannot be negative!")
    parser.print_help()
    sys.exit(1)

if args.prebins is not None:
    preBins = []
    for x in args.prebins:
        if x.isnumeric():
            preBins.append(x)
        else:
            try:
                p = [int(t) if t else None for t in x.split(':')]
                p[1] += 1
                preBins.extend(range(int(p[1]))[slice(*p)])
            except Exception:
                raise Exception(f"Could not parse prebin {x}")
    args.prebins = [str(k) for k in preBins]
    del preBins


if not args.input:
    try:
        fcntl.fcntl(sys.stdin.fileno(), F_SETPIPE_SZ, int(open("/proc/sys/fs/pipe-max-size", 'r').read()))
    except Exception:
        pass
    fInput = sys.stdin
else:
    fInput = xopen.xopen(args.input, 'r')

csvFile = csv.reader(fInput, delimiter=args.delimiter)

inputHeader = None
if not args.no_header:
    for header in csvFile:
        if header[0].startswith('#'):
            continue
        inputHeader = header
        break

if not args.no_header and inputHeader is None:
    raise Exception("Could not find header row")

if args.flatten:
    inputHeader = []

outputFile = None
dictWriter = None
fullHist = []
flatHist = collections.Counter() if args.prebins is None else collections.Counter({k: 0 for k in args.prebins})

if args.prebins is not None:
    outputFile = sys.stdout if not args.output else xopen.xopen(args.output, 'w')

    header = list(flatHist.keys())
    if all(x.isdigit() for x in header):
        header.sort(key=int)
    header = inputHeader[:args.column] + header + inputHeader[args.column + 1:]

    dictWriter = csv.DictWriter(outputFile, delimiter=args.delimiter, fieldnames=header, extrasaction='ignore')

    if not args.no_header:
        dictWriter.writeheader()

for line in csvFile:
    if line[0].startswith('#'):
        continue
    if inputHeader is None:
        inputHeader = list(range(len(line)))

    if args.prebins is not None and not args.force_prebins and any(False if k in args.prebins else True for k, _ in (x.split(args.bucket_delimiter) for x in line[args.column].split(args.value_delimiter))):
      invalidBins = [k for k, _ in (x.split(args.bucket_delimiter) for x in line[args.column].split(args.value_delimiter)) if k not in args.prebins]
      raise Exception(f'ERROR: input data contained bins {invalidBins} which were not provided over the prebins')

    values = functools.reduce(lambda a, b: a + collections.Counter(b), ({k: float(v)} for k, v in (x.split(args.bucket_delimiter) for x in line[args.column].split(args.value_delimiter)) if args.prebins is None or k in args.prebins), collections.Counter())

    flatHist.update(values)

    if not args.flatten:
        outDict = {
            **{k: v for k, v in zip(inputHeader[:args.column], line[:args.column])},
            **values,
            **{k: v for k, v in zip(inputHeader[args.column + 1:], line[args.column + 1:])}
        }
        if args.prebins is not None:
            dictWriter.writerow(outDict)
        else:
            fullHist.append(outDict)

if dictWriter is None:
    outputFile = outputFile if outputFile is not None else sys.stdout if not args.output else xopen.xopen(args.output, 'w')

    header = list(flatHist.keys())
    if all(x.isdigit() for x in header):
        header.sort(key=int)
    header = inputHeader[:args.column] + header + inputHeader[args.column + 1:]

    dictWriter = csv.DictWriter(outputFile, delimiter=args.delimiter, fieldnames=header, extrasaction='ignore')

if args.prebins is None:
    if not args.no_header:
        dictWriter.writeheader()
    if not args.flatten:
        dictWriter.writerows(fullHist)

if args.flatten:
    outDict = {
        **{k: v for k, v in zip(inputHeader[:args.column], line[:args.column])},
        **flatHist,
        **{k: v for k, v in zip(inputHeader[args.column + 1:], line[args.column + 1:])}
    }
    dictWriter.writerow(outDict)

if outputFile is not None and args.output:
    outputFile.close()
