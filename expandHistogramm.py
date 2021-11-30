#!/usr/bin/env python3
#
import argparse
import csv
import sys
import os
import xopen
import collections
import fcntl

F_SETPIPE_SZ = 1031 if not hasattr(fcntl, "F_SETPIPE_SZ") else fcntl.F_SETPIPE_SZ
F_GETPIPE_SZ = 1032 if not hasattr(fcntl, "F_GETPIPE_SZ") else fcntl.F_GETPIPE_SZ

parser = argparse.ArgumentParser(description="Expand compressed histogramm notation")
parser.add_argument("input", nargs="?", help="compressed histogramm csv")
parser.add_argument("-vd", "--value-delimiter", help="histogramm delimiter between category and value (default '%(default)s')", default='/')
parser.add_argument("-bd", "--bucket-delimiter", help="histogramm delimiter between category and value (default '%(default)s')", default=':')
parser.add_argument("--delimiter", help="csv delimiter (default '%(default)s')", default=';')
parser.add_argument("--flatten", action="store_true", help="output flat histogramm", default=False)
parser.add_argument("--no-header", action="store_true", help="input file does not contain a header", default=False)
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

entryColumn = ""

if not args.no_header:
    for header in csvFile:
        if header[0].startswith('#'):
            continue
        entryColumn = header[0]
        break

fullHist = []
flatHist = collections.Counter()
seenIndex = {}

for line in csvFile:
    if line[0].startswith('#'):
        continue
    if len(line) < 2:
        continue

    values = {k: int(v) for k, v in (x.split(args.bucket_delimiter) for x in line[1].split(args.value_delimiter))}
    flatHist.update(values)
    if not args.flatten:
        if line[0] in seenIndex:
            prevValues = fullHist[seenIndex[line[0]]]
            del prevValues[entryColumn]
            fullHist[seenIndex[line[0]]] = {entryColumn: line[0], **(collections.Counter(prevValues) + collections.Counter(values))}
        else:
            fullHist.append({entryColumn: line[0], **values})
            seenIndex[line[0]] = len(fullHist) - 1

outputFile = sys.stdout if not args.output else xopen.xopen(args.output, 'w')

header = list(flatHist)
if all(x.isdigit() for x in header):
    header.sort(key=int)

header = [entryColumn] + header

dictWriter = csv.DictWriter(outputFile, delimiter=args.delimiter, fieldnames=header, extrasaction='ignore')
dictWriter.writeheader()

if args.flatten:
    dictWriter.writerow({entryColumn: 'flat', **flatHist})
else:
    dictWriter.writerows(fullHist)

if (args.output):
    outputFile.close()
