#!/usr/bin/env python3

from argparse import ArgumentParser
import numpy
import sys


def isFloat(val):
    if val is None:
        return False
    try:
        float(val)
        return True
    except ValueError:
        return False


props = {
    'count': {'label': 'Count', 'func': len},
    'sum': {'label': 'Sum', 'func': numpy.sum},
    'min': {'label': 'Min', 'func': numpy.min},
    'max': {'label': 'Max', 'func': numpy.max},
    'q1': {'label': 'Q1', 'func': lambda x: numpy.percentile(x, 25)},
    'q2': {'label': 'Q2', 'func': lambda x: numpy.percentile(x, 50)},
    'q3': {'label': 'Q3', 'func': lambda x: numpy.percentile(x, 75)},
    'p%': {'label': 'P%', 'func': lambda x, y: numpy.percentile(x, y)},
    'avg': {'label': 'Avg', 'func': numpy.mean},
    'std': {'label': 'σ', 'func': numpy.std},
    'var': {'label': 'σ²', 'func': numpy.var}
}

defaultProps = ['count', 'sum', 'min', 'max', 'q2', 'avg', 'std']

parser = ArgumentParser(description="output number properties from numbers passed or read from stdin")
parser.add_argument("--stdin", help=f"read from stdin even if numbers are provided", default=False, action="store_true")
parser.add_argument("--precision", help=f"force a specific precision", type=int, default=None)
parser.add_argument("-p", "--properties", help=f"format output (default {', '.join(defaultProps)}) (valid {', '.join(list(props.keys())).replace('%','%%')})", nargs="+", default=defaultProps)
parser.add_argument("-q", "--quiet", help="minimal output", default=False, action="store_true")
parser.add_argument("numbers", help="numbers to calculate properties on (default read from stdin)", default=None, nargs='*')
args = parser.parse_args()

if args.numbers is None or len(args.numbers) == 0:
    args.stdin = True
    args.numbers = []

if args.stdin:
    for line in sys.stdin.readlines():
        args.numbers.extend(line.split())

numbers = numpy.array([float(x) for x in args.numbers if isFloat(x)])

if (len(numbers) == 0):
    exit(1)

results = []
labels = None if args.quiet else []

for f in args.properties:
    f = f.lower()
    parm = None

    if f[0] == 'p' and isFloat(f[1:]):
        parm = f[1:]
        f = 'p%'

    if f not in props:
        raise Exception(f"Could not find property '{f}'")

    if not args.quiet:
        if parm is None:
            labels.append(props[f]['label'])
        else:
            labels.append(props[f]['label'].replace('%', parm))

    if parm is None:
        results.append(props[f]['func'](numbers))
    else:
        results.append(props[f]['func'](numbers, float(parm)))

resultFormat = '{:}'

if args.precision:
    resultFormat = f'{{:.{args.precision}f}}'

results = [resultFormat.format(x) for x in results]

if args.quiet:
    print(' '.join(results))
else:
    lengths = [max(len(str(x)), len(str(y))) for x, y in zip(labels, results)]
    rowFormat = ' '.join([f"{{:>{length}}}" for length in lengths])
    print(rowFormat.format(*labels))
    print(rowFormat.format(*results))
