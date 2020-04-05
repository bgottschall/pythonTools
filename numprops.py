#!/usr/bin/env python3

from argparse import ArgumentParser
import numpy


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
    'var': {'label': 'σ²', 'func': numpy.var},
    'pval': {'label': 'P-Value', 'func': lambda x, y: stats.ttest_ind(x, y, equal_var=False)[1]}
}

defaultProps = ['count', 'sum', 'min', 'max', 'q2', 'avg', 'std']

parser = ArgumentParser(description="output number properties from numbers passed or read from stdin")
parser.add_argument("--stdin", help=f"read from stdin even if numbers are provided", default=False, action="store_true")
parser.add_argument("--precision", help=f"force a specific precision", type=int, default=None)
parser.add_argument("-p", "--properties", help=f"format output (default {', '.join(defaultProps)}) (valid {', '.join(list(props.keys())).replace('%','%%')})", type=str.lower, nargs="+", default=defaultProps)
parser.add_argument("-q", "--quiet", help="minimal output", default=False, action="store_true")
parser.add_argument("--others", help="other number set used e.g. to calculate p-value statistics", default=[], nargs='*')
parser.add_argument("primary", help="numbers to calculate properties on (default read from stdin)", default=[], nargs='*')
args = parser.parse_args()

pValue = False
if 'pval' in args.properties:
    # One number set can be read from stdin, if both primary and other is empty we cannot calculate p-value statistics
    if len(args.primary) == 0 and len(args.others) == 0:
        raise Exception('two number sets are required to calculate p-value statistics, provide them via primary and --others or stdin')
    # One number set is provided, read the other one from stdin
    if len(args.primary) == 0 or len(args.others) == 0:
        args.stdin = True
    from scipy import stats
    pValue = True

if len(args.primary) == 0:
    args.stdin = True

if args.stdin:
    import sys
    stdinSet = []
    for line in sys.stdin.readlines():
        stdinSet.extend(line.split())
    if pValue and len(args.others) == 0:
        args.others = stdinSet
    else:
        args.primary.extend(stdinSet)

args.primary = numpy.array([float(x) for x in args.primary if isFloat(x)])
if pValue:
    args.others = numpy.array([float(x) for x in args.others if isFloat(x)])

if pValue and (len(args.primary) <= 1 or len(args.others) <= 1):
    raise Exception('number sets too small to calculate p-value statistics!')
    exit(1)

if (len(args.primary) == 0):
    exit(1)

results = []
labels = []

for p in args.properties:
    if p[0] == 'p' and isFloat(p[1:]):
        p, percentile = 'p%', p[1:]
        labels.append(props[p]['label'].replace('%', percentile))
        results.append(props[p]['func'](args.primary, float(percentile)))
    else:
        if p not in props:
            raise Exception(f"Could not find property '{p}'")
        labels.append(props[p]['label'])
        if p == 'pval':
            results.append(props[p]['func'](args.primary, args.others))
        else:
            results.append(props[p]['func'](args.primary))

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
