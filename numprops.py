#!/usr/bin/env python3

from argparse import ArgumentParser
import numpy
import re
from scipy import stats


def isFloat(val):
    if val is None:
        return False
    try:
        float(val)
        return True
    except ValueError:
        return False


def numbersFromStdIn():
    import sys
    stdinSet = []
    for line in sys.stdin.readlines():
        stdinSet.extend(line.split())
    return numpy.array([float(x) for x in stdinSet if isFloat(x)])


# props = { propname : { 'label' : outputlabel, 'func' : propertyfunction(l1 or l1, arg), 'secondary' : }}
# % in property name is taken as argument

props = {
    'count':      {'label': 'Count',     'secondary': False, 'argument': False, 'func': lambda l1, l2, arg: len(l1)    },
    'sum':        {'label': 'Sum',       'secondary': False, 'argument': False, 'func': lambda l1, l2, arg: numpy.sum(l1)},
    'min':        {'label': 'Min',       'secondary': False, 'argument': False, 'func': lambda l1, l2, arg: numpy.min(l1)},
    'max':        {'label': 'Max',       'secondary': False, 'argument': False, 'func': lambda l1, l2, arg: numpy.max(l1)},
    'q1':         {'label': 'Q1',        'secondary': False, 'argument': False, 'func': lambda l1, l2, arg: numpy.percentile(l1, 25)},
    'q2':         {'label': 'Q2',        'secondary': False, 'argument': False, 'func': lambda l1, l2, arg: numpy.percentile(l1, 50)},
    'q3':         {'label': 'Q3',        'secondary': False, 'argument': False, 'func': lambda l1, l2, arg: numpy.percentile(l1, 75)},
    'p%':         {'label': 'P%',        'secondary': False, 'argument': True,  'func': lambda l1, l2, arg: numpy.percentile(l1, float(arg))},
    'avg':        {'label': 'Avg',       'secondary': False, 'argument': False, 'func': lambda l1, l2, arg: numpy.mean(l1)},
    'std':        {'label': 'σ',         'secondary': False, 'argument': False, 'func': lambda l1, l2, arg: numpy.std(l1)},
    'var':        {'label': 'σ²',        'secondary': False, 'argument': False, 'func': lambda l1, l2, arg: numpy.var(l1)},
    'pvalue':     {'label': 'P-Value',   'secondary': True,  'argument': False, 'func': lambda l1, l2, arg: stats.ttest_ind(l1, l2, equal_var=False)[1]},
    'spearmanr':  {'label': 'SpearmanR', 'secondary': True,  'argument': False, 'func': lambda l1, l2, arg: stats.spearmanr(l1, l2)[0]},
    'pearsonr':   {'label': 'PearsonR',  'secondary': True,  'argument': False, 'func': lambda l1, l2, arg: stats.pearsonr(l1, l2)[0]},
}

defaultProps = ['count', 'sum', 'min', 'max', 'q2', 'avg', 'std']

parser = ArgumentParser(description="output number properties from numbers passed or read from stdin")
parser.add_argument("--stdin", help=f"read from stdin even if numbers are provided", default=False, action="store_true")
parser.add_argument("--precision", help=f"force a specific precision", type=int, default=None)
parser.add_argument("-p", "--properties", help=f"format output (default {', '.join(defaultProps)}) (valid {', '.join(list(props.keys())).replace('%','%%')})", type=str.lower, nargs="+", default=defaultProps)
parser.add_argument("-q", "--quiet", help="minimal output", default=False, action="store_true")
parser.add_argument("--secondary", help="secondary number set used e.g. to calculate p-value statistics", default=[], nargs='*')
parser.add_argument("--debug", help="turn on debug output", action="store_true")
parser.add_argument("primary", help="numbers to calculate properties on (default read from stdin)", default=[], nargs='*')
args = parser.parse_args()

l1 = None
l2 = None
stdinConsumed = False

if args.precision is not None and args.precision < 0:
    args.precision = None

if len(args.primary) == 0:
    if args.debug:
        print('[DEBUG] no primary number set given, reading from stdin')
    l1 = numbersFromStdIn()
    stdinConsumed = True
else:
    l1 = numpy.array([float(x) for x in args.primary if isFloat(x)])

if len(l1) == 0:
    raise Exception(f"Could not parse any numbers")

if args.debug:
    print(f'[DEBUG][L1] {l1}')

if len(args.secondary) > 0:
    l2 = numpy.array([float(x) for x in args.secondary if isFloat(x)])
    if args.debug:
        print(f'[DEBUG][L2] {l2}')

results = []
labels = []

for p in args.properties:
    arg = None
    prop = None
    if p in props and not props[p]['argument']:
        prop = p
    else:
        for cp in props:
            if props[cp]['argument']:
                pattern = re.compile(re.escape(cp).replace('\%', '(.+)', 1))  # [-+]?\d*\.\d+|\d+)', 1))
                reres = pattern.search(p)
                if reres and len(reres.groups()) == 1:
                    arg = reres.group(1)
                    prop = cp
                    break
    if prop is None:
        raise Exception(f"Could not find property '{p}'")

    if props[prop]['secondary'] and l2 is None:
        if stdinConsumed:
            raise Exception(f"Property '{p}' requires a secondary number set, provided via stdin or --secondary")
        if args.debug:
            print('[DEBUG] no secondary number set given, reading from stdin')
        l2 = numbersFromStdIn()
        if args.debug:
            print(f'[DEBUG][L2] {l2}')

    if props[prop]['argument']:
        if args.debug:
            print(f'[DEBUG][ARG] {arg}')
        labels.append(props[prop]['label'].replace('%', arg, 1))
    else:
        labels.append(props[prop]['label'])

    results.append(props[prop]['func'](l1, l2, arg))
    if args.debug:
        print(f'[DEBUG][{p}] {results[-1]}')

if args.precision is not None:
    if args.precision == 0:
        results = ['{:d}'.format(int(x)) for x in results]
    else:
        results = [f'{{:.{args.precision}f}}'.format(x) for x in results]
else:
    results = ['{:}'.format(x) for x in results]

if args.quiet:
    print(' '.join(results))
else:
    lengths = [max(len(str(x)), len(str(y))) for x, y in zip(labels, results)]
    rowFormat = ' '.join([f"{{:>{length}}}" for length in lengths])
    print(rowFormat.format(*labels))
    print(rowFormat.format(*results))
