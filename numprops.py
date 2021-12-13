#!/usr/bin/env python3

from argparse import ArgumentParser
import numpy
import re
import math
import os
from scipy import stats


def isFloat(val):
    if val is None:
        return False
    try:
        val = float(val)
        if math.isnan(val):
            return False
        return True
    except ValueError:
        return False


def numbersFromFile(filefd):
    nset = []
    for line in filefd.readlines():
        nset.extend(line.split())
    return numpy.array([float(x) for x in nset if isFloat(x)])


def numbersFromStdIn():
    import sys
    return numbersFromFile(sys.stdin)

    

quiet = False
formatter = '{:}'

# props = { propname : { 'label' : outputlabel, 'func' : propertyfunction(l1 or l1, arg), 'secondary' : }}
# % in property name is taken as argument


def polyfitProp(l1, l2, arg):
    arg = int(arg)
    fitted = numpy.polyfit(l1, l2, arg)
    res = ''
    if quiet:
        if formatter == '{:.0f}':
            res = str([int(formatter.format(x)) for x in fitted])
        else:
            res = str([float(formatter.format(x)) for x in fitted])
    else:
        res += formatter.format(fitted[0]) + f'x^{arg}'
        for i, a in enumerate(fitted[1:-1]):
            res += ((' + ' + formatter.format(a)) if a >= 0 else (' - ' + formatter.format(a * -1))) + f'x^{arg - i - 1}'
        res += (' + ' + formatter.format(fitted[-1])) if fitted[-1] >= 0 else (' - ' + formatter.format(fitted[-1] * -1))
    return res


props = {
    'count':      {'label': 'Count',      'secondary': False, 'argument': False, 'func': lambda l1, l2, arg: len(l1)},
    'sum':        {'label': 'Sum',        'secondary': False, 'argument': False, 'func': lambda l1, l2, arg: numpy.sum(l1)},
    'min':        {'label': 'Min',        'secondary': False, 'argument': False, 'func': lambda l1, l2, arg: numpy.min(l1)},
    'max':        {'label': 'Max',        'secondary': False, 'argument': False, 'func': lambda l1, l2, arg: numpy.max(l1)},
    'q1':         {'label': 'Q1',         'secondary': False, 'argument': False, 'func': lambda l1, l2, arg: numpy.percentile(l1, 25)},
    'q2':         {'label': 'Q2',         'secondary': False, 'argument': False, 'func': lambda l1, l2, arg: numpy.percentile(l1, 50)},
    'q3':         {'label': 'Q3',         'secondary': False, 'argument': False, 'func': lambda l1, l2, arg: numpy.percentile(l1, 75)},
    'p%':         {'label': 'P%',         'secondary': False, 'argument': True,  'func': lambda l1, l2, arg: numpy.percentile(l1, float(arg))},
    'hmean':      {'label': 'HMean',      'secondary': False, 'argument': False, 'func': lambda l1, l2, arg: stats.hmean(l1)},
    'gmean':      {'label': 'GMean',      'secondary': False, 'argument': False, 'func': lambda l1, l2, arg: stats.mstats.gmean(l1)},
    'mean':       {'label': 'Mean',       'secondary': False, 'argument': False, 'func': lambda l1, l2, arg: numpy.mean(l1)},
    'avg':        {'label': 'Avg',        'secondary': False, 'argument': False, 'func': lambda l1, l2, arg: numpy.mean(l1)},
    'std':        {'label': 'σ',          'secondary': False, 'argument': False, 'func': lambda l1, l2, arg: numpy.std(l1)},
    'var':        {'label': 'σ²',         'secondary': False, 'argument': False, 'func': lambda l1, l2, arg: numpy.var(l1)},
    'pvalue':     {'label': 'P-Value',    'secondary': True,  'argument': False, 'func': lambda l1, l2, arg: stats.ttest_ind(l1, l2, equal_var=False)[1]},
    'spearmanr':  {'label': 'SpearmanR',  'secondary': True,  'argument': False, 'func': lambda l1, l2, arg: stats.spearmanr(l1, l2)[0]},
    'pearsonr':   {'label': 'PearsonR',   'secondary': True,  'argument': False, 'func': lambda l1, l2, arg: stats.pearsonr(l1, l2)[0]},
    'polyfit%d':  {'label': 'PolyFit%D',  'secondary': True,  'argument': True,  'func': polyfitProp}
}

defaultProps = ['count', 'sum', 'min', 'max', 'q2', 'avg', 'std']

parser = ArgumentParser(description="output number properties from numbers passed or read from stdin")
parser.add_argument("--stdin", help="read from stdin even if numbers are provided", default=False, action="store_true")
parser.add_argument("--precision", help="force a specific precision", type=int, default=None)
parser.add_argument("-p", "--properties", help=f"format output (default {', '.join(defaultProps)}) (valid {', '.join(list(props.keys())).replace('%','%%')})", type=str.lower, nargs="+", default=defaultProps)
parser.add_argument("-q", "--quiet", help="minimal output", default=False, action="store_true")
parser.add_argument("--secondary", help="secondary number set used e.g. to calculate p-value statistics", default=[], nargs='*')
parser.add_argument("--debug", help="turn on debug output", action="store_true")
parser.add_argument("primary", help="numbers to calculate properties on (default read from stdin)", default=[], nargs='*')
args = parser.parse_args()

quiet = args.quiet
if args.precision is not None:
    formatter = f'{{:.{args.precision}f}}'

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
    l1 = []
    for x in args.primary:
        if os.path.exists(x):
            l1.extend(numbersFromFile(open(x, 'r')))
        else:
            l1.append(x)
    l1 = numpy.array([float(x) for x in l1 if isFloat(x)])

if len(l1) == 0:
    raise Exception("Could not parse any numbers")

if args.debug:
    print(f'[DEBUG][L1] {l1}')

if len(args.secondary) > 0:
    l2 = []
    for x in args.secondary:
        if os.path.exists(x):
            l2.extend(numbersFromFile(open(x, 'r')))
        else:
            l2.append(x)
    l2 = numpy.array([float(x) for x in l2 if isFloat(x)])
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
            try:
                if props[cp]['argument']:
                    pattern = re.compile(re.escape(cp).replace('%', '(.+)', 1))  # [-+]?\d*\.\d+|\d+)', 1))
                    reres = pattern.search(p)

                    if reres and len(reres.groups()) == 1:
                        arg = float(reres.group(1))
                        prop = cp
                        break
            except Exception:
                pass

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
        tmp = str(arg)
        if (int(arg) == arg):
            tmp = str(int(arg))
        labels.append(props[prop]['label'].replace('%', str(tmp), 1))
    else:
        labels.append(props[prop]['label'])

    results.append(props[prop]['func'](l1, l2, arg))
    if args.debug:
        print(f'[DEBUG][{p}] {results[-1]}')

results = [formatter.format(x) if isFloat(x) else x for x in results]

if args.quiet:
    print(' '.join(results))
else:
    lengths = [max(len(str(x)), len(str(y))) for x, y in zip(labels, results)]
    rowFormat = ' '.join([f"{{:>{length}}}" for length in lengths])
    print(rowFormat.format(*labels))
    print(rowFormat.format(*results))
