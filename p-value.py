#!/usr/bin/env python

from sys import exit
from argparse import ArgumentParser
from scipy import stats

parser = ArgumentParser(description="calculate the p-value of given distributions")
parser.add_argument("--equal-variance", action="store_true", help="assume equal variance", default=False)
parser.add_argument("--a", type=float, help="distribution a", nargs="+")
parser.add_argument("--b", type=float, help="distribution a", nargs="+")
parser.add_argument("--verbose", action="store_true", help="show verbose output", default=False)
args = parser.parse_args()

if (not args.a or not args.b):
    print("ERROR: distributions a and b are required!")
    parser.print_help()
    exit(1)

if (len(args.a) <= 1 or len(args.b) <= 1):
    print("ERROR: to few values provided!")
    exit(1)

if args.verbose:
    print(f"Distribution A: {args.a}")
    print(f"Distribution B: {args.b}")

res = stats.ttest_ind(args.a, args.b, equal_var=(args.equal_variance))
if args.verbose:
    print(f"T-Statistics: {res[0]:.16f}")
    print(f"P-Value: {res[1]:.16f}")
else:
    print(f'{res[1]:.16f}')
