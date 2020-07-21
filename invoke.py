#!/usr/bin/env python3

from argparse import ArgumentParser
import os
import sys
import subprocess
import copy
from datetime import datetime

invokeSpec = {}


def ensureDictionaryKeys(target: dict, keys: list, default=None):
    for k in keys:
        if k not in target:
            target[k] = default


parser = ArgumentParser(description="Invoke and control benchmarks")
parser.add_argument("-c", "--config", help="use this invoke specification", action="extend", default=[])
parser.add_argument("-w", "--wrapper", help="Use these invoke wrappers", action="extend", default=[])
parser.add_argument("-e", "--environment", help="Use these invoke environments", action="extend", default=[])
parser.add_argument("-s", "--suite", help="Invoke these benchmark suites", action="extend", default=[])
parser.add_argument("-i", "--input", help="Use these input sets", default=[])
parser.add_argument("-v", "--variable", help="define varibles e.g. -v var=test", nargs='+', action="extend", default=[])
parser.add_argument("-f", "--force", help="ignore benchmark invocations that could not be resolved", action="store_true", default=False)
parser.add_argument("--pre-cmd", help="invoke this command before each benchmark invocation", default=False)
parser.add_argument("--post-cmd", help="invoke this command after each benchmark invocation", default=False)
parser.add_argument("-l", "--list", help="Show available benchmarks, suites, wrappers, environments and variables", action="store_true", default=False)

parser.add_argument("--compile", help="compile shell script", action="store_true", default=False)
parser.add_argument("--simulate", help="simulate invocation with verbose output", action="store_true", default=False)
parser.add_argument("--verbose", help="verbose output", action="store_true", default=False)

parser.add_argument("benchmarks", help="Invoke these benchmarks", nargs="*", default=[])

args = parser.parse_args();

if args.simulate:
    print('Simulating, no invocations or changes to the filesystems will be made!')
    args.verbose = True
    args.compile = False

if args.compile:
    args.verbose = False
    args.simulate = True


if len(args.config) == 0:
    if os.path.exists(os.path.curdir + '/' + 'invoke.spec.py'):
        args.config = [os.path.curdir + '/' + 'invoke.spec.py'];
    elif os.path.exists(os.path.dirname(__file__) + '/' + 'invoke.spec.py'):
        args.config = [os.path.dirname(__file__) + '/' + 'invoke.spec.py'];
    else:
        raise Exception(f'Could not find any invoke specification files!')

for config in args.config:
    if not os.path.exists(config):
        if args.force:
            if args.verbose:
                print(f"WARNING: could not find invoke specification {config}")
            continue
        raise Exception(f'Invoke specification {args.config} not found!')
    oldInvokeSpec = invokeSpec
    exec(open(config).read())

    if not isinstance(invokeSpec, dict):
        if args.force:
            if args.verbose:
                print(f"WARNING: '{config}' contains an invalid specification")
            invokeSpec = oldInvokeSpec
            continue
        raise Exception(f"specification '{config}' is invalid")

    # merge
    for k in invokeSpec:
        if isinstance(invokeSpec[k], dict):
            if k in oldInvokeSpec:
                oldInvokeSpec[k] = {**oldInvokeSpec[k], **invokeSpec[k]}
            if k not in oldInvokeSpec:
                oldInvokeSpec[k] = invokeSpec[k]
        else:
            oldInvokeSpec[k] = invokeSpec[k]
    invokeSpec = oldInvokeSpec


if invokeSpec is None:
    raise Exception('Invalid invoke specification')

#Sanitize Config to make parsing easier
ensureDictionaryKeys(invokeSpec, ['dir', 'preCmd', 'postCmd', 'environment', 'defaultInput' , 'variables', 'wrappers', 'environments', 'benchmarks', 'suites'])
if invokeSpec['benchmarks'] is not None:
    for b in invokeSpec['benchmarks']:
        ensureDictionaryKeys(invokeSpec['benchmarks'][b], ['desc', 'dir', 'exec', 'params', 'preCmd', 'postCmd', 'inputs', 'environment'])
        if invokeSpec['benchmarks'][b]['inputs'] is not None:
            for i in invokeSpec['benchmarks'][b]['inputs']:
                ensureDictionaryKeys(invokeSpec['benchmarks'][b]['inputs'][i], ['dir', 'exec', 'params', 'preCmd', 'postCmd', 'workloads', 'environment'])
                if invokeSpec['benchmarks'][b]['inputs'][i]['workloads'] is not None:
                    for w in invokeSpec['benchmarks'][b]['inputs'][i]['workloads']:
                        ensureDictionaryKeys(w, ['dir', 'exec', 'params', 'preCmd', 'postCmd', 'environment'])
if invokeSpec['suites'] is not None:
    for s in invokeSpec['suites']:
        ensureDictionaryKeys(invokeSpec['suites'][s], ['desc', 'benchmarks'])

if invokeSpec['variables'] is None:
    invokeSpec['variables'] = {}

if len(args.variable) > 0:
    for v in args.variable:
        vsplit = v.split('=')
        invokeSpec['variables'][vsplit[0]] = '='.join(vsplit[1:])


if args.list:
    print(f"{'Benchmark':24s}  {'Inputs':24s}  {'Description'}")
    print('-------------------------------------------------------------')
    if invokeSpec['benchmarks'] is None or len(invokeSpec['benchmarks']) == 0:
        print('No benchmarks specified')
    else:
        for b in invokeSpec['benchmarks']:
            desc = '-' if invokeSpec['benchmarks'][b]['desc'] is None else invokeSpec['benchmarks'][b]['desc']
            print(f"{b:24s}  {', '.join(invokeSpec['benchmarks'][b]['inputs'].keys()):24s}  {desc}")
    print('')

    print(f"{'Suite':24s}  {'Description':30s}  {'Benchmarks'}")
    print('-------------------------------------------------------------')
    if invokeSpec['suites'] is None or len(invokeSpec['suites']) == 0:
        print('No suites specified!')
    else:
        for s in invokeSpec['suites']:
            desc = '-' if invokeSpec['suites'][s]['desc'] is None else invokeSpec['suites'][s]['desc']
            print(f"{s:24s}  {desc[:30]:30s}  {', '.join(invokeSpec['suites'][s]['benchmarks'])}")
    print('')
    print(f"{'Wrapper':24s} {'Definition'}")
    print('-------------------------------------------------------------')
    if invokeSpec['wrappers'] is None or len(invokeSpec['wrappers']) == 0:
        print('No wrappers specified')
    else:
        for w in invokeSpec['wrappers']:
            print(f"{w:24s} {invokeSpec['wrappers'][w]}")
    print('')
    print(f"{'Variable':24s} {'Definition'}")
    print('-------------------------------------------------------------')
    if invokeSpec['variables'] is None or len(invokeSpec['variables']) == 0:
        print('No variables specified')
    else:
        for v in invokeSpec['variables']:
            print(f"{v:24s} {invokeSpec['variables'][v]}")
    print('')

    print(f"{'Environment':24s} {'Definition'}")
    print('-------------------------------------------------------------')
    if invokeSpec['environments'] is None or len(invokeSpec['environments']) == 0:
        print('No environments specified')
    else:
        for e in invokeSpec['environments']:
            print(f"{e:24s} {invokeSpec['environments'][e]}")
    exit(0)


if invokeSpec['benchmarks'] is None or len(invokeSpec['benchmarks']) == 0:
    raise Exception('No benchmarks are specifiec in the configuration!')
   
if len(args.input) == 0:
    if invokeSpec['defaultInput'] is None:
        raise Exception('No input set given to execute, please specify --inputs')
    args.input = invokeSpec['defaultInput'] if isinstance(invokeSpec['defaultInput'], list) else [invokeSpec['defaultInput']]


for suite in args.suite:
    if suite not in invokeSpec['suites']:
        raise Exception(f"Suite '{suite}' not found")
    if isinstance(invokeSpec['suites'][suite]['benchmarks'], list):
        args.benchmarks.extend(invokeSpec['suites'][suite]['benchmarks'])
    else:
        args.benchmarks.append(invokeSpec['suites'][suite]['benchmarks'])

# create a unique benchmark selection, no need to invoke a benchmark more than once (no logical for this script)
args.benchmarks = list(set(args.benchmarks))

if len(args.benchmarks) == 0:
    print('No benchmarks selected for invocation!')
    exit(1)

# Compile will create a shell script, do not force the python environment on it
environment = {}
varEnvironment = {}

# Make sure its all string
for v in invokeSpec['variables']:
    invokeSpec['variables'][v] = str(invokeSpec['variables'][v])

if isinstance(invokeSpec['environment'], dict):
   for k, v in invokeSpec['environment'].items():
       v = str(v)
       if str(v).count('%') >= 2:
           varEnvironment[k] = v
       else:
           environment[k] = v

if len(args.environment) > 0:
    for e in args.environment:
        if invokeSpec['environments'] is None or e not in invokeSpec['environments']:
            raise Exception(f"Environment '{e}' not found!")
        for k, v in invokeSpec['environments'][e].items():
            v = str(v)
            if v.count('%') >= 2:
                varEnvironment[k] = v
            else:
                environment[k] = v

nVarEnvironment = {}

for k, v in varEnvironment.items():
    for var in invokeSpec['variables']:
        v = v.replace('%' + var + '%', invokeSpec['variables'][var])
    if v.count('%') < 2:
        environment[k] = v
    else:
        nVarEnvironment[k] = v

varEnvironment = nVarEnvironment

# Lets build the default Invoke CMD line
defaultInvoke = ''

if len(args.wrapper) > 0:
    for w in args.wrapper:
        if invokeSpec['wrappers'] is None or w not in invokeSpec['wrappers']:
            raise Exception(f"Wrapper '{w}' not found!")
        defaultInvoke += invokeSpec['wrappers'][w]
        if not defaultInvoke.endswith(' '):
            defaultInvoke += ' '

invokeCounter = 0


def updateBenchSpec(origBenchSpec: dict, source: dict, subdirectory = None):
    benchSpec = copy.deepcopy(origBenchSpec)
    if isinstance(source['dir'], str):
        # If a path is given use it
        benchSpec['dir'] = source['dir'] if os.path.isabs(source['dir']) else os.path.abspath(benchSpec['dir'] + '/' + source['dir'])
    elif source['dir'] is None and subdirectory is not None:
        # If not use the benchmark name as subdirectory
        benchSpec['dir'] = os.path.abspath(benchSpec['dir'] + '/' + subdirectory)
    if isinstance(source['exec'], str):
        benchSpec['exec'] = source['exec'] if os.path.isabs(source['exec']) else os.path.abspath(benchSpec['dir'] + '/' + source['exec'])
    if isinstance(source['environment'], dict):
        benchSpec['environment'] = source['environment'] if benchSpec['environment'] is None else {**benchSpec['environment'], **source['environment']}

    for k in ['params', 'preCmd', 'postCmd']:
        if isinstance(source[k], str):
            benchSpec[k] = source[k]
        elif source[k] == False:
            benchSpec[k] = None

    return benchSpec


def batchReplace(target: str, what: dict, wrapper = '%'):
    for k, v in what.items():
        target = target.replace(wrapper + k + wrapper, v);
    return target


if args.compile:
    print('#!/bin/sh\n')

if (args.compile or (args.simulate and args.verbose)) and len(environment) > 0:
    if args.compile:
        print('export ', end='')
    else:
        print('/:$ export ', end='')
    for k in environment:
        print(f'{k}={environment[k]} ', end='')
    print('')

if not args.compile:
    environment = {**os.environ.copy(), **environment}

failedInvokes = 0

configDir = os.path.dirname(os.path.abspath(args.config[-1]))


for benchmark in args.benchmarks:
    benchSpecL0 = {
        'dir' : invokeSpec['dir'] if isinstance(invokeSpec['dir'], str) and os.path.isabs(invokeSpec['dir']) else configDir + '/' + invokeSpec['dir'] if isinstance(invokeSpec['dir'], str) else configDir,
        'exec' : None,
        'params' : None,
        'preCmd' : invokeSpec['preCmd'],
        'postCmd' : invokeSpec['postCmd'],
        'environment': {}
    }

    if benchmark not in invokeSpec['benchmarks']:
        raise Exception(f"Could not find specification for benchmark '{benchmark}'!")

    benchSpecL1 = updateBenchSpec(benchSpecL0, invokeSpec['benchmarks'][benchmark], benchmark)

    for input in args.input:
        if input not in invokeSpec['benchmarks'][benchmark]['inputs']:
            print(f"WARNING: Could not find input '{input}' for benchmark '{benchmark}', will skip invocation...", file=sys.stderr)
            continue

        benchSpecL2 = updateBenchSpec(benchSpecL1, invokeSpec['benchmarks'][benchmark]['inputs'][input], input)
       
        if len(invokeSpec['benchmarks'][benchmark]['inputs'][input]['workloads']) == 0:
            print(f"WARNING: no workloads defined for benchmark '{benchmark}', will skip invocation...", file=sys.stderr)
            continue

        for workload, _ in enumerate(invokeSpec['benchmarks'][benchmark]['inputs'][input]['workloads']):
            benchSpec = updateBenchSpec(benchSpecL2, invokeSpec['benchmarks'][benchmark]['inputs'][input]['workloads'][workload])

            # the %now% variable is replaced by the datetime, if compiling its resolved through the shell
            sDate = '$(date +"%Y-%m-%d_%H%M%S")' if args.compile else datetime.now().strftime("%Y-%m-%d_%H%M%S")

            replaceVars = {**{'counter' : str(invokeCounter), 'workload': str(workload), 'input' : str(input), 'benchmark': str(benchmark), 'now': sDate}, **invokeSpec['variables']}

            if benchSpec['exec'] is None:
                if args.force:
                    if args.verbose:
                        print(f"WARNING: ignored workload {workload} of '{benchmark}' because no executable was defined")
                    continue
                raise Exception(f"No executable defined for benchmark '{benchmark}'")

            if benchSpec['params'] is None:
                benchSpec['params'] = ''

            if args.pre_cmd:
                benchSpec['preCmd'] = args.pre_cmd
            if args.post_cmd:
                benchSpec['postCmd'] = args.post_cmd

            benchSpec['dir'] = batchReplace(benchSpec['dir'], replaceVars)
            benchSpec['exec'] = batchReplace(benchSpec['exec'], replaceVars)

            if not os.path.exists(benchSpec['exec']):
                if args.force:
                    if args.verbose:
                        print(f"WARNING: ignored workload {workload} of '{benchmark}' because executable '{benchSpec['exec']}' was not found")
                    continue
                raise Exception(f"Could not find executable '{benchSpec['exec']}'")

            benchSpec['exec'] = os.path.realpath(benchSpec['exec'])
            execName = os.path.basename(benchSpec['exec'])
            if not os.path.isdir(benchSpec['dir']):
                if args.verbose:
                    print(f"Creating directory '{benchSpec['dir']}'")
                if args.compile or not args.simulate:
                    os.mkdir(benchSpec['dir'])

            benchSpec['dir'] = os.path.realpath(benchSpec['dir'])

            if not os.path.exists(benchSpec['dir'] + '/' + execName):
                if args.verbose:
                    print(f"Symlinking '{benchSpec['exec']}' to '{benchSpec['dir'] + '/' + execName}'")
                if args.compile or not args.simulate:
                    os.symlink(benchSpec['exec'], benchSpec['dir'] + '/' + execName)
            elif benchSpec['exec'] != os.path.realpath(benchSpec['dir'] + '/' + execName):
                print(f"WARNING: target executable '{benchSpec['dir'] + '/' + execName}' differs from specified executable '{benchSpec['exec']}'", file=sys.stderr)

            if args.compile:
                print(f"# Execute workload {workload} of the '{input}' input of benchmark '{benchmark}'")

            invokeCmd = defaultInvoke + './' + execName + ' ' + benchSpec['params']

            benchSpec['environment'] = {**benchSpec['environment'], **varEnvironment}

            # Postprocess the invoke cmd lines after variables
            invokeCmd = batchReplace(invokeCmd, replaceVars)
            if isinstance(benchSpec['preCmd'], str):
                benchSpec['preCmd'] = batchReplace(benchSpec['preCmd'])
            if isinstance(benchSpec['postCmd'], str):
                benchSpec['postCmd'] = batchReplace(benchSpec['postCmd'])
            for k in benchSpec['environment']:
                benchSpec['environment'][k] = batchReplace(benchSpec['environment'][k], replaceVars)

            if not args.compile:
                invokeEnvironment = {**environment, **benchSpec['environment']}

            if args.compile:
                print('(')
                print(f"  cd \"{benchSpec['dir']}\"")
                if len(benchSpec['environment']) > 0:
                    print('  export ', end='')
                    for k in benchSpec['environment']:
                        print(f"{k}={benchSpec['environment'][k]} ", end='')
                    print('')
            if isinstance(benchSpec['preCmd'], str):
                if args.verbose:
                    if len(benchSpec['environment']) > 0:
                        print(f"{benchSpec['dir']}:$ export ", end='')
                        for k in benchSpec['environment']:
                            print(f"{k}={benchSpec['environment'][k]} ", end='')
                        print('')
                    print(f"{benchSpec['dir']}:$ {benchSpec['preCmd']}")
                if args.compile:
                    print(f"  {benchSpec['preCmd']}")
                if not args.simulate:
                    if subprocess.call(benchSpec['preCmd'], shell=True, cwd=benchSpec['dir'], env=invokeEnvironment) != 0:
                        failedInvokes += 1


            if args.verbose:
                if len(benchSpec['environment']) > 0:
                    print(f"{benchSpec['dir']}:$ export ", end='')
                    for k in benchSpec['environment']:
                        print(f"{k}={benchSpec['environment'][k]} ", end='')
                    print('')
                print(f"{benchSpec['dir']}:$ {invokeCmd}")
            if args.compile:
                print(f"  {invokeCmd}")
            if not args.simulate:
                if subprocess.call(invokeCmd, shell=True, cwd=benchSpec['dir'], env=invokeEnvironment) != 0:
                    failedInvokes += 1

            if isinstance(benchSpec['postCmd'], str):
                if args.verbose:
                    if len(benchSpec['environment']) > 0:
                        print(f"{benchSpec['dir']}:$ export ", end='')
                        for k in benchSpec['environment']:
                            print(f"{k}={benchSpec['environment'][k]} ", end='')
                        print('')
                    print(f"{benchSpec['dir']}:$ {benchSpec['postCmd']}")
                if args.compile:
                    print(f"  {benchSpec['postCmd']}")
                if not args.simulate:
                    if subprocess.call(benchSpec['postCmd'], shell=True, cwd=benchSpec['dir'], env=invokeEnvironment) != 0:
                        failedInvokes += 1
            invokeCounter += 1
            if args.compile:
                print(')')

if failedInvokes != 0:
    print(f'WARNING: detected {failedInvokes} invocations with an error return code!', file=sys.stderr)
