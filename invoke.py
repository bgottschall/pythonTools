#!/usr/bin/env python3

from argparse import ArgumentParser
import os
import sys
import subprocess
import copy
import json
from datetime import datetime
from pathlib import Path

invokePyVersion = '0.1'

invokeSpec = {}
currentConfigPath = os.path.curdir

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
    if isinstance(source['stdout'], str):
        benchSpec['stdout'] = source['stdout'] if os.path.isabs(source['stdout']) else os.path.abspath(benchSpec['dir'] + '/' + source['stdout'])
    if isinstance(source['stdout'], str):
        benchSpec['stderr'] = source['stderr'] if os.path.isabs(source['stderr']) else os.path.abspath(benchSpec['dir'] + '/' + source['stderr'])
    if isinstance(source['environment'], dict):
        benchSpec['environment'] = source['environment'] if benchSpec['environment'] is None else {**benchSpec['environment'], **source['environment']}
    if isinstance(source['disabled'], bool):
        benchSpec['disabled'] = source['disabled']

    for k in ['params', 'precmd', 'postcmd']:
        if isinstance(source[k], str):
            benchSpec[k] = source[k]
        elif source[k] == False:
            benchSpec[k] = None

    return benchSpec


def batchReplace(target, what: dict, wrapper = '%'):
    if isinstance(target, dict):
        for x in target:
            for k, v in what.items():
                target[x] = str(target[x]).replace(wrapper + k + wrapper, v)
    elif isinstance(target, str):
        for k, v in what.items():
            target = target.replace(wrapper + k + wrapper, v);
    return target


def ensureDictionaryKeys(target: dict, keys: list, default=None):
    for k in keys:
        if k not in target:
            target[k] = default


def loadSpecification(spec: dict):
    global invokeSpec
    global currentConfigPath
    if not isinstance(invokeSpec, dict):
        invokeSpec = {}

    # Patching the path of the config file into the specification
    # Resolves relative paths inside the speficiation relative to
    # the config file.
    if isinstance(spec['specs'], list):
        for s in spec['specs']:
            if 'dir' in s and not os.path.isabs(s['dir']):
                s['dir'] = currentConfigPath + '/' + s['dir']
            elif 'dir' not in s:
                s['dir'] = currentConfigPath
    # merge
    for k in spec:
        if isinstance(spec[k], dict):
            if k in invokeSpec and isinstance(invokeSpec[k], dict):
                invokeSpec[k] = {**invokeSpec[k], **spec[k]}
            else:
                invokeSpec[k] = spec[k]
        elif isinstance(spec[k], list):
            if k in invokeSpec and isinstance(invokeSpec[k], list):
                invokeSpec[k].extend(spec[k])
            else:
                invokeSpec[k] = spec[k]
        else:
            invokeSpec[k] = spec[k]

parser = ArgumentParser(description="Invoke and control benchmarks")
parser.add_argument("-c", "--config", help="use this invoke specification", action="append", default=[])
parser.add_argument("-w", "--wrapper", help="Use these invoke wrappers", action="append", default=[])
parser.add_argument("-e", "--environment", help="Use these invoke environments", action="append", default=[])
parser.add_argument("-s", "--suite", help="Invoke these benchmark suites", action="append", default=[])
parser.add_argument("-i", "--input", help="Use these input sets", action="append", default=[])
parser.add_argument("-v", "--variable", help="define varibles e.g. -v var=test", action="append", default=[])
parser.add_argument("-f", "--force", help="ignore benchmark invocations that could not be resolved", action="store_true", default=False)
parser.add_argument("--stdout", help="redirect stdout from benchmark invocation", default=False)
parser.add_argument("--stderr", help="reiderct stderr from benchmark invocation", default=False)
parser.add_argument("--precmd", help="execute this command before each benchmark invocation", default=False)
parser.add_argument("--postcmd", help="execute this command after each benchmark invocation", default=False)
parser.add_argument("--specs", help="Show available benchmarks, suites, wrappers, environments and variables", action="store_true", default=False)
parser.add_argument("--list-benchmarks", help="show a list of specified benchmarks", action="store_true", default=False)
parser.add_argument("--list-suites", help="show a list of specified suites", action="store_true", default=False)

parser.add_argument("--compile", help="compile shell script", action="store_true", default=False)
parser.add_argument("--simulate", help="simulate invocation with verbose output", action="store_true", default=False)
parser.add_argument("--prepare", help="create directories and links", action="store_true", default=False)
parser.add_argument("--verbose", help="verbose output", action="store_true", default=False)
parser.add_argument("--version", help="print version number", action="store_true", default=False)

parser.add_argument("benchmarks", help="Invoke these benchmarks", nargs="*", default=[])

args = parser.parse_args();

if args.version:
    print(f'invoke.py version {invokePyVersion} -- sourced at https://github.com/bgottschall/pythonTools/blob/master/invoke.py')
    exit(0)

if args.simulate:
    print('Simulating, no invocations or changes to the filesystems will be made!')
    args.compile = False
    args.prepare = False

if args.compile:
    args.verbose = False
    args.simulate = False


if len(args.config) == 0:
    if os.path.exists(os.path.curdir + '/' + 'invoke.spec.json'):
        args.config = [os.path.curdir + '/' + 'invoke.spec.json'];
    elif os.path.exists(os.path.dirname(__file__) + '/' + 'invoke.spec.json'):
        args.config = [os.path.dirname(__file__) + '/' + 'invoke.spec.json'];
    else:
        raise Exception(f'Could not find any invoke specification files!')

for config in args.config:
    if not os.path.exists(config):
        if args.force:
            if args.verbose:
                print(f"WARNING: could not find invoke specification {config}", file=sys.stderr)
            continue
        raise Exception(f'Invoke specification {args.config} not found!')
    currentConfigPath = os.path.dirname(config)
    try:
        loadSpecification(json.load(open(config)))
    except Exception:
        print(f"Could not parse configuration file {config}", file=sys.stderr)
        raise



if not isinstance(invokeSpec, dict):
    raise Exception('Invalid invoke specification')

benchmarksAvailable = False

#Sanitize Config to make parsing easier
ensureDictionaryKeys(invokeSpec, ['specs', 'variables', 'wrappers', 'environments', 'suites'])
if invokeSpec['specs'] is not None:
    for spec in invokeSpec['specs']:
        ensureDictionaryKeys(spec, ['dir', 'precmd', 'postcmd', 'environment', 'stdout', 'stderr', 'input' , 'benchmarks'])
        if spec['benchmarks'] is not None:
            benchmarksAvailable = benchmarksAvailable or len(spec['benchmarks']) > 0
            for b in spec['benchmarks']:
                ensureDictionaryKeys(spec['benchmarks'][b], ['disabled', 'dir', 'exec', 'params', 'precmd', 'postcmd', 'stdout', 'stderr', 'environment', 'input'])
                if spec['benchmarks'][b]['inputs'] is not None:
                    for i in spec['benchmarks'][b]['inputs']:
                        ensureDictionaryKeys(spec['benchmarks'][b]['inputs'][i], ['disabled', 'dir', 'exec', 'params', 'precmd', 'postcmd', 'stdout', 'stderr', 'environment', 'workloads'])
                        if spec['benchmarks'][b]['inputs'][i]['workloads'] is not None:
                            for w in spec['benchmarks'][b]['inputs'][i]['workloads']:
                                ensureDictionaryKeys(w, ['disabled', 'dir', 'exec', 'params', 'precmd', 'postcmd', 'stdout', 'stderr', 'environment'])

if invokeSpec['variables'] is None:
    invokeSpec['variables'] = {}

if len(args.variable) > 0:
    for v in args.variable:
        vsplit = v.split('=')
        invokeSpec['variables'][vsplit[0]] = '='.join(vsplit[1:])

if args.list_suites:
    if invokeSpec['suites'] is None or len(invokeSpec['suites']) == 0:
        print('No suites specified!')
    else:
        for s in invokeSpec['suites']:
            print(s)
    exit(0)

if args.list_benchmarks:
    if not benchmarksAvailable or invokeSpec['specs'] is None:
        print('No benchmarks specified')
    else:
        if len(args.suite) > 0:
            for s in args.suite:
                if s not in invokeSpec['suites']:
                    raise Exception(f"Suite '{s}' not found")
                for b in invokeSpec['suites'][s]['benchmarks']:
                    for spec in invokeSpec['specs']:
                        if b in spec['benchmarks'] and (not spec['benchmarks'][b]['disabled'] or args.force):
                            print(b)
        else:
            for spec in invokeSpec['specs']:
                for b in spec['benchmarks']:
                    if not spec['benchmarks'][b]['disabled'] or args.force:
                        print(b)
    exit(0)


if args.specs:
    print(f"{'Benchmark':24s} {'Inputs'}")
    print('---')
    if not benchmarksAvailable or invokeSpec['specs'] is None:
        print('No benchmarks specified')
    else:
        for spec in invokeSpec['specs']:
            for b in spec['benchmarks']:
                print(f"{b:24s} {', '.join(spec['benchmarks'][b]['inputs'].keys())}{' (disabled)' if spec['benchmarks'][b]['disabled'] else ''}")
    print('')
    print(f"{'Suite':24s} {'Benchmarks'}")
    print('---')
    if invokeSpec['suites'] is None or len(invokeSpec['suites']) == 0:
        print('No suites specified!')
    else:
        for s in invokeSpec['suites']:
            print(f"{s:24s} {', '.join(invokeSpec['suites'][s]['benchmarks'])}")
    print('')
    print(f"{'Wrapper':24s} {'Definition'}")
    print('---')
    if invokeSpec['wrappers'] is None or len(invokeSpec['wrappers']) == 0:
        print('No wrappers specified')
    else:
        for w in invokeSpec['wrappers']:
            print(f"{w:24s} {invokeSpec['wrappers'][w]}")
    print('')
    print(f"{'Variable':24s} {'Definition'}")
    print('---')
    if invokeSpec['variables'] is None or len(invokeSpec['variables']) == 0:
        print('No variables specified')
    else:
        for v in invokeSpec['variables']:
            print(f"{v:24s} {invokeSpec['variables'][v]}")
    print('')

    print(f"{'Environment':24s} {'Definition'}")
    print('---')
    if invokeSpec['environments'] is None or len(invokeSpec['environments']) == 0:
        print('No environments specified')
    else:
        for e in invokeSpec['environments']:
            print(f"{e:24s} {invokeSpec['environments'][e]}")
    exit(0)


if not benchmarksAvailable:
    raise Exception('No benchmarks are specified in the configuration!')
   
for suite in args.suite:
    if suite not in invokeSpec['suites']:
        raise Exception(f"Suite '{suite}' not found")
    if isinstance(invokeSpec['suites'][suite]['benchmarks'], list):
        args.benchmarks.extend(invokeSpec['suites'][suite]['benchmarks'])
    else:
        args.benchmarks.append(invokeSpec['suites'][suite]['benchmarks'])

# create a unique benchmark selection, no need to invoke a benchmark more than once (no logical for this script)
uniqueBenchmarks = set()
args.benchmarks = [b for b in args.benchmarks if b not in uniqueBenchmarks and not uniqueBenchmarks.add(b)]
del uniqueBenchmarks

if len(args.benchmarks) == 0:
    print('No benchmarks selected for invocation!')
    exit(1)


# Make sure its all string
for v in invokeSpec['variables']:
    invokeSpec['variables'][v] = str(invokeSpec['variables'][v])
           

# Lets build the default Invoke CMD line
defaultInvoke = ''

if len(args.wrapper) > 0:
    for w in args.wrapper:
        if invokeSpec['wrappers'] is None or w not in invokeSpec['wrappers']:
            raise Exception(f"Wrapper '{w}' not found!")
        defaultInvoke += invokeSpec['wrappers'][w]
        if not defaultInvoke.endswith(' '):
            defaultInvoke += ' '


# Start constructing the environment
globalEnvironment = {}
globalVarEnvironment = {}

if len(args.environment) > 0:
    for e in args.environment:
        if invokeSpec['environments'] is None or e not in invokeSpec['environments']:
            raise Exception(f"Environment '{e}' not found!")
        for k, v in invokeSpec['environments'][e].items():
            v = str(v)
            if v.count('%') >= 2:
                globalVarEnvironment[k] = v
            else:
                globalEnvironment[k] = v

globalVarEnvironment = batchReplace(globalVarEnvironment, invokeSpec['variables'])
tempDict = {}
for k, v in globalVarEnvironment.items():
    if v.count('%') < 2:
        globalEnvironment[k] = v
    else:
        tempDict[k] = v
globalVarEnvironment = tempDict


if args.compile:
    shellScript = '#!/bin/sh\n\n'

if args.compile and len(globalEnvironment) > 0:
    shellScript +='export '
    for k in globalEnvironment:
        shellScript += f'{k}={globalEnvironment[k]} '
    shellScript += '\n'

if args.simulate and len(globalEnvironment) > 0:
    print('/:$ export ', end='')
    for k in glonalEnvironment:
        print(f'{k}={globalEnvironment[k]} ', end='')
    print('')

for d in ['outdir', 'output']:
    if d in invokeSpec['variables']:
        if os.path.isabs(invokeSpec['variables'][d]) and not os.path.exists(invokeSpec['variables'][d]):
            if args.prepare or (not args.compile and not args.simulate):
                try:
                    Path(invokeSpec['variables'][d]).mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
            elif args.compile:
                shellScript += f"mkdir -p {invokeSpec['variables'][d]} 2>/dev/null || true\n"
            if args.simulate:
                print(f"/:$ mkdir -p {invokeSpec['variables'][d]} || true")

if args.compile:
    shellScript += '\n'

invokeCounter = 0
failedInvokes = []
outputFiles = []

for benchmark in args.benchmarks:
    benchmarkFound = False
    for specIndex, spec in enumerate(invokeSpec['specs']):
        if benchmark not in spec['benchmarks']:
            continue

        benchmarkFound = True
        subEnvironment = {}

        benchSpecL0 = {
            'dir' : spec['dir'] if spec['dir'] is not None else os.path.curdir,
            'exec' : None,
            'params' : None,
            'precmd' : spec['precmd'],
            'postcmd' : spec['postcmd'],
            'stdout' : spec['stdout'],
            'stderr' : spec['stderr'],
            'environment': {},
            'disabled': False,
        }

        if isinstance(spec['environment'], dict):
            for k, v in spec['environment'].items():
                v = str(v)
                if v.count('%') >= 2:
                    v = batchReplace(v, invokeSpec['variables'])
                if v.count('%') >= 2:
                    benchSpec['environment'][k] = v
                else:
                    subEnvironment[k] = v

            if args.compile and len(subEnvironment) > 0:
                shellScript +='export '
                for k in subEnvironment:
                    shellScript += f'{k}={subEnvironment[k]} '
                shellScript += '\n'

            if args.simulate and len(subEnvironment) > 0:
                print('/:$ export ', end='')
                for k in subEnvironment:
                    print(f'{k}={subEnvironment[k]} ', end='')
                print('')


        if len(args.input) == 0:
            if spec['input'] is None:
                if args.force:
                    if args.verbose:
                        print(f'WARNING: specification {specIndex} odes not provide a default input, skipping.', file=sys.stderr)
                    continue
                raise Exception(f"Specficiation {specIndex} does not have a default input, please specify one!")
            useInputs = spec['input'] if isinstance(spec['input'], list) else [spec['input']]
        else:
            useInputs = args.input


        benchSpecL1 = updateBenchSpec(benchSpecL0, spec['benchmarks'][benchmark], benchmark)

        for input in useInputs:
            if input not in spec['benchmarks'][benchmark]['inputs']:
                print(f"WARNING: Could not find input '{input}' for benchmark '{benchmark}', will skip invocation...", file=sys.stderr)
                continue

            benchSpecL2 = updateBenchSpec(benchSpecL1, spec['benchmarks'][benchmark]['inputs'][input], input)

            if len(spec['benchmarks'][benchmark]['inputs'][input]['workloads']) == 0:
                print(f"WARNING: no workloads defined for benchmark '{benchmark}', will skip invocation...", file=sys.stderr)
                continue

            symlinked = False

            for workload, _ in enumerate(spec['benchmarks'][benchmark]['inputs'][input]['workloads']):
                benchSpec = updateBenchSpec(benchSpecL2, spec['benchmarks'][benchmark]['inputs'][input]['workloads'][workload])

                # the %now% variable is replaced by the datetime, if compiling its resolved through the shell

                if benchSpec['disabled']:
                    if args.force:
                        print(f"WARNING: ignore disabled flag for benchmark '{benchmark}'", file=sys.stderr)
                    else:
                        if args.verbose:
                            print(f"Ignore disabled benchmark '{benchmark}'")
                        continue

                if benchSpec['exec'] is None:
                    if args.force:
                        if args.verbose:
                            print(f"WARNING: ignored workload {workload} of '{benchmark}' because no executable was defined", file=sys.stderr)
                        continue
                    raise Exception(f"No executable defined for benchmark '{benchmark}'")

                if benchSpec['params'] is None:
                    benchSpec['params'] = ''

                if args.precmd:
                    benchSpec['precmd'] = args.precmd
                if args.postcmd:
                    benchSpec['postcmd'] = args.postcmd

                if args.stdout:
                    if os.path.isabs(args.stdout):
                        benchSpec['stdout'] = os.path.abspath(args.stdout)
                    else:
                        benchSpec['stdout'] = os.path.abspath(os.path.curdir + '/' + args.stdout)
                if args.stderr:
                    if os.path.isabs(args.stderr):
                        benchSpec['stderr'] = os.path.abspath(args.stderr)
                    else:
                        benchSpec['stderr'] = os.path.abspath(os.path.curdir + '/' + args.stderr)

                if args.compile and ('%now%' in str(benchSpec)) or ('%now%' in defaultInvoke):
                    sDate = '${NOW}'
                else:
                    sDate = datetime.now().strftime("%Y-%m-%d_%H%M%S")

                replaceVars = {**{'counter' : str(invokeCounter), 'workload': str(workload), 'input' : str(input), 'benchmark': str(benchmark), 'now': sDate}, **invokeSpec['variables']}

                benchSpec['dir'] = batchReplace(benchSpec['dir'], replaceVars)
                benchSpec['exec'] = batchReplace(benchSpec['exec'], replaceVars)

                execName = os.path.basename(benchSpec['exec'])


                benchSpec['exec'] = os.path.abspath(benchSpec['exec'])
                if not os.path.isdir(benchSpec['dir']):
                    if args.verbose:
                        print(f"Creating directory '{benchSpec['dir']}'")
                    if args.simulate:
                        print(f"/:$ mkdir -p {benchSpec['dir']}")
                    else:
                        Path(benchSpec['dir']).mkdir(parents=True, exist_ok=True)


                # Executable is not where it is supposed to be, but one is available in the input directory
                if not os.path.exists(benchSpec['exec']) and os.path.exists(benchSpec['dir'] + '/' + execName):
                    if args.verbose:
                        print(f"WARNING: couldn't find {benchSpec['exec']}, will use {benchSpec['dir'] + '/' + execName} instead", file=sys.stderr)
                    benchSpec['exec'] = benchSpec['dir'] + '/' + execName

                if not os.path.exists(benchSpec['exec']):
                    if args.force:
                        if args.verbose:
                            print(f"WARNING: ignored workload {workload} of '{benchmark}' because executable '{benchSpec['exec']}' was not found", file=sys.stderr)
                        continue
                    raise Exception(f"Could not find executable '{benchSpec['exec']}'")



                benchSpec['dir'] = os.path.abspath(benchSpec['dir'])


                if args.compile:
                    shellScript += f"# Execute workload {workload} of the '{input}' input of benchmark '{benchmark}'\n"
                    if sDate == '${NOW}':
                        shellScript +='NOW="$(date +\'%Y-%m-%d_%H%M%S\')"\n'

                if args.verbose and not args.prepare:
                    print(f"Executing benchmark '{benchmark}', input '{input}', workload {workload}")

                if not os.path.exists(benchSpec['dir'] + '/' + execName):
                    if args.verbose:
                        print(f"Symlinking '{benchSpec['exec']}' to '{benchSpec['dir'] + '/' + execName}'")
                    if args.simulate:
                        print(f"/:$ ln -s {benchSpec['exec']} {benchSpec['dir'] + '/' + execName}")
                    elif args.prepare or not args.compile:
                        os.symlink(benchSpec['exec'], benchSpec['dir'] + '/' + execName)
                    else:
                        shellScript += f"ln -s {benchSpec['exec']} {benchSpec['dir'] + '/' + execName}\n"
                    symlinked = True
                elif os.path.realpath(benchSpec['exec']) != os.path.realpath(benchSpec['dir'] + '/' + execName):
                    print(f"WARNING: target executable '{benchSpec['dir'] + '/' + execName}' differs from specified executable '{benchSpec['exec']}'", file=sys.stderr)


                invokeCmd = defaultInvoke + './' + execName
                if len(benchSpec['params']) > 0:
                    invokeCmd += ' ' + benchSpec['params']

                benchSpec['environment'] = {**benchSpec['environment'], **globalVarEnvironment}

                replaceVars = {**replaceVars, **{'dir': benchSpec['dir'], 'exec': execName}}

                # Postprocess the invoke cmd lines after variables
                invokeCmd = batchReplace(invokeCmd, replaceVars)
                if isinstance(benchSpec['precmd'], str):
                    benchSpec['precmd'] = batchReplace(benchSpec['precmd'], replaceVars)
                if isinstance(benchSpec['postcmd'], str):
                    benchSpec['postcmd'] = batchReplace(benchSpec['postcmd'], replaceVars)

                stdoutDate = False
                if isinstance(benchSpec['stdout'], str):
                    stdoutDate = '%now%' in os.path.dirname(benchSpec['stdout'])
                    benchSpec['stdout'] = batchReplace(benchSpec['stdout'], replaceVars)

                stderrDate = False
                if isinstance(benchSpec['stderr'], str):
                    stderrDate = '%now%' in os.path.dirname(benchSpec['stderr'])
                    benchSpec['stderr'] = batchReplace(benchSpec['stderr'], replaceVars)

                benchSpec['environment'] = batchReplace(benchSpec['environment'], replaceVars)

                if not args.compile:
                    invokeEnvironment = {**os.environ.copy(), **subEnvironment, **benchSpec['environment'], **globalEnvironment}

                if benchSpec['stdout'] is not None:
                    if not os.path.exists(os.path.dirname(benchSpec['stdout'])) and benchSpec['stdout'] not in outputFiles:
                        if args.prepare or (not args.compile and not args.simulate):
                            if not stdoutDate:
                                Path(os.path.dirname(benchSpec['stdout'])).mkdir(parents=True, exist_ok=True)
                        if (args.compile and not args.prepare) or '${NOW}' in benchSpec['stdout']:
                            shellScript += f"mkdir -p {os.path.dirname(benchSpec['stdout'])}\n"
                        if args.simulate:
                            print(f"/:$ mkdir -p {os.path.dirname(benchSpec['stdout'])}")

                    if benchSpec['stdout'] not in outputFiles:
                        outputFiles.append(benchSpec['stdout'])
                        # if args.compile or args.simulate:
                        #     if args.compile:
                        #         shellScript += f"> {benchSpec['stdout']}\n"
                        #     else:
                        #         print(f"/:$ > {benchSpec['stdout']}")
                        # else:
                        #     with open(benchSpec['stdout'], 'w') as _: pass

                    if args.compile or args.simulate:
                        if args.stdout and os.path.isabs(args.stdout):
                            invokeCmd += f" >>{benchSpec['stdout']}"
                        else:
                            invokeCmd += f" >>{os.path.relpath(benchSpec['stdout'], benchSpec['dir'])}"
                    else:
                        benchSpec['stdout'] = open(benchSpec['stdout'], 'a')


                if benchSpec['stderr'] is not None:
                    if not os.path.exists(os.path.dirname(benchSpec['stderr'])) and not benchSpec['stderr'] in outputFiles:
                        if args.prepare or (not args.compile and not args.simulate):
                            if not stderrDate:
                                Path(os.path.dirname(benchSpec['stderr'])).mkdir(parents=True, exist_ok=True)
                        if (args.compile and not args.prepare) or '${NOW}' in benchSpec['stderr']:
                            shellScript += f"mkdir -p {os.path.dirname(benchSpec['stderr'])}\n"
                        if args.simulate:
                            print(f"/:$ mkdir -p {os.path.dirname(benchSpec['stderr'])}")

                    if benchSpec['stderr'] not in outputFiles:
                        outputFiles.append(benchSpec['stderr'])
                        # if args.compile or args.simulate:
                        #     if args.compile:
                        #         shellScript += f"> {benchSpec['stderr']}\n"
                        #     else:
                        #         print(f"/:$ > {benchSpec['stderr']}")
                        # else:
                        #     with open(benchSpec['stderr'], 'w') as _: pass

                    if args.compile or args.simulate:
                        if args.stderr and os.path.isabs(args.stderr):
                            invokeCmd += f" 2>>{benchSpec['stderr']}"
                        else:
                            invokeCmd += f" 2>>{os.path.relpath(benchSpec['stderr'], benchSpec['dir'])}"
                    else:
                        benchSpec['stderr'] = open(benchSpec['stderr'], 'a')

                if args.compile:
                    shellScript += '(\n'
                    shellScript += '  set -x\n'
                    shellScript += f"  cd \"{benchSpec['dir']}\"\n"
                    if len(benchSpec['environment']) > 0:
                        shellScript += '  export '
                        for k in benchSpec['environment']:
                            shellScript += f"{k}={benchSpec['environment'][k]} "
                        shellScript += '\n'
                if args.simulate:
                    if len(benchSpec['environment']) > 0:
                        print(f"{benchSpec['dir']}:$ export ", end='')
                        for k in benchSpec['environment']:
                            print(f"{k}={benchSpec['environment'][k]} ", end='')
                        print('')
                if args.verbose and len(benchSpec['environment']) > 0:
                    print(f"Setting environment to {benchSpec['environment']}")

                if isinstance(benchSpec['precmd'], str):
                    if not args.prepare:
                        if args.verbose:
                            print(f"Executing pre invoke command '{benchSpec['precmd']}'")
                        if args.simulate:
                            print(f"{benchSpec['dir']}:$ {benchSpec['precmd']}")
                        elif not args.prepare and not args.compile:
                            ret = subprocess.call(benchSpec['precmd'], shell=True, cwd=benchSpec['dir'], env=invokeEnvironment)
                            if ret != 0:
                                if args.verbose:
                                    print(f"Execution failed with return code {ret}")
                                failedInvokes.append(f"{benchmark}-{input}-{workload}-precmd")
                    if args.compile:
                        shellScript += f"  {benchSpec['precmd']}\n"

                if not args.prepare and args.verbose:
                    print(f"Invoke command line '{invokeCmd}'")
                    if not args.compile and not args.simulate:
                        if benchSpec['stdout'] is not None:
                            print(f"Redirect stdout to {benchSpec['stdout'].name}")
                        if benchSpec['stderr'] is not None:
                            print(f"Redirect stderr to {benchSpec['stderr'].name}")
                if args.simulate:
                    print(f"{benchSpec['dir']}:$ {invokeCmd}")
                elif not args.prepare and not args.compile:
                    ret = subprocess.call(invokeCmd, shell=True, cwd=benchSpec['dir'], env=invokeEnvironment, stdout=benchSpec['stdout'], stderr=benchSpec['stderr'])
                    if ret != 0:
                        if args.verbose:
                            print(f"Execution failed with return code {ret}")
                        failedInvokes.append(f"{benchmark}-{input}-{workload}")
                if args.compile:
                    shellScript += f"  {invokeCmd}\n"

                if isinstance(benchSpec['postcmd'], str):
                    if not args.prepare:
                        if args.verbose:
                            print(f"Executing post invoke command '{benchSpec['postcmd']}'")
                        if args.simulate:
                            print(f"{benchSpec['dir']}:$ {benchSpec['postcmd']}")
                        elif not args.compile:
                            ret = subprocess.call(benchSpec['postcmd'], shell=True, cwd=benchSpec['dir'], env=invokeEnvironment)
                            if ret != 0:
                                if args.verbose:
                                    print(f"Execution failed with return code {ret}")
                                failedInvokes.append(f"{benchmark}-{input}-{workload}-postcmd")
                    if args.compile:
                        shellScript += f"  {benchSpec['postcmd']}\n"

                invokeCounter += 1

                if args.compile:
                    shellScript += ')\n'

                if symlinked and not args.prepare:
                    if args.verbose:
                        print("Remove previously created symlink")
                    if args.simulate:
                        print(f"/:$ rm {benchSpec['dir'] + '/' + execName}")
                    elif not args.compile:
                        os.unlink(benchSpec['dir'] + '/' + execName)
                    if args.compile:
                        shellScript += f"rm {benchSpec['dir'] + '/' + execName}\n"

                if args.compile:
                    shellScript += '\n'

    if not benchmarkFound:
        raise Exception(f"Could not find specification for benchmark '{benchmark}'!")

if args.compile:
    print(shellScript, end='')

if len(failedInvokes) != 0:
    print(f'WARNING: detected {failedInvokes} with an error return code!', file=sys.stderr)
