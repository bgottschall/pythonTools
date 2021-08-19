# numprops.py

Calculates number properties of the number set passed via paramters or from stdin. Filters any numbers from the given input and can combine number sets given via paramter and stdin. Adjustable precision and POSIX friendly output.

Currently supported number properties:
* count
* sum, min, max
* q1, q2, q3, p% (percentile e.g. p99)
* avg, std, var
* pvalue, spearmanr, pearsonr (require second number set)

```
usage: numprops [-h] [--stdin] [--precision PRECISION]
                [-p PROPERTIES [PROPERTIES ...]] [-q]
                [--secondary [SECONDARY [SECONDARY ...]]] [--debug]
                [primary [primary ...]]

output number properties from numbers passed or read from stdin

positional arguments:
  primary               numbers to calculate properties on (default read from
                        stdin)

optional arguments:
  -h, --help            show this help message and exit
  --stdin               read from stdin even if numbers are provided
  --precision PRECISION
                        force a specific precision
  -p PROPERTIES [PROPERTIES ...], --properties PROPERTIES [PROPERTIES ...]
                        format output (default count, sum, min, max, q2, avg,
                        std) (valid count, sum, min, max, q1, q2, q3, p%, avg,
                        std, var, pvalue, spearmanr, pearsonr)
  -q, --quiet           minimal output
  --secondary [SECONDARY [SECONDARY ...]]
                        secondary number set used e.g. to calculate p-value
                        statistics
  --debug               turn on debug output
```

### Requirements

PIP3 packages:

*  numpy
*  scipy

### Examples

```
> numprops.py 1 2 3 4 5
Count  Sum Min Max  Q2 Avg                  σ
    5 15.0 1.0 5.0 3.0 3.0 1.4142135623730951
    
> numprops.py --properties pvalue --secondary 1 2 3 -- 3 4 5
            P-Value
0.07048399691021993

> cat /proc/cpuinfo | grep MHz | numprops.py
Count                Sum      Min      Max                 Q2        Avg                  σ
   12 45696.465000000004 3781.705 3912.226 3787.1270000000004 3808.03875 43.673346303599956

> cat /proc/cpuinfo | grep MHz | numprops.py --precision 2 -p min max avg p99
    Min     Max     Avg     P99
2444.84 3604.36 3173.42 3590.52
```
