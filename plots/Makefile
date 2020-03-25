all: gantt.png gantt_time.png line.png scatter.png bar.png box.png violin.png subplots.png

gantt.png:
	../plot.py -i gantt.csv --plot bar --bar-mode stack --bar-text-position inside --x-type date --orientation h --output $@

gantt_time.png:
	../plot.py -i gantt_time.csv --plot bar --orientation h --x-type date --bar-text-position outside --bar-mode stack --output $@

line.png:
	../plot.py -i lines.csv --plot line --output $@

scatter.png:
	../plot.py -i scatter.csv --plot line --line-mode markers --output $@

bar.png:
	../plot.py -i bar.csv --plot bar --output $@

box.png:
	../plot.py -i distribution.csv --plot box --output $@

violin.png:
	../plot.py -i distribution.csv --plot violin --output $@

subplots.png:
	../plot.py -i lines.csv --colspan 2 -i scatter.csv --row 2 --colspan 1 --line-mode markers -i bar.csv --col 2 --plot bar -i distribution.csv --row 3 --col 1 --plot violin -i distribution.csv --col 2 --plot box --horizontal-spacing 0.05 --vertical-spacing 0.05 --per-trace-colours --output $@
