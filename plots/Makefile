all: gantt.png gantt_time.png line.png scatter.png bar.png box.png violin.png subplots.png

clean:
	rm -f gantt.png gantt_time.png line.png scatter.png bar.png box.png violin.png subplots.png

gantt.png:
	../plot.py -i gantt.tsv --plot bar --bar-mode stack --bar-text-position inside --x-type date --orientation h --output $@

gantt_time.png:
	../plot.py -i gantt_time.tsv --plot bar --orientation h --x-type date --bar-text-position outside --bar-mode stack --output $@

line.png:
	../plot.py -i lines.tsv --plot line --output $@

scatter.png:
	../plot.py -i scatter.tsv --plot line --line-mode markers --output $@

bar.png:
	../plot.py -i bar.tsv --plot bar --output $@

box.png:
	../plot.py -i distribution.tsv --plot box --output $@

violin.png:
	../plot.py -i distribution.tsv --plot violin --output $@

subplots.png:
	../plot.py -i lines.tsv --colspan 2 -i scatter.tsv --row 2 --colspan 1 --line-mode markers -i bar.tsv --col 2 --plot bar -i distribution.tsv --row 3 --col 1 --plot violin -i distribution.tsv --col 2 --plot box --horizontal-spacing 0.05 --vertical-spacing 0.05 --per-trace-colours --output $@
