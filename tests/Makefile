


PLOTPY := ../plot.py

INPUT  := testdata.csv
OUTDIR := tests.out

SKIP_CHECK := false

.PHONY: tests general tests.dev tests.md5s $(OUTDIR)/$(INPUT)

define check_exists
	$(SKIP_CHECK) || [ -e "$(1)" ] && echo "> PASSED" || { echo "> FAILED"; exit 1; }
endef

define check_output
	$(SKIP_CHECK) || [ "$$(sed "$$(($(3) + 1))q;d" $(1).checksums 2>/dev/null)" = "$$(md5sum $(OUTDIR)/$(1).$(3).out)" ] && echo "> PASSED" || { echo "> FAILED"; exit 1; }
endef

tests:
	$(MAKE) --no-print-directory general
	$(MAKE) --no-print-directory $(OUTDIR)/$(INPUT)

tests.dev:
	$(MAKE) SKIP_CHECK=true tests

tests.md5s: $(INPUT)
	rm -f $@.checksums; \
	for i in tests.out/$@.*.out; do \
		md5sum $$i >> $@.checksums; \
	done


$(OUTDIR):
	mkdir $(OUTDIR)

$(INPUT):
	echo "input;A;B;C;D"      > $@
	echo "0;1;2;3;4"          >> $@
	echo "1;1;2;3;4"          >> $@
	echo "2;1;2;3;4"          >> $@
	echo "3;1;2;3;4"          >> $@
	echo "4;1;2;3;4"          >> $@
	echo ";NaN;nan;nAn;"     >> $@
	echo "i0;1;2;3;4"         >> $@
	echo "i1;1;2;3;4"         >> $@
	echo "i2;1;2;3;4"         >> $@
	echo "i3;1;2;3;4"         >> $@
	echo "i4;1;2;3;4"         >> $@
	echo "f0;1.2;2.2;3.2;4.2" >> $@
	echo "f1;1.4;2.4;3.4;4.4" >> $@
	echo "f2;1.6;2.6;3.6;4.6" >> $@
	echo "f3;2.8;3.8;4.8;5.8" >> $@
	echo "f3;2;3;4;5"         >> $@

general:
	@echo "> Checking plot.py executeable..."
	$(PLOTPY) --help
	@echo "> PASSED"

$(OUTDIR)/$(INPUT) : $(INPUT) $(OUTDIR)
	@echo "> Reading in $< and writing it as text file..."
	$(PLOTPY) -i $< --file $@.0.out
	@$(call check_output,$<,0)
	@echo "> Reading in $< and writing it as pickle file..."
	$(PLOTPY) -i $< --pickle $@.1.out
	@$(call check_output,$<,1)
	@echo "> Reading in $< and printing it..."
	$(PLOTPY) -i $< --print > $@.2.out
	@$(call check_output,$<,2)
	@echo "> Reading in $@.1.out and printing it..."
	$(PLOTPY) -i $@.1.out --print > $@.3.out
	@$(call check_output,$<,3)
	@echo "> Create basic line plot as py, png, jpg and pdf..."
	$(PLOTPY) -i $< --plot line -o $@.line.py $@.line.png $@.line.jpg $@.line.pdf -q
	@$(call check_exists,$@.line.py)
	@$(call check_exists,$@.line.png)
	@$(call check_exists,$@.line.jpg)
	@$(call check_exists,$@.line.pdf)
	@echo "> Create basic bar plot as py, png, jpg and pdf..."
	$(PLOTPY) -i $< --plot bar -o $@.bar.py $@.bar.png $@.bar.jpg $@.bar.pdf -q
	@$(call check_exists,$@.bar.py)
	@$(call check_exists,$@.bar.png)
	@$(call check_exists,$@.bar.jpg)
	@$(call check_exists,$@.bar.pdf)
	@echo "> Create basic box plot as py, png, jpg and pdf..."
	$(PLOTPY) -i $< --plot box -o $@.box.py $@.box.png $@.box.jpg $@.box.pdf -q
	@$(call check_exists,$@.box.py)
	@$(call check_exists,$@.box.png)
	@$(call check_exists,$@.box.jpg)
	@$(call check_exists,$@.box.pdf)
	@echo "> Create basic violin plot as py, png, jpg and pdf..."
	$(PLOTPY) -i $< --plot violin -o $@.violin.py $@.violin.png $@.violin.jpg $@.violin.pdf -q
	@$(call check_exists,$@.violin.py)
	@$(call check_exists,$@.violin.png)
	@$(call check_exists,$@.violin.jpg)
	@$(call check_exists,$@.violin.pdf)


clean:
	rm -Rf $(INPUT) $(OUTDIR)
