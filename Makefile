
PROFILE_FILE=tests/main.pstats
COVERAGE_REPORT=coverage
HTML_DIR=build/html
PDF_DIR=build/pdf
WWW_DIR=build/website

PYVER := $(shell pyversions -vd)
ARCH := $(shell uname -m)

rst2latex=rst2latex --documentclass=scrartcl \
					--use-latex-citations \
					--strict \
					--use-latex-footnotes \
					--stylesheet ../../doc/misc/style.tex

rst2html=rst2html --date --strict --stylesheet=nifti.css --link-stylesheet


all: build


build: build-stamp
build-stamp:
	python setup.py config --noisy
	python setup.py build_ext
	python setup.py build_py
	# to overcome the issue of not-installed _nifticlib.so
	ln -sf ../build/lib.linux-$(ARCH)-$(PYVER)/nifti/_nifticlib.so nifti/
	touch $@


clean: distclean
distclean:
	-rm MANIFEST
	-rm nifti/*.{c,pyc,pyo,so} nifti/nifticlib.py
	-rm tests/*.pyc
	-rm $(COVERAGE_REPORT)
	@find . -name '*.py[co]' \
		 -o -name '*,cover' \
		 -o -name '.coverage' \
		 -o -iname '*~' \
		 -o -iname '*.kcache' \
		 -o -iname '*.pstats' \
		 -o -iname '*.prof' \
		 -o -iname '#*#' | xargs -l10 rm -f
	-rm -r build
	-rm -r dist
	-rm build-stamp apidoc-stamp


$(PROFILE_FILE): build tests/main.py
	@cd tests && \
		PYTHONPATH=.. ../tools/profile -K  -O ../$(PROFILE_FILE) main.py


$(HTML_DIR):
	if [ ! -d $(HTML_DIR) ]; then mkdir -p $(HTML_DIR); fi


$(PDF_DIR):
	if [ ! -d $(PDF_DIR) ]; then mkdir -p $(PDF_DIR); fi


$(WWW_DIR):
	if [ ! -d $(WWW_DIR) ]; then mkdir -p $(WWW_DIR); fi


test-%: build
	@cd tests && PYTHONPATH=.. python test_$*.py


test: build
	@cd tests && PYTHONPATH=.. python main.py


coverage: build
	@cd tests && { \
	  export PYTHONPATH=..; \
	  python-coverage -x main.py; \
	  python-coverage -r -i -o /usr >| ../$(COVERAGE_REPORT); \
	  grep -v '100%$$' ../$(COVERAGE_REPORT); \
	  python-coverage -a -i -o /usr; }


#
# Documentation
#

apidoc: apidoc-stamp
apidoc-stamp: $(PROFILE_FILE)
	mkdir -p $(HTML_DIR)/api
	epydoc --config doc/api/epydoc.conf
	touch $@


htmlchangelog: $(HTML_DIR)
	$(rst2html) Changelog $(HTML_DIR)/changelog.html


htmlmanual: $(HTML_DIR)
	$(rst2html) doc/manual/manual.txt $(HTML_DIR)/manual.html
	# copy images
	cp -r doc/misc/{*.css,pics} doc/manual/pics $(HTML_DIR)


# convert rsT documentation in doc/* to PDF.
pdfmanual: $(PDF_DIR)
	cat doc/manual/manual.txt Changelog | $(rst2latex) > $(PDF_DIR)/manual.tex
	-cp -r doc/manual/pics $(PDF_DIR)
	cd $(PDF_DIR) && pdflatex manual.tex


website: $(WWW_DIR) htmlmanual htmlchangelog pdfmanual apidoc
	cp $(HTML_DIR)/manual.html $(WWW_DIR)/index.html
	cp -r $(HTML_DIR)/{pics,changelog.html,*.css} $(WWW_DIR)
	cp $(PDF_DIR)/manual.pdf $(WWW_DIR)
	cp -r $(HTML_DIR)/api $(WWW_DIR)


upload-website: website
	rsync -rzhvp --delete --chmod=Dg+s,g+rw $(WWW_DIR)/* \
		shell.sourceforge.net:/home/groups/n/ni/niftilib/htdocs/pynifti/


printables: pdfmanual


#
# Sources
#

pylint: distclean
	# do distclean first to silence SWIG's sins
	pylint --rcfile doc/misc/pylintrc nifti


orig-src: distclean 
	# clean existing dist dir first to have a single source tarball to process
	-rm -rf dist
	# update manpages
	help2man -N -n "compute peristimulus timeseries of fMRI data" \
		bin/pynifti_pst > man/pynifti_pst.1
	# check versions
	grep -iR 'version[ ]*[=:]' * | python tools/checkversion.py
	# let python create the source tarball
	python setup.py sdist --formats=gztar
	# rename to proper Debian orig source tarball and move upwards
	# to keep it out of the Debian diff
	file=$$(ls -1 dist); \
		ver=$${file%*.tar.gz}; \
		ver=$${ver#pynifti-*}; \
		mv dist/$$file ../pynifti_$$ver.tar.gz


bdist_wininst:
	# THIS IS ONLY FOR WINDOWS!
	# Consider this a set of notes on how to build PyNIfTI on win32, rather
	# than an actually working target
	#
	# assumes Dev-Cpp to be installed at C:\Dev-Cpp
	python setup.py build_ext -c mingw32 --swig-opts \
		"-C:\Dev-Cpp\include/nifti -DWIN32" \
		-IC:\Dev-Cpp\include nifti
	
	# for some stupid reason the swig wrapper is in the wrong location
	move /Y nifticlib.py nifti
	
	# now build the installer
	python setup.py bdist_wininst


.PHONY: orig-src pylint apidoc
