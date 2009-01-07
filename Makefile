PROFILE_FILE=tests/main.pstats
COVERAGE_REPORT=coverage
HTML_DIR=build/html
PDF_DIR=build/pdf
WWW_DIR=build/website

#
# Details on the Python/system
#

PYVER := $(shell python -V 2>&1 | cut -d ' ' -f 2,2 | cut -d '.' -f 1,2)
DISTUTILS_PLATFORM := $(shell python -c "import distutils.util; print distutils.util.get_platform()")


all: build

# build included 3rd party pieces (if present)
3rd: 3rd-stamp
3rd-stamp:
	find 3rd -mindepth 1 -maxdepth 1  -type d | \
	 while read d; do \
	  [ -f "$$d/Makefile" ] && $(MAKE) -C "$$d"; \
     done
	touch $@


build: build-stamp
build-stamp: 3rd
	python setup.py config --noisy
	python setup.py build_ext
	python setup.py build_py
	# to overcome the issue of not-installed _nifticlib.so
	ln -sf ../build/lib.$(DISTUTILS_PLATFORM)-$(PYVER)/nifti/_nifticlib.so nifti/
	ln -sf ../build/src.$(DISTUTILS_PLATFORM)-$(PYVER)/nifti/nifticlib.py nifti/
	touch $@


clean:
	-rm -rf build
	-rm *-stamp
	-rm nifti/nifticlib.py nifti/_nifticlib.so
	find 3rd -mindepth 2 -maxdepth 2  -type f -name '*-stamp' | xargs -L10 rm -f


distclean: clean
	-rm MANIFEST
	-rm tests/*.pyc
	-rm $(COVERAGE_REPORT)
	@find . -name '*.py[co]' \
		 -o -name '*.a' \
		 -o -name '*,cover' \
		 -o -name '.coverage' \
		 -o -iname '*~' \
		 -o -iname '*.kcache' \
		 -o -iname '*.pstats' \
		 -o -iname '*.prof' \
		 -o -iname '#*#' | xargs -L10 rm -f
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


ut-%: build
	@cd tests && PYTHONPATH=.. python test_$*.py


unittest: build
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

htmldoc: build $(HTML_DIR)
	cd doc && PYTHONPATH=$(CURDIR) $(MAKE) html


# convert rsT documentation in doc/* to PDF.
pdfmanual: $(PDF_DIR)
	cat doc/manual/manual.txt Changelog | $(rst2latex) > $(PDF_DIR)/manual.tex
	-cp -r doc/manual/pics $(PDF_DIR)
	cd $(PDF_DIR) && pdflatex manual.tex


website: $(WWW_DIR) htmldoc pdfmanual
	cp $(HTML_DIR)/manual.html $(WWW_DIR)/index.html
	cp -r -t $(WWW_DIR) $(HTML_DIR)/pics \
						$(HTML_DIR)/changelog.html \
						$(HTML_DIR)/*.css
	cp $(PDF_DIR)/manual.pdf $(WWW_DIR)
	cp -r $(HTML_DIR)/api $(WWW_DIR)


upload-website: website
	rsync -rzhvp --delete --chmod=Dg+s,g+rw $(WWW_DIR)/* \
		web.sourceforge.net:/home/groups/n/ni/niftilib/htdocs/pynifti/

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


bdist_rpm: 3rd
	python setup.py bdist_rpm \
	  --doc-files "doc" \
	  --packager "Michael Hanke <michael.hanke@gmail.com>" \
	  --vendor "Michael Hanke <michael.hanke@gmail.com>"


# build MacOS installer -- depends on patched bdist_mpkg for Leopard
bdist_mpkg: 3rd
	python tools/mpkg_wrapper.py setup.py build_ext
	python tools/mpkg_wrapper.py setup.py install


.PHONY: orig-src pylint apidoc
