COVERAGE_REPORT=coverage
HTML_DIR=build/html
LATEX_DIR=build/latex
WWW_DIR=build/website
DOCSRC_DIR=doc

#
# Determine details on the Python/system
#

PYVER := $(shell python -V 2>&1 | cut -d ' ' -f 2,2 | cut -d '.' -f 1,2)
DISTUTILS_PLATFORM := \
	$(shell \
		python -c "import distutils.util; print distutils.util.get_platform()")

#
# Building
#

all: build

build: 
	python setup.py config --noisy
	python setup.py build


#
# Cleaning
#

clean:
	-rm -rf build
	-rm *-stamp

distclean: clean
	-rm MANIFEST
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
	-rm build-stamp
#	-rm tests/data/*.hdr.* tests/data/*.img.* tests/data/something.nii \
#		tests/data/noise* tests/data/None.nii


#
# Little helpers
#

$(WWW_DIR):
	if [ ! -d $(WWW_DIR) ]; then mkdir -p $(WWW_DIR); fi


#
# Tests
#

test: unittest testmanual


ut-%: build
	@PYTHONPATH=.:$(PYTHONPATH) nosetests nifti/tests/test_$*.py


unittest: build
	@PYTHONPATH=.:$(PYTHONPATH) nosetests nifti --with-doctest

testmanual: build
# go into data, because docs assume now data dir
	@PYTHONPATH=.:$(PYTHONPATH) nosetests --with-doctest --doctest-extension=.txt doc


coverage: build
	@PYTHONPATH=.:$(PYTHONPATH) nosetests --with-coverage


#
# Documentation
#

htmldoc: build
	cd $(DOCSRC_DIR) && PYTHONPATH=$(CURDIR) $(MAKE) html


pdfdoc: build
	cd $(DOCSRC_DIR) && PYTHONPATH=$(CURDIR) $(MAKE) latex
	cd $(LATEX_DIR) && $(MAKE) all-pdf


#
# Website
#

website: website-stamp
website-stamp: $(WWW_DIR) htmldoc pdfdoc
	cp -r $(HTML_DIR)/* $(WWW_DIR)
	cp $(LATEX_DIR)/*.pdf $(WWW_DIR)
	touch $@


upload-website: website
	rsync -rzhvp --delete --chmod=Dg+s,g+rw $(WWW_DIR)/* \
		web.sourceforge.net:/home/groups/n/ni/niftilib/htdocs/nibabel/

#
# Sources
#

pylint: distclean
	# do distclean first to silence SWIG's sins
	pylint --rcfile doc/misc/pylintrc nifti


#
# Distributions
#

# we need to build first to be able to update the manpage
orig-src: build
	# clean existing dist dir first to have a single source tarball to process
	-rm -rf dist
	# now clean
	$(MAKE) distclean
	# check versions
	grep -iR 'version[ ]*[=:]' * | python tools/checkversion.py
	# let python create the source tarball
	python setup.py sdist --formats=gztar
	# rename to proper Debian orig source tarball and move upwards
	# to keep it out of the Debian diff
	file=$$(ls -1 dist); \
		ver=$${file%*.tar.gz}; \
		ver=$${ver#nibabel-*}; \
		mv dist/$$file ../nibabel_$$ver.tar.gz


bdist_rpm:
	python setup.py bdist_rpm \
	  --doc-files "doc" \
	  --packager "nibabel authors <pkg-exppsy-pynifti@lists.alioth.debian.org>" \
	  --vendor "nibabel authors <pkg-exppsy-pynifti@lists.alioth.debian.org>"


# build MacOS installer -- depends on patched bdist_mpkg for Leopard
bdist_mpkg:
	python tools/mpkg_wrapper.py setup.py install


.PHONY: orig-src pylint
