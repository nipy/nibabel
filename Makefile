COVERAGE_REPORT=coverage
HTML_DIR=build/html
LATEX_DIR=build/latex
WWW_DIR=build/website
DOCSRC_DIR=doc
SF_USER ?= matthewbrett
#
# The Python executable to be used
#
PYTHON ?= python
NOSETESTS = $(PYTHON) $(shell which nosetests)

#
# Determine details on the Python/system
#

PYVER := $(shell $(PYTHON) -V 2>&1 | cut -d ' ' -f 2,2 | cut -d '.' -f 1,2)
DISTUTILS_PLATFORM := \
	$(shell \
		$(PYTHON) -c "import distutils.util; print(distutils.util.get_platform())")

# Helpers for version handling.
# Note: can't be ':='-ed since location of invocation might vary
DEBCHANGELOG_VERSION = $(shell dpkg-parsechangelog | egrep ^Version | cut -d ' ' -f 2,2 | cut -d '-' -f 1,1)
SETUPPY_VERSION = $(shell $(PYTHON) setup.py -V)
#
# Automatic development version
#
#yields: LastTagName_CommitsSinceThat_AbbrvHash
DEV_VERSION := $(shell git describe --abbrev=4 HEAD |sed -e 's/-/+/g' |cut -d '/' -f 2,2)

# By default we are releasing with setup.py version
RELEASE_VERSION ?= $(SETUPPY_VERSION)

#
# Building
#

all: build

build: 
	$(PYTHON) setup.py config --noisy
	$(PYTHON) setup.py build


#
# Cleaning
#

clean:
	$(MAKE) -C doc clean
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
	-rm -r .tox
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
	@PYTHONPATH=.:$(PYTHONPATH) $(NOSETESTS) nibabel/tests/test_$*.py


unittest: build
	@PYTHONPATH=.:$(PYTHONPATH) $(NOSETESTS) nibabel --with-doctest

testmanual: build
	@cd doc/source && PYTHONPATH=../..:$(PYTHONPATH) $(NOSETESTS) --with-doctest --doctest-extension=.rst . dicom


coverage: build
	@PYTHONPATH=.:$(PYTHONPATH) $(NOSETESTS) --with-coverage --cover-package=nibabel


#
# Documentation
#

htmldoc: build
	cd $(DOCSRC_DIR) && PYTHONPATH=$(CURDIR):$(PYTHONPATH) $(MAKE) html


pdfdoc: build
	cd $(DOCSRC_DIR) && PYTHONPATH=$(CURDIR):$(PYTHONPATH) $(MAKE) latex
	cd $(LATEX_DIR) && $(MAKE) all-pdf


gitwash-update: build
	cd $(DOCSRC_DIR) && PYTHONPATH=$(CURDIR):$(PYTHONPATH) $(MAKE) gitwash-update

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

# This one udates for the specific user named at the top of the makefile
upload-htmldoc: htmldoc upload-htmldoc-$(SF_USER)

upload-htmldoc-%: htmldoc
	rsync -rzhvp --delete --chmod=Dg+s,g+rw $(HTML_DIR)/* \
		$*,nipy@web.sourceforge.net:/home/groups/n/ni/nipy/htdocs/nibabel/

#
# Sources
#

pylint: distclean
	# do distclean first to silence SWIG's sins
	PYTHONPATH=.:$(PYTHONPATH) pylint --rcfile doc/misc/pylintrc nibabel


#
# Distributions
#

# Check either everything was committed
check-nodirty:
	# Need to run in clean tree. If fails: commit or clean first
	[ "x$$(git diff)" = "x" ]
# || $(error "")

check-debian:
	# Need to run in a Debian packaging branch
	[ -d debian ]

check-debian-version: check-debian
	# Does debian version correspond to setup.py version?
	[ "$(DEBCHANGELOG_VERSION)" = "$(SETUPPY_VERSION)" ]

embed-dev-version: check-nodirty
	# change upstream version
	sed -i -e "s/$(SETUPPY_VERSION)/$(DEV_VERSION)/g" setup.py nibabel/__init__.py
	# change package name
	sed -i -e "s/= 'nibabel',/= 'nibabel-snapshot',/g" setup.py

deb-dev-autochangelog: check-debian
	# removed -snapshot from pkg name for now
	$(MAKE) check-debian-version || \
		dch --newversion $(DEV_VERSION)-1 --package nibabel-snapshot \
		 --allow-lower-version "NiBabel development snapshot."

deb-mergedev:
	git merge --no-commit origin/dist/debian/dev

orig-src: distclean distclean
	# clean existing dist dir first to have a single source tarball to process
	-rm -rf dist
	# let python create the source tarball
	$(PYTHON) setup.py sdist --formats=gztar
	# rename to proper Debian orig source tarball and move upwards
	# to keep it out of the Debian diff
	tbname=$$(basename $$(ls -1 dist/*tar.gz)) ; ln -s $${tbname} ../nibabel-snapshot_$(DEV_VERSION).orig.tar.gz
	mv dist/*tar.gz ..
	# clean leftover
	rm MANIFEST

devel-src: check-nodirty
	-rm -rf dist
	git clone -l . dist/nibabel-snapshot
	RELEASE_VERSION=$(DEV_VERSION) \
	  $(MAKE) -C dist/nibabel-snapshot -f ../../Makefile embed-dev-version orig-src
	mv dist/*tar.gz ..
	rm -rf dist

devel-dsc: check-nodirty
	-rm -rf dist
	git clone -l . dist/nibabel-snapshot
	RELEASE_VERSION=$(DEV_VERSION) \
	  $(MAKE) -C dist/nibabel-snapshot -f ../../Makefile embed-dev-version orig-src deb-mergedev deb-dev-autochangelog
	# create the dsc -- NOT using deb-src since it would clean the hell first
	cd dist && dpkg-source -i'\.(gbp.conf|git\.*)' -b nibabel-snapshot
	mv dist/*.gz dist/*dsc ..
	rm -rf dist

# make Debian source package
# # DO NOT depend on orig-src here as it would generate a source tarball in a
# Debian branch and might miss patches!
deb-src: check-debian distclean
	cd .. && dpkg-source -i'\.(gbp.conf|git\.*)' -b $(CURDIR)


bdist_rpm:
	$(PYTHON) setup.py bdist_rpm \
	  --doc-files "doc" \
	  --packager "nibabel authors <http://mail.scipy.org/mailman/listinfo/nipy-devel>"
	  --vendor "nibabel authors <http://mail.scipy.org/mailman/listinfo/nipy-devel>"


# build MacOS installer -- depends on patched bdist_mpkg for Leopard
bdist_mpkg:
	$(PYTHON) tools/mpkg_wrapper.py setup.py install

# Check for files not installed
check-files:
	$(PYTHON) -c 'from nisext.testers import check_files; check_files("nibabel")'

# Print out info for possible install methods
check-version-info:
	$(PYTHON) -c 'from nisext.testers import info_from_here; info_from_here("nibabel")'

# Run tests from installed code
installed-tests:
	$(PYTHON) -c 'from nisext.testers import tests_installed; tests_installed("nibabel")'

# Run tests from packaged distributions
sdist-tests:
	$(PYTHON) -c 'from nisext.testers import sdist_tests; sdist_tests("nibabel", doctests=False)'

bdist-egg-tests:
	$(PYTHON) -c 'from nisext.testers import bdist_egg_tests; bdist_egg_tests("nibabel", doctests=False, label="not script_test")'

sdist-venv: clean
	rm -rf dist venv
	unset PYTHONPATH && $(PYTHON) setup.py sdist --formats=zip
	virtualenv --system-site-packages --python=$(PYTHON) venv
	. venv/bin/activate && pip install --ignore-installed nose
	mkdir venv/tmp
	cd venv/tmp && unzip ../../dist/*.zip
	. venv/bin/activate && cd venv/tmp/nibabel* && python setup.py install
	unset PYTHONPATH && . venv/bin/activate && cd venv && nosetests --with-doctest nibabel nisext

source-release: distclean
	$(PYTHON) -m compileall .
	make distclean
	$(PYTHON) setup.py sdist --formats=gztar,zip

venv-tests:
	# I use this for python2.5 because the sdist-tests target doesn't work
	# (the tester routine uses a 2.6 feature)
	make distclean
	- rm -rf $(VIRTUAL_ENV)/lib/python$(PYVER)/site-packages/nibabel
	$(PYTHON) setup.py install
	cd .. && nosetests $(VIRTUAL_ENV)/lib/python$(PYVER)/site-packages/nibabel

tox-fresh:
	# tox tests with fresh-installed virtualenvs.  Needs network.  And
	# pytox, obviously.
	tox -c tox.ini

tox-stale:
	# tox tests with MB's already-installed virtualenvs (numpy and nose
	# installed)
	tox -e python25,python26,python27,python32,np-1.2.1

.PHONY: orig-src pylint

