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

# Helpers for version handling.
# Note: can't be ':='-ed since location of invocation might vary
DEBCHANGELOG_VERSION = $(shell dpkg-parsechangelog | egrep ^Version | cut -d ' ' -f 2,2 | cut -d '-' -f 1,1)
SETUPPY_VERSION = $(shell python setup.py -V)
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
	@PYTHONPATH=.:$(PYTHONPATH) nosetests nibabel/tests/test_$*.py


unittest: build
	@PYTHONPATH=.:$(PYTHONPATH) nosetests nibabel --with-doctest

testmanual: build
# go into data, because docs assume now data dir
	@PYTHONPATH=.:$(PYTHONPATH) nosetests --with-doctest --doctest-extension=.txt doc


coverage: build
	@PYTHONPATH=.:$(PYTHONPATH) nosetests --with-coverage --cover-package=nibabel


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
	git merge --no-commit origin/debian/dev

orig-src: distclean distclean
	# clean existing dist dir first to have a single source tarball to process
	-rm -rf dist
	# check versions
	grep -iR 'version[ ]*[=:]' * | python tools/checkversion.py
	# let python create the source tarball
	python setup.py sdist --formats=gztar
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
	python setup.py bdist_rpm \
	  --doc-files "doc" \
	  --packager "nibabel authors <pkg-exppsy-pynifti@lists.alioth.debian.org>" \
	  --vendor "nibabel authors <pkg-exppsy-pynifti@lists.alioth.debian.org>"


# build MacOS installer -- depends on patched bdist_mpkg for Leopard
bdist_mpkg:
	python tools/mpkg_wrapper.py setup.py install


.PHONY: orig-src pylint
