all:

distclean:
	-rm MANIFEST Changelog
	-rm nifti/*.c nifti/*.pyc nifti/*.so
	-rm nifti/nifticlib.py
	-rm -r build
	-rm -r dist

	
orig-src: 
	# clean existing dist dir first to have a single source tarball to process
	-rm -rf dist
	# the debian changelog is also the upstream changelog
	cp debian/changelog Changelog
	# let python create the source tarball
	python setup.py sdist --formats=gztar
	# rename to proper Debian orig source tarball and move upwards
	# to keep it out of the Debian diff
	file=$$(ls -1 dist); ver=$${file%*.tar.gz}; ver=$${ver#pynifti-*}; mv dist/$$file ../pynifti_$$ver.orig.tar.gz
