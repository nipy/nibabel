# Minimalistic makefile for libnifti
LIB=libniftiio.a libznz.a
MISC=nifti1_io.c znzlib.c

LIBFLAGS=-fPIC
CFLAGS=-Wall -O2 -I. -DHAVE_ZLIB

# Use the below CFLAGS on OSX
# CFLAGS=-Wall -O2 -I. -DHAVE_ZLIB -arch i386 -arch ppc
TDIR=../../build/nifticlibs

all: prep $(LIB)

prep: prep-stamp
prep-stamp:
	mkdir -p $(TDIR)
	touch $@


.c.o:
	gcc $(CFLAGS) $(LIBFLAGS) -c $^ -o $(TDIR)/$(^:.c=.o)

libniftiio.a: nifti1_io.o
	ar curs $(TDIR)/$@ $(TDIR)/$^

libznz.a: znzlib.o
	ar curs $(TDIR)/$@ $(TDIR)/$^

clean:
	rm -rf $(TDIR)
	rm *-stamp
