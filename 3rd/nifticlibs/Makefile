# Minimalistic makefile for libnifti
LIB=libniftiio.a libznz.a
MISC=nifti1_io.c znzlib.c

LIBFLAGS=-fPIC
CFLAGS=-Wall -O2 -I. -DHAVE_ZLIB

# Use the below CFLAGS on OSX
# CFLAGS=-Wall -O2 -I. -DHAVE_ZLIB -arch i386 -arch ppc

all: $(LIB)

.c.o:
	gcc $(CFLAGS) $(LIBFLAGS) -c $^

libniftiio.a: nifti1_io.o
	ar curs $@ $^

libznz.a: znzlib.o
	ar curs $@ $^

clean:
	rm -f $(LIB) $(MISC:.c=.o)
