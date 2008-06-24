# Minimalistic makefile for libnifti
LIB=libniftiio.a libznz.a
MISC=nifti1_io.c znzlib.c

LIBFLAGS=-fPIC
CFLAGS=-Wall -O2 -I. -DHAVE_ZLIB

all: $(LIB)

.c.o:
	gcc $(CFLAGS) $(LIBFLAGS) -c $^

libniftiio.a: nifti1_io.o
	ar cur $@ $^

libznz.a: znzlib.o
	ar cur $@ $^

clean:
	rm -f $(LIB) $(MISC:.c=.o)
