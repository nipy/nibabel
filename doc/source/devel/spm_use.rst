.. -*- mode: rst -*-

========================
 Image use-cases in SPM
========================

SPM uses a *vol struct* as a structure characterizing an object.  This
is a Matlab ``struct``.  A ``struct`` is like a Python dictionary, where
field names (strings) are associated with values.  There are various
functions operating on vol structs, so the vol struct is rather like an
object, where the methods are implemented as functions.  Actually, the
distinction between methods and functions in Matlab is fairly subtle -
their call syntax is the same for example.

.. sourcecode:: matlab

   >> fname = 'some_image.nii';
   >> vol = spm_vol(fname) % the vol struct
   
   vol = 

         fname: 'some_image.nii'
           mat: [4x4 double]
           dim: [91 109 91]
            dt: [2 0]
         pinfo: [3x1 double]
             n: [1 1]
       descrip: 'NIFTI-1 Image'
       private: [1x1 nifti]

   >> vol.mat % the 'affine'

   ans =

       -2     0     0    92
        0     2     0  -128
        0     0     2   -74
        0     0     0     1

   >> help spm_vol 
     Get header information etc for images.
     FORMAT V = spm_vol(P)
     P - a matrix of filenames.
     V - a vector of structures containing image volume information.
     The elements of the structures are:
           V.fname - the filename of the image.
           V.dim   - the x, y and z dimensions of the volume
           V.dt    - A 1x2 array.  First element is datatype (see spm_type).
                     The second is 1 or 0 depending on the endian-ness.
           V.mat   - a 4x4 affine transformation matrix mapping from
                     voxel coordinates to real world coordinates.
           V.pinfo - plane info for each plane of the volume.
                  V.pinfo(1,:) - scale for each plane
                  V.pinfo(2,:) - offset for each plane
                     The true voxel intensities of the jth image are given
                     by: val*V.pinfo(1,j) + V.pinfo(2,j)
                  V.pinfo(3,:) - offset into image (in bytes).
                     If the size of pinfo is 3x1, then the volume is assumed
                     to be contiguous and each plane has the same scalefactor
                     and offset.
    ____________________________________________________________________________

     The fields listed above are essential for the mex routines, but other
     fields can also be incorporated into the structure.

     The images are not memory mapped at this step, but are mapped when
     the mex routines using the volume information are called.

     Note that spm_vol can also be applied to the filename(s) of 4-dim
     volumes. In that case, the elements of V will point to a series of 3-dim
     images.

     This is a replacement for the spm_map_vol and spm_unmap_vol stuff of
     MatLab4 SPMs (SPM94-97), which is now obsolete.
    _______________________________________________________________________
     Copyright (C) 2005 Wellcome Department of Imaging Neuroscience


   >> spm_type(vol.dt(1))

   ans =

   uint8

   >> vol.private

   ans = 

   NIFTI object: 1-by-1
               dat: [91x109x91 file_array]
               mat: [4x4 double]
        mat_intent: 'MNI152'
              mat0: [4x4 double]
       mat0_intent: 'MNI152'
           descrip: 'NIFTI-1 Image'


So, in our (provisional) terms:

* ``vol.mat`` == ``img.affine``
* ``vol.dim`` == ``img.shape``
* ``vol.dt(1)`` (``vol.dt[0]`` in Python) is equivalent to
  ``img.get_data_dtype()``
* ``vol.fname`` == ``img.get_filename()``

SPM abstracts the implementation of the image to the ``vol.private``
member, that is not in fact required by the image interface.

Images in SPM are always 3D.  Note this behavior:

.. sourcecode:: matlab

   >> fname = 'functional_01.nii';
   >> vol = spm_vol(fname)

   vol = 

   191x1 struct array with fields:
       fname
       mat
       dim
       dt
       pinfo
       n
       descrip
       private

That is, one vol struct per 3D volume in a 4D dataset.

SPM image methods / functions
=============================

Some simple ones:

.. sourcecode:: matlab

   >> fname = 'some_image.nii';
   >> vol = spm_vol(fname);
   >> img_arr = spm_read_vols(vol);
   >> size(img_arr) % just loads in scaled data array

   ans =

       91   109    91

   >> spm_type(vol.dt(1)) % the disk-level (IO) type is uint8

   ans =

   uint8

   >> class(img_arr) % always double regardless of IO type

   ans =

   double

   >> new_fname = 'another_image.nii';
   >> new_vol = vol;  % matlab always copies
   >> new_vol.fname = new_fname;
   >> spm_write_vol(new_vol, img_arr)

   ans = 

         fname: 'another_image.nii'
           mat: [4x4 double]
           dim: [91 109 91]
            dt: [2 0]
         pinfo: [3x1 double]
             n: [1 1]
       descrip: 'NIFTI-1 Image'
       private: [1x1 nifti]


Creating an image from scratch, and writing plane by plane (slice by slice):

.. sourcecode:: matlab

   >> new_vol = struct();
   >> new_vol.fname = 'yet_another_image.nii';
   >> new_vol.dim = [91 109 91];
   >> new_vol.dt = [spm_type('float32') 0]; % little endian (0)
   >> new_vol.mat = vol.mat;
   >> new_vol.pinfo = [1 0 0]';
   >> new_vol = spm_create_vol(new_vol);
   >> for vox_z = 1:new_vol.dim(3)
   new_vol = spm_write_plane(new_vol, img_arr(:,:,vox_z), vox_z);
   end

I think it's true that writing the plane does not change the image
scalefactors, so it's only practical to use ``spm_write_plane`` for data
for which you already know the dynamic range across the volume.

Simple resampling from an image:

.. sourcecode:: matlab

   >> fname = 'some_image.nii';
   >> vol = spm_vol(fname);
   >> % for voxel coordinate 10,15,20 (1-based)
   >> hold_val = 3; % third order spline resampling
   >> val = spm_sample_vol(vol, 10, 15, 20, hold_val)

   val =

       0.0510

   >> img_arr = spm_read_vols(vol);
   >> img_arr(10, 15, 20)  % same as simple indexing for integer coordinates

   ans =

       0.0510

   >> % more than one point
   >> x = [10, 10.5]; y = [15, 15.5]; z = [20, 20.5];
   >> vals = spm_sample_vol(vol, x, y, z, hold_val)

   vals =

       0.0510    0.0531

   >> % you can also get the derivatives, by asking for more output args
   >> [vals, dx, dy, dz] = spm_sample_vol(vol, x, y, z, hold_val)

   vals =

       0.0510    0.0531


   dx =

       0.0033    0.0012


   dy =

       0.0033    0.0012


   dz =

       0.0020   -0.0017


This is to speed up optimization in registration - where the optimizer
needs the derivatives.

``spm_sample_vol`` always works in voxel coordinates.  If you want some
other coordinates, you would transform them yourself.  For example,
world coordinates according to the affine looks like:

.. sourcecode:: matlab

   >> wc = [-5, -12, 32];
   >> vc = inv(vol.mat) * [wc 1]'

   vc =

      48.5000
      58.0000
      53.0000
       1.0000

   >> vals = spm_sample_vol(vol, vc(1), vc(2), vc(3), hold_val)  

   vals =

       0.6792

Odder sampling, often used, can be difficult to understand:

.. sourcecode:: matlab

   >> slice_mat = eye(4);
   >> out_size = vol.dim(1:2);
   >> slice_no = 4; % slice we want to fetch
   >> slice_mat(3,4) = slice_no;
   >> arr_slice = spm_slice_vol(vol, slice_mat, out_size, hold_val);
   >> img_slice_4 = img_arr(:,:,slice_no);
   >> all(arr_slice(:) == img_slice_4(:))

   ans =

        1


This is the simplest use - but in general any affine transform can go in
``slice_mat`` above, giving optimized (for speed) sampling of slices
from volumes, as long as the transform is an affine.

Miscellaneous functions operating on vol structs:

* ``spm_conv_vol`` - convolves volume with seperable functions in x, y, z
* ``spm_render_vol`` - does a projection of a volume onto a surface
* ``spm_vol_check`` - takes array of vol structs and checks for sameness of
  image dimensions and ``mat`` (affines) across the list.

And then, many SPM functions accept vol structs as arguments.
