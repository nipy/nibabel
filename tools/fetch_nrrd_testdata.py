#!/usr/bin/env python
"""Download the NRRD test files into nibabel-data/nitest-nrrd

The files are the ``TestingData`` release of slicerio, revision rev001, written
by 3D Slicer, ITK and pynrrd. See
https://github.com/lassoan/slicerio/releases/tag/TestingData

Each file is checked against the SHA256 checksum below. Files that are already
present with the right checksum are not downloaded again. Run from the nibabel
root directory::

    python tools/fetch_nrrd_testdata.py [output_directory]
"""

import hashlib
import os
import shutil
import sys
import urllib.request

BASE_URL = 'https://github.com/lassoan/slicerio/releases/download/TestingData/'
DEFAULT_DIR = os.path.join('nibabel-data', 'nitest-nrrd')

# From manifest_rev001.json of the release
FILES = {
    'CTCardio_2Dt.seq.nrrd': 'a672d6bbb9d400e6ba1812b17ad90660b90ffd9e45a1552328bc2ded7ed2df08',
    'CTCardio_3Dt.seq.nrrd': '88231385aa9c25a16831244514c943b45ccfc9e779cfd9cc0979ca9a22743b51',
    'CTCardio_3Dt_displacement.seq.nrrd': '20c66b531998947afdf1f890ef0b401cbbbce54aeeb76631345ea668572341dc',
    'CTCardio_3Dt_displacement_RAS.seq.nrrd': 'e092d799cec19d00ee9bd45442b138041392c147b7adfad44bd6dc1fc3976064',
    'CTCardio_3Dt_displacement_resampled.seq.nrrd': '9ff309a87a0e4d24501e89452351e882815622dc8e34cda25bd4539979c94c36',
    'CTCardio_3Dt_velocity.seq.nrrd': '369b83f7516803b504162a314a9c361e368bc9aae79311ba2c63dc14ee361760',
    'DTIBrain_3D_tensor.nrrd': 'fad191e1f20ece57a41aeec9e6d5f97d3c8942955ac0919be1f71c138a2bdaea',
    'DTIBrain_3D_tensor_symmetric.nrrd': '2f7ae81472c516f5b78c3084b9f60a7a69636884cc2355b0e8b5d8c441b09f8c',
    'DTIBrain_FA.nrrd': '903001f5fdb594014b60fee871e4f5a71798bc3a54669d1fedb12116a41004c9',
    'DTIBrain_baseline.nrrd': '66ccbe7f7f69d72f69c4e495fdbe973dd2043cd043ee1cbe5bf61406fcf93c24',
    'DWI_small.nrrd': '8a0d61c58dbd52b6a64bd355ecc32c8f47ff579cc70016c38e5665532c4c45e4',
    'DWI_small_b0.nrrd': '08380523fe020d99708c9a794793779b0355ae3801edb0116e9479b3fd53b23e',
    'MRHead_2D.nrrd': '17eee8f967d6a943ce15934f9428449e85270c0ce95d70276a061d32ed542c48',
    'MRHead_3D.nrrd': '616270f3a465cdb8d5e7ff2966e50b6d0ef641b328a12bbb575ab39fa615def1',
    'MRHead_3D_RAS.nrrd': '0662ec1ec9fd16797a53957cb92d26c83fa5cef82ff41bb38933cb5f79d5dcae',
    'MRHead_3D_detached.nhdr': '00c747dbaf90db3510b0adee26b9e2e6bd80600109d91bee52bd74bd070a3600',
    'MRHead_3D_detached.raw.gz': '069fb299c9b0fe5b087496077f516f920278f6eb8b7145fea3a780c460df8ca0',
    'MRHead_3D_oblique.nrrd': 'eb6c4863bb5f963e9c9afd85db5d2df0225e9cd832b1ef2ba0200e68e4c13758',
    'MRHead_3D_raw.nrrd': '234fff7ef38209fb58d69fc3635bedf6e5b095d9f3df7c7b49dd37b3928ff4e1',
    'MRHead_displacement.nrrd': '77d1f47d804155f22f052092ef7c63485a1585c0a180ea4259a2466232af7da2',
    'MRHead_displacement_RAS.nrrd': 'fdc2d17b1bb3c1de812a21d1da6fd72fea75b24cc4026ddbcd939b717e558a24',
    'MRHead_displacement_resampled.nrrd': '86f75ccb3ad52e5cd300aa2708b497f005e46b546e79f703a913e09149f6cb33',
    'MRHead_oblique_displacement.nrrd': '9ca15dab9ffb114e0bd4e86bcf6e8cd42d6eb5b17ebb30d7fa8e3d591fb83db7',
    'MRHead_oblique_displacement_resampled.nrrd': '2e580de25fac31f88a1925689cfcad428f113cd7ef51fc0d1a5e9e9e00d8e203',
    'MRHead_segmentation.seg.nrrd': '20629a7a4f16552a69e03acade00b252c78fa88e24cf5aeef866b56ae19812d5',
    'MRHead_segmentation_RAS.seg.nrrd': '4d5675df94087528b1eab9024503e1ab2301083dc65b9a99d87b4461da810b37',
    'MRHead_segmentation_oblique.seg.nrrd': '071d6b7c67d4cd7f4a6ec76b08ced22c7ff10c4ed776fc962bdb91bc54421672',
    'MRHead_segmentation_overlapping.seg.nrrd': 'a9fbac9d4e4b69d540d10ec9024c28d89c5d8494b063906677dbdb9dbe2cb8fc',
    'MRHead_velocity.nrrd': '1144cc542359a76fd58e9d479dc4abce86bc803ff0d96ce267420fff302e73e1',
    'MRHead_velocity_RAS.nrrd': 'b986b91d6b33a7850d28ec3c0c21cb05419d514d42cab73e505dc3a30baf753e',
    'VisibleHumanHead_2D_RGB.nrrd': 'c88d998d63ccf40e1c1ad98e4197117669afa19ead08fc6e1f3360a2ecbcf85d',
    'VisibleHumanHead_2D_RGBA.nrrd': 'bee20fb484864a6310e3291fa4b6899e263a59a5e371a55bd4abea7b43903c20',
    'VisibleHumanHead_3D_RGB.nrrd': '8fe06f116408657b1ec0e496dab7ec7504c3d82a325eb3d1a565203d33b70a60',
    'VisibleHumanHead_3D_RGBA.nrrd': '1b8018fed1ef4314079d23c310d78d7598698b131c15654535df065ce0b5bc9e',
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fobj:
        for block in iter(lambda: fobj.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch(name, expected, out_dir):
    path = os.path.join(out_dir, name)
    if os.path.exists(path) and sha256(path) == expected:
        print(f'{name}: already present')
        return
    part = path + '.part'
    print(f'{name}: downloading')
    with urllib.request.urlopen(BASE_URL + name, timeout=60) as response, open(part, 'wb') as fobj:
        shutil.copyfileobj(response, fobj)
    actual = sha256(part)
    if actual != expected:
        os.remove(part)
        raise RuntimeError(f'{name}: SHA256 is {actual}, expected {expected}')
    os.replace(part, path)


def main(argv):
    out_dir = argv[1] if len(argv) > 1 else DEFAULT_DIR
    os.makedirs(out_dir, exist_ok=True)
    for name, expected in FILES.items():
        fetch(name, expected, out_dir)
    print(f'{len(FILES)} files in {out_dir}')


if __name__ == '__main__':
    main(sys.argv)
