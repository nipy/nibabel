''' Defaults for volumeimages '''

import logging

error_level = 40
log_level = 30
logger = logging.getLogger('nifti.global')
logger.addHandler(logging.StreamHandler())
