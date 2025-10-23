name = "actinet"
__author__ = "Aidan Acquah, Shing Chan, Aiden Doherty"
__maintainer__ = "Shing Chan"
__maintainer_email__ = "shing.chan@ndph.ox.ac.uk"
__license__ = "See LICENSE file."

__classifiers__ = {
    'walmsley': {
        'version': 'ssl-ukb-c24-rw-30s-20251023',
        'md5': '00508bd3217290d9a0339c34303eb29e'
    },
    'willetts': {
        'version': 'ssl-ukb-c24-mw-30s-20251023',
        'md5': 'ec3bf58f2aed2520755aaf8e9ba862fd'
    }
}

from . import _version

__version__ = _version.get_versions()["version"]
