name = "actinet"
__author__ = "Aidan Acquah, Shing Chan, Aiden Doherty"
__maintainer__ = "Shing Chan"
__maintainer_email__ = "shing.chan@ndph.ox.ac.uk"
__license__ = "See LICENSE file."

__classifiers__ = {
    'walmsley': {
        'version': 'ssl-ukb-c24-rw-30s-20250815',
        'md5': '148a5e798f2465985fe97a3f8b42ba96'
    },
    'willetts': {
        'version': 'ssl-ukb-c24-mw-30s-20251001',
        'md5': 'da6c896b2168ef7d1cbe99932c4ff0b3'
    }
}

from . import _version

__version__ = _version.get_versions()["version"]
