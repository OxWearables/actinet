name = "actinet"
__author__ = "Aidan Acquah, Shing Chan, Aiden Doherty"
__maintainer__ = "Shing Chan"
__maintainer_email__ = "shing.chan@ndph.ox.ac.uk"
__license__ = "See LICENSE file."

__classifiers__ = {
    "walmsley": {
        "version": "ssl-ukb-c24-rw-30s-20260128",
        "md5": "a829041ba84b18084b4b2897fb1b36a6",
    },
    "willetts": {
        "version": "ssl-ukb-c24-mw-30s-20260128",
        "md5": "59a9492b50ee922c7a31f88888c5f14b",
    },
}

from . import _version

__version__ = _version.get_versions()["version"]
