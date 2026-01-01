# defenses/__init__.py
"""
Defense mechanisms for federated learning backdoor attacks
"""

from . import pca_deflect
from . import nab_defense
from . import nab_lga_detection
from . import nab_pseudo_label
from . import nab_train

__all__ = [
    'pca_deflect',
    'nab_defense', 
    'nab_lga_detection',
    'nab_pseudo_label',
    'nab_train'
]

print("[DEFENSES] Defense package initialized")