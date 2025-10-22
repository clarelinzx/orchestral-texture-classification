"""
File: annotations.py
Description:
    Provide standardized mappings of annotations, the class-id,
    vector-style (multi-label), and its meaning.
Author: Zih-Syuan Lin (2025)
"""
from typing import Final

LABEL_DCT:Final[dict[int, dict]] = {
    0: {'label': [0, 0, 0], 'meaning': 'None'},
    1: {'label': [1, 0, 0], 'meaning': 'Melody'},
    2: {'label': [0, 1, 0], 'meaning': 'Rhythm'},
    3: {'label': [1, 1, 0], 'meaning': 'Melody+Rhythm'},
    4: {'label': [0, 0, 1], 'meaning': 'Harmony'},
    5: {'label': [1, 0, 1], 'meaning': 'Melody+Harmony'},
    6: {'label': [0, 1, 1], 'meaning': 'Rhythm+Harmony'},
    7: {'label': [1, 1, 1], 'meaning': 'All'}
}

LABEL_LST:Final[list[str]] = [LABEL_DCT[_]['meaning'] for _ in range(8)]
LABEL_DCT_INV:Final[dict[str,int]] = {
    'none': 0,
    'mel': 1,
    'rhythm': 2,
    'mel+rhythm': 3,
    'harm': 4,
    'mel+harm': 5,
    'rhythm+harm': 6,
    'mel+rhythm+harm': 7
}