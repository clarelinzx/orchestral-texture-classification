"""
File: data_preprocessing.py
Author: Zih-Syuan Lin (2025)
"""

from typing import Final, Tuple

# Time-signature–related columns (per bar or segment)
TS_COLS: Final[Tuple[str, ...]] = (
    "onset", "measure_length", "numerator", "denominator",
)

# Note-level columns (per note event, tied to a track-bar unit)
NOTE_COLS: Final[Tuple[str, ...]] = (
    "onset", "offset", "duration", "time_signature",
    "measure", "measure_offset",
    "onset_time_step", "offset_time_step",
    "midi_number", "pitch_name", "velocity","inst_name",
)

# Piece-level metadata to record once per movement/piece
METADATA_COLS: Final[Tuple[str, ...]] = (
    "piece_name", "total_bar","total_beat","track_names",
)

# Piano-roll / quantization resolution (ticks per quarter note)
RESOLUTION: Final[int] = 24

# The default volume for note events
VOLUME: Final[int] = 96

# Related to time signatures
# YC_NQBEAT_DCT, conceptual beats used in meter perception 
# (e.g. 6/8 perceived as 2 beats, not 3).
YC_NQBEAT_DCT:Final[str:int] = {
    '4/4': 4,
    '3/4': 3,
    '2/2': 2,
    '2/4': 2,
    '6/8': 2,
    '3/2': 3,
    '6/4': 6,
    '4/8': 4,
    '5/4': 5,
    '12/8': 4,
}
# TS_TO_TIME_STEP: Maps a time signature string to structural timing parameters.
# Each entry specifies:
#   - quarters_per_bar: number of quarter notes per bar
#   - ticks_per_quarter: ticks per quarter (based on resolution)
#   - beats_per_bar: denoted in YC_NQBEAT_DCT
TS_TO_TIME_STEP: Final[dict[str,tuple[int,float,int]]] = {
    # time_signature: (quarters_per_bar, ticks_per_quarter, beats_per_bar)
    '4/4': (4, RESOLUTION/4, YC_NQBEAT_DCT['4/4']),
    '3/4': (3, RESOLUTION/3, YC_NQBEAT_DCT['3/4']),
    '2/2': (4, RESOLUTION/4, YC_NQBEAT_DCT['2/2']),  # cut time: 2 beats per bar
    '2/4': (2, RESOLUTION/2, YC_NQBEAT_DCT['2/4']),
    '6/8': (3, RESOLUTION/3, YC_NQBEAT_DCT['6/8']),  # 6/8 ≈ 3 quarters per bar (two dotted quarters); perceived two-beat meter
    '3/2': (6, RESOLUTION/6, YC_NQBEAT_DCT['3/2']),
    '6/4': (6, RESOLUTION/6, YC_NQBEAT_DCT['6/4']), 
    '4/8': (4, RESOLUTION/4, YC_NQBEAT_DCT['4/8']),
    '5/4': (5, RESOLUTION/5, YC_NQBEAT_DCT['5/4']),   
    '12/8': (6, RESOLUTION/6, YC_NQBEAT_DCT['12/8']),
}
# TS_SPLIT: if a quater note equals to 24 ticks, mapping of time signatures
#.          to possible note-division units (in ticks) for hierarchical splitting
TS_SPLIT :Final[dict[str,list[int]]]= {
    '4/4': [96, 48, 24, 12, 6, 3],
    '3/4': [72, 32, 16,  8, 4, 2],
    '2/2': [48, 24, 12, 6, 3],
    '2/4': [48, 24, 12, 6, 3],
    '6/8': [48, 16,  8, 4, 2],
    '3/2': [72, 32, 16,  8, 4, 2],
    '6/4': [144, 48, 16,  8, 4, 2],
    '4/8': [96, 48, 24, 12, 6, 3],
    '5/4': [120, 24, 12, 6, 3],
    '12/8': [144, 72, 36, 12, 6, 3]
}
