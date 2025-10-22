"""
File: dataset_info.py
Author: Zih-Syuan Lin (2025)
"""
import os
from typing import Final

dataset_name:Final[str] = 's3'
piece_names:Final[list[str]] = [
    'mo1', 'mo2', 'mo3', 'mo4',  # mozart op 41, k551
    'be1', 'be2', 'be3', 'be4',  # beethoven op 125
    'dv1', 'dv2', 'dv3', 'dv4',  # dvorak op 95, no 9
    'tc1', 'tc2', 'tc3', 'tc4'   # tchaikovsky op 74, no 6
]
int_to_string:Final[dict[int,str]] = {
    k:v for k,v in enumerate(piece_names)
}
string_to_int:Final[dict[str,int]] = {v:k for k,v in int_to_string.items()}
meta_csv_path:Final[str] = './s3_dataset/metadata.csv'
file_path = {
    'original_path': os.path.join(".", "dataset_s3"),
    'converted_path': os.path.join('.', 'converted_dataset', dataset_name),
    'metadata_csv_path': os.path.join(
        '.', 'converted_dataset', dataset_name, 'metadata.csv'),
}
