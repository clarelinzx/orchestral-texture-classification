"""
File: dataset_info.py
Author: Zih-Syuan Lin (2025)
"""
import os
from typing import Final

dataset_name:Final[str] = 'orchestration'

piece_names_dct = {
    # for each composer: [[original file names],[short file names]]
    'beethoven': [[
        'beethoven-symph9-op125-mvt1', 'beethoven-symph1-op21-mvt1', 'beethoven-symph2-op36-mvt1', 
        'beethoven-symph3-op55-mvt1', 'beethoven-symph4-op60-mvt1', 'beethoven-symph5-op67-mvt1', 
        'beethoven-symph6-op68-mvt1', 'beethoven-symph7-op92-mvt1', 'beethoven-symph8-op93-mvt1'],
        [
        'beethoven_op125-1', 'beethoven_op21-1', 'beethoven_op36-1',
        'beethoven_op55-1', 'beethoven_op60-1', 'beethoven_op67-1',
        'beethoven_op68-1', 'beethoven_op92-1', 'beethoven_op93-1',]
    ],
    'haydn': [[
        'haydn-symph099-mvt1', 'haydn-symph100-mvt1', 'haydn-symph101-mvt1',
        'haydn-symph103-mvt1', 'haydn-symph104-mvt1', # 'haydn-symph102-mvt1', 
        ],
        ['hob099-1', 'hob100-1', 'hob101-1', 'hob103-1', 'hob104-1',],
    ],
    'mozart': [[
        'mozart-symph38-k504-mvt1', #'mozart-symph35-k385-mvt1', 'mozart-symph34-k338-mvt1',
        'mozart-symph39-k543-mvt1', 'mozart-symph40-k550-mvt1', #'mozart-symph33-k319-mvt1', 'mozart-symph36-k425-mvt1', 'mozart-symph32-k318-mvt1', 
        'mozart-symph41-k551-mvt1'], 
        [
        'k504-1', 'k543-1', 'k550-1', 'k551-1',]
    ]
}
int_to_string:Final[dict[int,str]] = {
    0: 'k504-1',
    1: 'k543-1',
    2: 'k550-1',
    3: 'k551-1',
    4: 'hob099-1',
    5: 'hob100-1',
    6: 'hob101-1',
    7: 'hob103-1',
    8: 'hob104-1',
    9: 'beethoven_op21-1',
    10: 'beethoven_op36-1',
    11: 'beethoven_op55-1',
    12: 'beethoven_op60-1',
    13: 'beethoven_op67-1',
    14: 'beethoven_op68-1',
    15: 'beethoven_op92-1',
    16: 'beethoven_op93-1',
    17: 'beethoven_op125-1',
}
string_to_int:Final[dict[str,int]] = {v:k for k,v in int_to_string.items()}
piece_names:Final[list[str]] = [int_to_string[_] for _ in range(18)]
meta_csv_path:Final[str] = './orchestration_dataset/metadata.csv'
file_path = {
    'original_path': os.path.join(".", "dataset"),
    'converted_path': os.path.join('.', 'converted_dataset', dataset_name),
    'metadata_csv_path': os.path.join(
        '.', 'converted_dataset', dataset_name, 'metadata.csv'),
}
