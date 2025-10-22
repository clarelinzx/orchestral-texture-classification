"""
File: data_cleaning.py
Purpose:
    Demonstrate how we clean and extract essential information from both datasets.

    For each piece (here, a piece means a movement, rather than an entire symphony):
    - track_order.txt: recording the instrument names (track names) order in the original score
    - beat.csv
    - downbeat.csv
    - time_signature.csv
    - measure_se.csv
    - pnoroll.npy: shape=(number of track, time span in ticks, 128)
    - replaymtx.npy: shape is same as pnoroll

    When taking "rest" as a "silent note" with "midi number" is 128, we have:
    - rest_pnoroll.npy: shape=(number of track, time span in ticks, 129)
    - rest_replaymtx.npy: shape is same as rest_pnoroll

    For S3 dataset, note events have already prepared in csv format. Therefore,
    only image-like array required to be created. For Orchestration dataset, we 
    need to extract all information from musicXML.

Author: Zih-Syuan Lin (2025)
"""

# %% [markdown]
# ## Imports
import os
import sys
import muspy
import numpy as np
import pypianoroll
import pandas as pd
from typing import Final, Tuple
from ..settings.annotations import LABEL_DCT, LABEL_LST
from ..settings.instruments import INSTRUMENT_NAME_VARIATION, INSTRUMENT_NAME, INSTRUMENT_LOOKUP_TABLE
from ..settings.data_preprocessing import TS_COLS, NOTE_COLS, METADATA_COLS, RESOLUTION, VOLUME, TS_TO_TIME_STEP, YC_NQBEAT_DCT, TS_SPLIT

# %% [markdown]
# ## Functions

def parse_time_signature_for_beat(time_signature_for_beat):
    time_signature_dct = {}
    raw_segs = time_signature_for_beat.split(',')
    for seg in raw_segs:
        interval, ts = seg.split('=')
        beg, end = interval.split(':')
        ts_nuemerator, ts_denominator = ts.split(':')
        beg, end = int(beg), int(end)
        for measure_number in range(beg, end+1, 1):
            time_signature_dct.update({measure_number: f'{ts_nuemerator}/{ts_denominator}'})
    return time_signature_dct

def parse_time_signature(time_signature:str):
        time_signature_list = []
        raw_segs = time_signature.split(',')
        for seg in raw_segs:
            n_quarter_beat = int(seg.split('=')[-1])
            interval = seg.split('=')[0]
            beg = int(interval.split(':')[0])
            end = int(interval.split(':')[1])
            time_signature_list.append([beg, end, n_quarter_beat])
        return time_signature_list

def get_rest_interval(rest_index):
    rest_interval = []
    rest_onset = rest_index[0]
    rest_offset = rest_index[0]+1
    for rdx in rest_index[1:]:
        if rdx == rest_offset:
            rest_offset = rdx+1
        else:
            rest_interval.append((rest_onset, rest_offset))
            rest_onset = rdx
            rest_offset = rdx + 1
    rest_interval.append((rest_onset, rest_offset))
    return rest_interval

def hierarchical_split_interval_cross_bar(bar_start, start, end, units=[48, 24, 12, 6, 3]):
    result = []
    pos = start
    while pos < end and (end-pos)>units[-1]:
        for unit in units:
            if pos % unit == 0 and (pos + unit) <= end:
                result.append((bar_start + pos, bar_start + pos + unit))
                pos += unit
                break
        else:
            # If no further split units are found, force splitting 
            # to the lowest level
            result.append((bar_start+pos, bar_start+end))
            break
    return result

def hierarchical_split_cross_meter(onset, offset, measure_se_df, ts_split):
    results = []
    mn_onset = max(np.searchsorted(measure_se_df['time_step'].to_numpy(), onset, side='right') - 1, 0)
    mn_offset = np.searchsorted(measure_se_df['time_step'].to_numpy(), offset, side='right') - 1
    bar_start = measure_se_df.loc[mn_onset, 'time_step']
    bar_end = measure_se_df.loc[mn_onset, 'time_step'] + measure_se_df.loc[mn_onset, 'number_of_quarter_note'] * measure_se_df.loc[mn_offset, 'quarter_length']
        
    if mn_onset==mn_offset:
        local_start = onset - bar_start
        local_end = offset - bar_start
        ts = measure_se_df.loc[mn_onset, 'time_signature']
        split_units = ts_split[ts]
        results.extend(hierarchical_split_interval_cross_bar(
                bar_start, local_start, local_end, split_units))
    else:
        local_start = onset - bar_start
        local_end = bar_end - bar_start
        ts = measure_se_df.loc[mn_onset, 'time_signature']
        split_units = ts_split[ts]
        results.extend(
            hierarchical_split_interval_cross_bar(
                bar_start, local_start, local_end, split_units
            ))
        # middle
        for mn in range(mn_onset+1, mn_offset):
            results.extend(
                [(measure_se_df.loc[mn, 'time_step'],
                measure_se_df.loc[mn+1, 'time_step']
            )])
        # end
        bar_start = measure_se_df.loc[mn_offset, 'time_step']
        if mn_offset+1==measure_se_df.shape[0]:
            bar_end = measure_se_df.loc[mn_offset, 'time_step'] + measure_se_df.loc[mn_offset, 'number_of_quarter_note'] * measure_se_df.loc[mn_offset, 'quarter_length']
        else:
            bar_end = measure_se_df.loc[mn_offset+1, 'time_step']
        local_start = 0
        local_end = offset - bar_start
        ts = measure_se_df.loc[mn_offset, 'time_signature']
        split_units = ts_split[ts]
        results.extend(
            hierarchical_split_interval_cross_bar(
                bar_start, local_start, local_end, split_units
        ))
    return results

def forward_fill(arr):
    arr = arr.copy()
    for i in range(arr.shape[1]):  # by column
        col = arr[:, i]
        mask = col == 0
        # find all positions of non-zero value
        non_zero_idx = np.where(~mask)[0]
        if len(non_zero_idx) == 0:
            continue  # skip all zero
        # fillin initial value
        col[:non_zero_idx[0]] = col[non_zero_idx[0]]
        for j in range(1, len(non_zero_idx)):
            start = non_zero_idx[j-1]
            end = non_zero_idx[j]
            col[start+1:end] = col[start]  # use the closest non-zero value to fillin the middle part
        col[non_zero_idx[-1]+1:] = col[non_zero_idx[-1]]  # fill the last part
        arr[:, i] = col
    return arr

def resolve_bar_roles(component_dict, label_dict, total_bar):
    # two representation ways of the orchestral texture annotations: 
    # multi-label(vector-like) to multi-class(index)
    label_vec_to_index = {tuple(v["label"]): k for k, v in label_dict.items()}

    # multi-label
    base_roles = {
        'mel': [1, 0, 0],
        'rhythm': [0, 1, 0],
        'harm': [0, 0, 1],
    }

    result = []
    for bdx in range(total_bar):
        components = component_dict[bdx]
        if components:
            total_weight = sum(w for _, w in components)

            def parse_component(comp_str):
                vec = [0, 0, 0]
                parts = comp_str.lower().split("+")
                for p in parts:
                    if p in base_roles:
                        vec = [x or y for x, y in zip(vec, base_roles[p])]
                return vec

            if total_weight > 1:
                merged = [0, 0, 0]
                for role_str, _ in components:
                    vec = parse_component(role_str)
                    merged = [x or y for x, y in zip(merged, vec)]
            else:
                # find the largest and only use its amount
                top_role, _ = max(components, key=lambda x: x[1])
                merged = parse_component(top_role)

            # convert into label_dict index
            label_index = label_vec_to_index[tuple(merged)]
            result.append(merged)
        else:
            result.append([0, 0, 0])

    return result


# %% [markdown]
# ## S3 dataset
if __name__=='__main__':
    print("="*20, "/n", "Cleaning s3 dataset\n", "="*20)
    from ..settings.s3_info import dataset_name, piece_names, int_to_string, string_to_int, meta_csv_path, file_path

    metadata = []
    yc_metadata = []
    for piece_name in piece_names[:]:
        print(f'converting {piece_name}...')
        # mkdir    
        os.makedirs(
            os.path.join(
                file_path['converted_path'], piece_name, 'temporal_attributes'
            ), exist_ok=True)
        texture_dir = os.path.join(
                file_path['converted_path'], piece_name, 'role'
            )
        os.makedirs(texture_dir, exist_ok=True)
        os.makedirs(os.path.join(
                file_path['converted_path'], piece_name, 'from_notecsv',
            ), exist_ok=True)
        os.makedirs(
                os.path.join(
                    file_path['converted_path'], piece_name, 'note'
                    ), exist_ok=True
                )
        # track_names
        track_names = [i[:-4] for i in os.listdir(
            os.path.join(file_path['original_path'], piece_name, 'annotations', 'note')
            ) if i.endswith('.csv')]
        with open(os.path.join(texture_dir, 'track_order.txt'), 'w') as f:
                f.write(';'.join(track_names))
        # downbeat
        downbeat_df = pd.read_csv(
            os.path.join(
                file_path['original_path'], piece_name, 'annotations', 'temporal_attributes', 'downbeat.csv'
        ))
        downbeat_df.to_csv(
                os.path.join(
                file_path['converted_path'], piece_name, 'temporal_attributes',
                'downbeat.csv'
        ))
        downbeats = downbeat_df.to_numpy().flatten().tolist()
        ts_df = pd.read_csv(
            os.path.join(
                file_path['original_path'], piece_name, 'annotations', 'temporal_attributes', 'time_signature.csv'
        ))
        ts_df.to_csv(
                os.path.join(
                file_path['converted_path'], piece_name, 'temporal_attributes',
                'time_signature.csv'
        ))
        # total_bar
        bar_num = total_bar = downbeat_df.shape[0]
    
        n_tracks = len(track_names)
        inst_list = track_names
        inst_list_pretty_name = inst_list.copy()
        inst_list_pure_name = [INSTRUMENT_NAME[INSTRUMENT_LOOKUP_TABLE[inst]] for inst in inst_list_pretty_name]

        # total beat
        total_beat = float(max(downbeat_df['onset']))
        for track in track_names:
            note_df = pd.read_csv(
                os.path.join(
                    file_path['original_path'], piece_name, 'annotations', 'note', track+'.csv'
                ))
            total_beat = max(total_beat, float(note_df['offset'].max()))
        
        ts_measure_length = ts_df.to_numpy()[-1,2]
        if total_beat - downbeats[-1] != ts_measure_length:
            total_beat = downbeats[-1] + ts_measure_length
        
        # measure_se
        measure_se = list(zip(downbeats, downbeats[1:]+[total_beat]))
        assert len(measure_se) == total_bar, f' [{total_bar=}] != [{len(measure_se)=}] '
        # beat
        beats = []
        ts_lst = ts_df.to_numpy().tolist()
        for mn_onset, mn_offset in measure_se:
            if ts_lst:
                if mn_onset >= ts_lst[0][0]:
                    ts_onset,_, ts_measure_length, ts_numerator, ts_denominator = ts_lst.pop(0)
                    beat_length = ts_measure_length / ts_numerator
            beats.append(np.arange(mn_onset, mn_offset, beat_length))
        beats = np.concatenate(beats).tolist()
        if beats[-1] - downbeats[-1] != ts_measure_length:
            beats += np.arange(
                            int(beats[-1]), 
                            int(downbeats[-1] + ts_measure_length + beat_length),
                            beat_length
                        ).tolist()
        beat_df = pd.DataFrame(beats, columns=['onset'])
        beat_df = beat_df.set_index('onset')
        beat_df.to_csv(
                os.path.join(
                file_path['converted_path'], piece_name, 'temporal_attributes',
                'beat.csv'
        ))
        # measure_se
        measure_se_df = pd.DataFrame(measure_se, columns=['onset', 'offset'])
        measure_time_step_lst = []
        measure_ts_lst = []
        measure_ts_nqbeat_yc_lst = []
        measure_ts_nqbeat_lst = []
        measure_ts_quarter_length_lst = []

        mn_lst = ts_df['measure'].tolist() + [total_bar+1]
        mn_lst = list(zip(mn_lst[:-1], mn_lst[1:]))
        mn_lst = [(a,b-1) for a,b in mn_lst]
        time_sign_yc = []
        time_sign_beat_yc = {}
        for idx,(onset, measure, ql, tsn, tsd) in enumerate(ts_df.to_numpy()):    
            time_sign_yc.append([int(measure), int(mn_lst[idx][1]), YC_NQBEAT_DCT[f'{int(tsn)}/{int(tsd)}'], f'{int(tsn)}/{int(tsd)}'])
            for mn in range(int(measure), int(mn_lst[idx][1])+1):
                time_sign_beat_yc[mn] = f'{int(tsn)}/{int(tsd)}'
        print(time_sign_yc)
        for mn in range(1, bar_num+1):
            steps = 0
            for (beg, end, n_q_beat, _) in time_sign_yc:
                if mn > end: # contain the whole segment
                    steps += (end-beg+1)*n_q_beat*24
                elif mn > beg: # break inside the segment
                    steps += (mn-beg)*n_q_beat*24
                else:
                    pass
            measure_time_step_lst.append(steps)
            measure_ts_lst.append(time_sign_beat_yc[mn])
            measure_ts_nqbeat_yc_lst.append(TS_TO_TIME_STEP[time_sign_beat_yc[mn]][2])
            measure_ts_nqbeat_lst.append(TS_TO_TIME_STEP[time_sign_beat_yc[mn]][0])
            measure_ts_quarter_length_lst.append(
                TS_TO_TIME_STEP[time_sign_beat_yc[mn]][2]*RESOLUTION/(TS_TO_TIME_STEP[time_sign_beat_yc[mn]][0])
                )
        measure_se_df['time_step'] = measure_time_step_lst
        measure_se_df['time_signature'] = measure_ts_lst
        measure_se_df['number_of_grid'] = measure_ts_nqbeat_yc_lst
        measure_se_df['number_of_quarter_note'] = measure_ts_nqbeat_lst
        measure_se_df['quarter_length'] = measure_ts_quarter_length_lst
        fig_time_step = measure_se_df.loc[bar_num-1,'time_step'] + measure_se_df.loc[bar_num-1, 'number_of_quarter_note'] * measure_se_df.loc[bar_num-1, 'quarter_length']
        fig_time_step = int(fig_time_step)
        measure_se_df.to_csv(
            os.path.join(
                file_path['converted_path'], piece_name, 'temporal_attributes',
                'measure_se.csv'
            ))
        # Label
        piece_label = np.zeros((total_bar, len(track_names), 3))
        for tdx,track in enumerate(track_names):
            # if track != 'violin2':
            #     continue
            # print(f"{track=}")
            texture_df = pd.read_csv(os.path.join(
                file_path['original_path'], piece_name, 'annotations', 'orchestral_texture', track+'.csv'
            ))
            first_label_array = {bdx: [] for bdx in range(total_bar)}

            for onset, offset, role in texture_df.to_numpy()[:]:
                mn_s = np.searchsorted(downbeats, onset, side='right') - 1
                mn_e = np.searchsorted(downbeats, offset, side='right') - 1
                portion_s = (measure_se[mn_s][1] - onset) / (measure_se[mn_s][1] - measure_se[mn_s][0])
                portion_e = (offset - measure_se[mn_e][0]) / (measure_se[mn_e][1] - measure_se[mn_e][0])
                if mn_s == mn_e:
                    portion = (offset - onset) / (measure_se[mn_s][1] - measure_se[mn_s][0])
                    first_label_array[mn_s].append([role, portion])
                elif mn_e==mn_s+1:
                    first_label_array[mn_s].append([role, portion_s])
                    first_label_array[mn_e].append([role, portion_e])
                else:
                    first_label_array[mn_s].append([role, portion_s])
                    for i in range(mn_s+1, mn_e):
                        first_label_array[i].append([role, 1])
                    first_label_array[mn_e].append([role, portion_e])
            second_label_array = resolve_bar_roles(first_label_array, LABEL_DCT, total_bar)
            piece_label[:, tdx, :] = np.array(second_label_array)
            # print(f"{first_label_array=}")
            # print(f"{second_label_array=}")
        #     break
        # break
        with open(os.path.join(texture_dir, 'role.npy'), 'wb') as f:
            np.save(f, piece_label)
        
        label_list = piece_label.copy()
        bar_num, track_num, _ = label_list.shape

        # note
        os.makedirs(
            os.path.join(
                file_path['converted_path'], piece_name, 'note'
                ), exist_ok=True
            )
        os.makedirs(
            os.path.join(
                file_path['converted_path'], piece_name, 'from_notecsv', 
                ), exist_ok=True
            )
        # Pnoroll
        pnoroll_from_csv, replaymtx_from_csv = [], []
        rest_from_pnoroll, rest_replaymtx_from_pnoroll = [], []
        fig_size = (fig_time_step, 128)
        for track in track_names:
            print(track, end='\r')
            note_df = pd.read_csv(
                    os.path.join(
                        file_path['original_path'], piece_name, 'annotations', 'note', track+'.csv'
                    ))
            a_pnoroll = np.zeros(fig_size)
            a_replaymtx = np.zeros(fig_size)
            new_note_list = []            
            for row in note_df.to_numpy():
                onset, offset, midi, pitch_name, dur, staff, measure, beat_in_measure, inst = row            
                mn = np.searchsorted(downbeats, onset, side='right') - 1
                mn_offset = np.searchsorted(downbeats, offset, side='right') - 1
                ts = time_sign_beat_yc[mn+1]
                onset_time_step = measure_se_df.loc[mn,'time_step'] + \
                    (onset-measure_se_df.loc[mn, 'onset'])*measure_se_df.loc[mn,'quarter_length']
                offset_time_step = measure_se_df.loc[mn_offset,'time_step'] + \
                    (offset-measure_se_df.loc[mn_offset, 'onset'])*measure_se_df.loc[mn_offset,'quarter_length']

                new_note_list.append([
                    onset, offset, dur, ts, mn, mn_offset, onset_time_step, offset_time_step,
                    midi, pitch_name, VOLUME, track
                ])

                a_pnoroll[int(onset_time_step):int(offset_time_step), int(midi)] = VOLUME
                a_replaymtx[int(onset_time_step), int(midi)] = 1

            # Treat "rest" like silent notes. First, find the intervals:
            rest_interval = get_rest_interval(np.where(a_pnoroll.sum(axis=1)==0)[0])
            # whether onset & offset require further spliting
            new_rest_interval = []
            for onset, offset in rest_interval:
                new_rest_interval += hierarchical_split_cross_meter(
                    onset, offset, measure_se_df, TS_SPLIT)
            a_rest = np.zeros((fig_time_step, 1))
            a_rest_replaymtx = np.zeros((fig_time_step, 1))
            for (r_onset, r_offset) in new_rest_interval:
                a_rest[int(r_onset):int(r_offset), 0] = 96
                a_rest_replaymtx[int(r_onset), 0] = 1
            rest_from_pnoroll.append(a_rest)
            rest_replaymtx_from_pnoroll.append(a_rest_replaymtx)
            
            new_pnoroll = np.concatenate([a_pnoroll, a_rest], axis=1)
            new_note_df = pd.DataFrame(new_note_list, columns=NOTE_COLS)

            pnoroll_from_csv.append(a_pnoroll)
            replaymtx_from_csv.append(a_replaymtx)

            new_note_df.set_index('onset').to_csv(os.path.join(
                    file_path['converted_path'], piece_name, 'note',
                    track+'.csv'
                    ))
        print()
        with open(os.path.join(
                file_path['converted_path'], piece_name, 'from_notecsv',
                'pnoroll.npy'
                ), 'wb') as f:
            np.save(f, pnoroll_from_csv)
        with open(os.path.join(
                file_path['converted_path'], piece_name, 'from_notecsv',
                'replaymtx.npy'
                ), 'wb') as f:
            np.save(f, replaymtx_from_csv)    
        with open(os.path.join(
                file_path['converted_path'], piece_name, 'from_notecsv',
                    'rest_pnoroll.npy'
                ), 'wb') as f:
            np.save(f, rest_from_pnoroll)
        with open(os.path.join(
                file_path['converted_path'], piece_name, 'from_notecsv',
                    'rest_replaymtx.npy'
                ), 'wb') as f:
            np.save(f, rest_replaymtx_from_pnoroll)

        yc_metadata.append([
            piece_name, 
            total_bar,
            os.path.join(file_path['converted_path'], piece_name, 'from_notecsv','pnoroll.npy'), 
            os.path.join(file_path['converted_path'], piece_name, 'role', 'role.npy'),
            ';'.join(track_names)
    ])
        metadata.append(
            [piece_name, total_bar, total_beat, ';'.join(track_names)]
        )
        # break
    metadata_df = pd.DataFrame(metadata, columns=METADATA_COLS)
    metadata_df.to_csv(file_path['metadata_csv_path'])
    print(f"{dataset_name}: the first piece is {piece_names[0]}, its piano roll shape is {pnoroll_from_csv[0].shape}")

# %%[markdown]
# ## Orchestration dataset
if __name__=="__main__":
    print("="*20, "/n", "Cleaning orchestration dataset\n", "="*20)
    from ..settings.orchestration_info import dataset_name, piece_names_dct, piece_names, int_to_string, string_to_int, meta_csv_path, file_path
    metadata = []
    meta_df = pd.read_csv(meta_csv_path)
    for composer,piece_names in piece_names_dct.items():
        for piece_name_old, piece_name_new in list(zip(piece_names[0], piece_names[1])):
            # if piece_name_new != 'k543-1':
            #     continue
            print(f"{piece_name_new=}, {meta_df.iloc[string_to_int[piece_name_new]]['annot_npy']}")
            label_list = np.load(meta_df.iloc[string_to_int[piece_name_new]]['annot_npy'])
            multitrack = pypianoroll.load(meta_df.iloc[string_to_int[piece_name_new]]['pianoroll'])
            
            n_tracks = len(multitrack.tracks)
            inst_list = [track.name for track in multitrack.tracks]
            inst_list_pretty_name = [INSTRUMENT_NAME_VARIATION[inst] for inst in inst_list]
            inst_list_pure_name = [INSTRUMENT_NAME[INSTRUMENT_LOOKUP_TABLE[inst]] for inst in inst_list_pretty_name]
            bar_num, track_num, _ = label_list.shape
            fig_time_step, pitch_cat_len = multitrack[0].pianoroll.shape

            # Extract info from xml
            xml = muspy.read_musicxml(
                os.path.join(
                    file_path['original_path'], 'scores_xml', 
                    composer, piece_name_new+'.musicxml'
                ))

            # Time signature
            time_sign_yc = parse_time_signature(
                meta_df.iloc[string_to_int[piece_name_new]]['time_signature'])
            time_sign_beat_yc =parse_time_signature_for_beat(
                meta_df.iloc[string_to_int[piece_name_new]]['time_signature_beat'])
            # Time signature
            time_signatures = []
            for ts in xml.time_signatures:
                ts_measure_length = ts.numerator / ts.denominator * 4
                beat_length = ts_measure_length / ts.numerator
                time_signatures.append([
                    ts.time / xml.resolution, ts_measure_length, ts.numerator, ts.denominator
                ])
            ts_df = pd.DataFrame(time_signatures, columns=TS_COLS)
            ts_df = ts_df.set_index('onset')
            os.makedirs(
                os.path.join(
                    file_path['converted_path'], piece_name_new, 'temporal_attributes'
                    ), exist_ok=True
                )
            ts_df.to_csv(
                os.path.join(
                    file_path['converted_path'], piece_name_new, 'temporal_attributes',
                    'time_signatures.csv'
                ))
            # downbeat
            downbeats = []
            for barline in xml.barlines:
                downbeats.append(barline.time / xml.resolution)
            total_bar = len(downbeats)
            assert total_bar == bar_num, f"{total_bar=} in xml != {bar_num=} from yc's label array"
            downbeat_df = pd.DataFrame(downbeats, columns=['onset'])
            downbeat_df = downbeat_df.set_index('onset')
            # beat
            beats = []
            for beat in xml.beats:
                beats.append(beat.time / xml.resolution)
            if beats[-1] - downbeats[-1] != ts_measure_length:
                beats += np.arange(
                                int(beats[-1]),
                                int(downbeats[-1] + ts_measure_length + beat_length),
                                beat_length
                            ).tolist()
            beat_df = pd.DataFrame(beats, columns=['onset'])
            beat_df = beat_df.set_index('onset')
            # measure_se
            measure_se = list(zip(downbeats, downbeats[1:]+[beats[-1]]))
            measure_se_df = pd.DataFrame(measure_se, columns=['onset', 'offset'])
            measure_time_step_lst = []
            measure_ts_lst = []
            measure_ts_nqbeat_yc_lst = []
            measure_ts_nqbeat_lst = []
            measure_ts_quarter_length_lst = []
            for mn in range(1, bar_num+1):
                steps = 0
                for (beg, end, n_q_beat) in time_sign_yc:
                    if mn > end: # contain the whole segment
                        steps += (end-beg+1)*n_q_beat*24
                    elif mn > beg: # break inside the segment
                        steps += (mn-beg)*n_q_beat*24
                    else:
                        pass
                measure_time_step_lst.append(steps)
                measure_ts_lst.append(time_sign_beat_yc[mn])
                measure_ts_nqbeat_yc_lst.append(TS_TO_TIME_STEP[time_sign_beat_yc[mn]][2])
                measure_ts_nqbeat_lst.append(TS_TO_TIME_STEP[time_sign_beat_yc[mn]][0])
                measure_ts_quarter_length_lst.append(
                    TS_TO_TIME_STEP[time_sign_beat_yc[mn]][2]*24/(TS_TO_TIME_STEP[time_sign_beat_yc[mn]][0])
                    )
            measure_se_df['time_step'] = measure_time_step_lst
            measure_se_df['time_signature'] = measure_ts_lst
            measure_se_df['number_of_grid'] = measure_ts_nqbeat_yc_lst
            measure_se_df['number_of_quarter_note'] = measure_ts_nqbeat_lst
            measure_se_df['quarter_length'] = measure_ts_quarter_length_lst
            measure_se_df.to_csv(
                os.path.join(
                    file_path['converted_path'], piece_name_new, 'temporal_attributes',
                    'measure_se.csv'
                ))
            # sys.exit()
            # note
            os.makedirs(
                os.path.join(
                    file_path['converted_path'], piece_name_new, 'note'
                    ), exist_ok=True
                )
            os.makedirs(
                os.path.join(
                    file_path['converted_path'], piece_name_new, 'from_notecsv', 
                    ), exist_ok=True
                )
            
            for tdx,track in enumerate(xml.tracks):
                if piece_name_new=='k551-1':
                    tdx_k551_1 = {
                        0:0, 1:1, 2:1, 3:2, 4:2, 5:3, 6:3, 7:4, 8:4, 9:5, 10:6, 11:7, 12:8, 13:9, 14:10, 15:11
                    }
                    inst_name = inst_list_pretty_name[tdx_k551_1[tdx]]
                else:
                    inst_name = inst_list_pretty_name[tdx]
                print(f'\t[{tdx}] processing {track.name} with new name {inst_name}' + ' '*30, end='\r')
                if piece_name_new=='k551-1' and tdx==[2,4,6,8]:
                    pass
                else:
                    note_list = []
                for note in track.notes:
                    onset = note.time / xml.resolution
                    duration = note.duration / xml.resolution
                    offset = onset + duration
                    mn = np.searchsorted(downbeats, onset, side='right') - 1
                    mn_offset = np.searchsorted(downbeats, offset, side='right') - 1
                    ts = time_sign_beat_yc[mn+1]
                    onset_time_step = measure_se_df.loc[mn,'time_step'] + \
                        (onset-measure_se_df.loc[mn, 'onset'])*measure_se_df.loc[mn,'quarter_length']
                    offset_time_step = measure_se_df.loc[mn_offset,'time_step'] + \
                        (offset-measure_se_df.loc[mn_offset, 'onset'])*measure_se_df.loc[mn_offset,'quarter_length']
                    pitch_name = note.pitch_str
                    midi_number = note.pitch
                    velocity = note.velocity
                    note_list.append([
                        onset, offset, duration, ts, mn, mn_offset, onset_time_step, offset_time_step,
                        midi_number, pitch_name, velocity, inst_name
                    ])
                if piece_name_new=='k551-1' and tdx==[1,3,5,7]: pass
                else:
                    note_df = pd.DataFrame(note_list, columns=NOTE_COLS)
                    # note_df = note_df.set_index('onset')
                    note_df.set_index('onset').to_csv(os.path.join(
                        file_path['converted_path'], piece_name_new, 'note',
                        inst_name+'.csv'
                        ))
            fig_size = (fig_time_step, pitch_cat_len)
            pnoroll_from_csv, replaymtx_from_csv = [], []
            rest_from_pnoroll, rest_replaymtx_from_pnoroll = [], []
            for tdx, inst_name in enumerate(inst_list_pretty_name):
                print()
                print(inst_name)
                a_pnoroll = np.zeros(fig_size)
                a_replaymtx = np.zeros(fig_size)
                note_df = pd.read_csv(os.path.join(
                        file_path['converted_path'], piece_name_new, 'note',
                        inst_name+'.csv'
                        ))
                # Pnoroll from note.csv
                for row in note_df.to_numpy():
                    onset, offset, duration, ts, mn, mn_offset, onset_time_step, offset_time_step, midi_number, pitch_name, velocity, inst = row
                    a_pnoroll[int(onset_time_step):int(offset_time_step), int(midi_number)] = velocity
                    a_replaymtx[int(onset_time_step), int(midi_number)] = 1
                pnoroll_from_csv.append(a_pnoroll)
                replaymtx_from_csv.append(a_replaymtx)
                # rest
                rest_interval = get_rest_interval(np.where(a_pnoroll.sum(axis=1)==0)[0])
                new_rest_interval = []
                for onset, offset in rest_interval:
                    new_rest_interval += hierarchical_split_cross_meter(
                        onset, offset, measure_se_df, TS_SPLIT)
                print(new_rest_interval)
                a_rest = np.zeros((fig_time_step, 1))
                a_rest_replaymtx = np.zeros((fig_time_step, 1))
                for (r_onset, r_offset) in new_rest_interval:
                    a_rest[int(r_onset):int(r_offset), 0] = 96
                    a_rest_replaymtx[int(r_onset), 0] = 1
                rest_from_pnoroll.append(a_rest)
                rest_replaymtx_from_pnoroll.append(a_rest_replaymtx)
                
                new_pnoroll = np.concatenate([a_pnoroll, a_rest], axis=1)
                new_replaymtx = np.concatenate([a_replaymtx, a_rest_replaymtx], axis=1)
            with open(os.path.join(
                    file_path['converted_path'], piece_name_new, 'from_notecsv',
                    'pnoroll.npy'
                    ), 'wb') as f:
                np.save(f, pnoroll_from_csv)
            with open(os.path.join(
                    file_path['converted_path'], piece_name_new, 'from_notecsv',
                    'replaymtx.npy'
                    ), 'wb') as f:
                np.save(f, replaymtx_from_csv)
            with open(os.path.join(
                    file_path['converted_path'], piece_name_new, 'from_notecsv',
                    'rest_pnoroll.npy'
                    ), 'wb') as f:
                np.save(f, rest_from_pnoroll)
            with open(os.path.join(
                    file_path['converted_path'], piece_name_new, 'from_notecsv',
                    'rest_replaymtx.npy'
                    ), 'wb') as f:
                np.save(f, rest_replaymtx_from_pnoroll)
            print()
    print(f"{dataset_name}: the first piece is {piece_names[0]}, its piano roll shape is {pnoroll_from_csv[0].shape}")

