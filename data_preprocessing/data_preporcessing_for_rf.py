"""
File: data_preprocessing_for_rf.py
Description:
    Preparing statistical features for Random Forest Classifier, calculating 30+ handcrafted
    statistical features defined in L. Soum-Fontez et al. (2021) (https://hal.science/hal-03322543).
    The features of all parts (different instrument parts/tracks/voices) are all in the same csv file for each piece.
    They are calculated at 2 "levels": (the part-axis, y-axis)
    - bar level: considering all notes' from all tracks
    - track-bar level: considering only notes' in the target track
    And there are 2 "types": (the temporal-axis, x-axis)
    - global: some features are designed to extract the attribute that won't change during the piece going
    - local: these features will change depends on the bar content
Author: Zih-Syuan Lin (2025)
"""

import os
import numpy as np
import pandas as pd

from ..settings.instruments import INSTRUMENT_NAME
from .utils import find_inst_index

# %%[markdown]
# ## Functions
def get_global_statistical_features(concat_all_note_df, measure_se):
    # normalized
    total_notes = concat_all_note_df.shape[0]
    piece_length = concat_all_note_df['offset'].max() - concat_all_note_df['onset'].min()
    # pitch_feature_cols = highest, lowest, mean, std
    global_highest_pitch = concat_all_note_df.midi_number.max() / 128.0
    global_lowest_pitch = concat_all_note_df.midi_number.min() / 128.0
    global_mean_pitch =concat_all_note_df.midi_number.mean() / 128.0
    global_std_pitch = concat_all_note_df.midi_number.std() / 128.0
    # note_dur_feature_cols = ['longest_duration', 'shortest_duration', 'mean_duration', 'std_duration']
    global_longest_duration = concat_all_note_df.duration.max() / piece_length
    global_shortest_duration =concat_all_note_df.duration.min() / piece_length
    global_mean_duration = concat_all_note_df.duration.mean() / piece_length
    global_std_duration = concat_all_note_df.duration.std() / piece_length
    # pitch_interval_feature_cols
    intervals = []
    global_number_of_syncopated_note = 0
    for mn in concat_all_note_df.groupby('measure').groups.keys():
        notes_in_mn_df = concat_all_note_df.groupby('measure').get_group(mn)
        notes_in_mn_df = notes_in_mn_df.sort_values('midi_number')
        intervals += [b-a for a,b in zip(notes_in_mn_df.midi_number.tolist()[:-1],
                                            notes_in_mn_df.midi_number.tolist()[1:])]
        # syncopation_feature_cols = # of syncopated notes
        # 針對每一顆音符，如果他不在拍上 onset，則計算
        ts_d = int(measure_se.loc[mn, 'time_signature'].split('/')[1])
        beat_length = 4/ts_d
        beat_lst = np.arange(
            measure_se.loc[mn, 'onset'], 
            measure_se.loc[mn, 'offset'],
            beat_length).tolist()
        global_number_of_syncopated_note += len(
            [_ for _ in notes_in_mn_df.onset.values if _ not in beat_lst]
        )

    if len(intervals) > 0:
        global_number_of_different_intervals = len(np.unique(intervals)) / len(intervals)
        global_largest_interval = (np.abs(np.unique(intervals))).max() / 128.0
        global_smallest_interval = (np.abs(np.unique(intervals))).min() / 128.0
        global_mean_interval= (np.abs(np.unique(intervals))).mean() / 128.0
        global_std_interval = (np.abs(np.unique(intervals))).std() / 128.0
    else:
        global_number_of_different_intervals, global_largest_interval, global_smallest_interval, global_mean_interval, global_std_interval = -1, -1, -1, -1, -1

    global_number_of_syncopated_note /= total_notes
    # onset synchrony
    onset_synchrony, total_synchrony = get_onset_synchrony(concat_all_note_df, measure_se)

    return {
        'global_highest_pitch': global_highest_pitch,
        'global_lowest_pitch': global_lowest_pitch,
        'global_mean_pitch': global_mean_pitch,
        'global_std_pitch': global_std_pitch,
        'global_longest_duration': global_longest_duration,
        'global_shortest_duration': global_shortest_duration,
        'global_mean_duration': global_mean_duration,
        'global_std_duration': global_std_duration,
        'global_number_of_different_intervals': global_number_of_different_intervals,
        'global_largest_interval': global_largest_interval,
        'global_smallest_interval': global_smallest_interval,
        'global_mean_interval': global_mean_interval,
        'global_std_interval': global_std_interval,
        'global_number_of_syncopated_note': global_number_of_syncopated_note,
        'total_synchrony': total_synchrony,
        'onset_synchrony': onset_synchrony,
    }


# see https://gitlab.com/algomus.fr/comparing-texture/-/blob/main/dist_huron.py?ref_type=heads
# Huron-based features: onset_synchrony
# for more info
def get_onset_synchrony(concat_all_note_df, measure_se):
    total_synchrony = 0
    length = measure_se['offset'].max() - measure_se['onset'].min()
    onset_synchrony = {
        bdx: {
            'onset': float(measure_se.loc[bdx, 'onset']),
            'offset': float(measure_se.loc[bdx, 'offset']),
            'notes': -1,
            'new_notes': -1,
            'synchrony': -1,
            'bar_length_in_quarter': float(measure_se.loc[bdx, 'offset']) - float(measure_se.loc[bdx, 'onset']),
            'local_synchrony': -1,
        } for bdx in measure_se.index
    }
    for bdx in measure_se.index:
        b_onset, b_offset = onset_synchrony[bdx]['onset'], onset_synchrony[bdx]['offset']
        # note played in this measure, offset is in this measure
        notes =  concat_all_note_df[(concat_all_note_df['offset']<=b_offset) & (concat_all_note_df['offset']>b_onset)].shape[0]
        # new notes in this measure, onset is in this measure
        new_notes = concat_all_note_df[(concat_all_note_df['onset']<b_offset) & (concat_all_note_df['onset']>=b_onset)].shape[0]
        if notes>0:
            onset_synchrony[bdx]['notes'] = notes
            onset_synchrony[bdx]['new_notes'] = new_notes
            onset_synchrony[bdx]['synchrony'] = new_notes / notes
            onset_synchrony[bdx]['local_synchrony'] = new_notes / notes * onset_synchrony[bdx]['bar_length_in_quarter']

            total_synchrony += onset_synchrony[bdx]['local_synchrony']# * onset_synchrony[bdx]['bar_length_in_quarter']
    total_synchrony /= length
    # Normalize by total length of the extract
    return onset_synchrony, total_synchrony

def get_local_statistical_features(note_df, measure_se, mn):
    # normalized
    bar_onset, bar_offset = int(measure_se.loc[mn, 'onset']), int(measure_se.loc[mn, 'offset'])
    bar_length = measure_se.loc[mn, 'offset'] - measure_se.loc[mn, 'onset']
    ts_n, ts_d = measure_se.loc[mn, 'time_signature'].split('/')
    ts_n, ts_d = int(ts_n), int(ts_d)
    number_of_ticks = ts_n * 4 / ts_d * 4  # number of ticks, a tick = semiquaver
    beat_length = 4/ts_d

    total_notes = note_df[(note_df['offset']>measure_se.loc[mn, 'onset']) &
                          (note_df['onset']<measure_se.loc[mn, 'offset'])]
    total_notes['new_onset'] = total_notes['onset'].apply(lambda x: x if x>bar_onset else bar_onset)
    total_notes['new_offset'] = total_notes['offset'].apply(lambda x: x if x<bar_offset else bar_offset)
    total_notes['new_duration'] = total_notes['duration'].apply(lambda x: x if x<bar_length else bar_length)
    total_new_notes = note_df[(note_df['onset']>=measure_se.loc[mn, 'onset']) &
                              (note_df['onset']<measure_se.loc[mn, 'offset'])]
    total_new_notes['new_onset'] = total_new_notes['onset'].apply(lambda x: x if x>bar_onset else bar_onset)
    total_new_notes['new_offset'] = total_new_notes['offset'].apply(lambda x: x if x<bar_offset else bar_offset)
    total_new_notes['new_duration'] = total_new_notes['duration'].apply(lambda x: x if x<bar_length else bar_length)
    number_of_note = total_notes.shape[0]
    nb_of_total_new_notes = total_new_notes.shape[0]

    # duration
    if total_notes.shape[0]==0:
        return {
            'normalized_duration': -1,
            'number_of_note': -1,
            'number_of_new_note': -1,
            'occupation_rate': -1,
            'polyphony_rate': -1,
            'highest_pitch': -1,
            'lowest_pitch': -1,
            'mean_pitch': -1,
            'std_pitch': -1,
            'longest_duration': -1,
            'shortest_duration': -1,
            'mean_duration': -1,
            'std_duration': -1,
            'number_of_different_intervals': -1,
            'largest_interval': -1,
            'smallest_interval': -1,
            'mean_interval': -1,
            'std_interval': -1,
            'number_of_syncopated_note': -1
        }
    norm_dur = total_notes['new_duration'].sum() / bar_length
    occupation_rate = total_notes['new_duration'].sum() # / this bar all note's duration
    if total_new_notes.shape[0]==0:
        polyphony_rate = -1
    else:
        polyphony_rate = sum(np.unique(total_new_notes['new_onset'], return_counts=True)[1] > 1) / number_of_ticks  
    # pitch_feature_cols = highest, lowest, mean, std
    highest_pitch = total_notes.midi_number.max() / 128.0
    lowest_pitch = total_notes.midi_number.min() / 128.0
    mean_pitch =total_notes.midi_number.mean() / 128.0
    std_pitch = total_notes.midi_number.std() / 128.0
    # note_dur_feature_cols = ['longest_duration', 'shortest_duration', 'mean_duration', 'std_duration']
    longest_duration = total_notes.duration.max() / bar_length
    shortest_duration =total_notes.duration.min() / bar_length
    mean_duration = total_notes.duration.mean() / bar_length
    std_duration = total_notes.duration.std() / bar_length
    
    # pitch_interval_feature_cols
    intervals = []
    number_of_syncopated_note = 0
    notes_in_mn_df = total_notes.sort_values('onset').sort_values('midi_number')
    intervals += [b-a for a,b in zip(notes_in_mn_df.midi_number.tolist()[:-1],
                                        notes_in_mn_df.midi_number.tolist()[1:])]
    # syncopation_feature_cols = # of syncopated notes
    # For each note, if it is not "on beat":
    beat_lst = np.arange(
        measure_se.loc[mn, 'onset'], 
        measure_se.loc[mn, 'offset'],
        beat_length).tolist()
    number_of_syncopated_note += len(
        [_ for _ in notes_in_mn_df.onset.values if _ not in beat_lst]
    )

    if len(intervals) > 0:
        number_of_different_intervals = len(np.unique(intervals)) / len(intervals)
        largest_interval = (np.abs(np.unique(intervals))).max() / 128.0
        smallest_interval = (np.abs(np.unique(intervals))).min() / 128.0
        mean_interval= (np.abs(np.unique(intervals))).mean() / 128.0
        std_interval = (np.abs(np.unique(intervals))).std() / 128.0
    else:
        number_of_different_intervals, largest_interval, smallest_interval, mean_interval, std_interval = -1, -1, -1, -1, -1

    number_of_syncopated_note /= number_of_note

    return {
        'normalized_duration': norm_dur,
        'number_of_note': number_of_note,
        'number_of_new_note': nb_of_total_new_notes,
        'occupation_rate': occupation_rate,
        'polyphony_rate': polyphony_rate,
        'highest_pitch': highest_pitch,
        'lowest_pitch': lowest_pitch,
        'mean_pitch': mean_pitch,
        'std_pitch': std_pitch,
        'longest_duration': longest_duration,
        'shortest_duration': shortest_duration,
        'mean_duration': mean_duration,
        'std_duration': std_duration,
        'number_of_different_intervals': number_of_different_intervals,
        'largest_interval': largest_interval,
        'smallest_interval': smallest_interval,
        'mean_interval': mean_interval,
        'std_interval': std_interval,
        'number_of_syncopated_note': number_of_syncopated_note
    }

def convert_dataset(dataset_name):
    if dataset_name == 's3':
        from ..settings.s3_info import piece_names, string_to_int, int_to_string, file_path
    elif dataset_name == 'orchestration':
        from ..settings.orchestration_info import piece_names, string_to_int, int_to_string, file_path
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')

    save_at = os.path.join('converted_dataset', f'rf', dataset_name)
    os.makedirs(save_at, exist_ok=True)

    for piece_name in piece_names:
        print(f'Processing {piece_name} ...')

        all_track_global_df = []
        all_track_local_df = []
        each_track_df = []

        all_note_df = {}
        all_bar_dct = {}
        # all notes (even from different instrument parts/voices/tracks) are put together in a single table
        with open(os.path.join(
            file_path['converted_path'], piece_name, 'role', 'track_order.txt')
        ) as f:
            track_names = f.readline().split(';')

        if dataset_name=='s3':
            metadata_df = pd.read_csv(os.path.join(
                file_path['converted_path'], 'metadata.csv'), index_col=0)
            role = np.load(
                    os.path.join(file_path['converted_path'], piece_name, 'role', 'role.npy')
                )
        elif dataset_name=='orchestration':
            metadata_df = pd.read_csv(os.path.join(
                '.', 'dataset', 'new_metadata-2025-05-21.csv'
            ))
            role = np.load(
                    os.path.join('.', 'dataset', 'annotations', f"{piece_name}.npy")
                )
        
        measure_se = pd.read_csv(os.path.join(
            file_path['converted_path'], piece_name, 'temporal_attributes', 'measure_se.csv'), index_col=0
            )

        print(f'{piece_name=} | {track_names=} | {role.shape=}')

        all_note_df = []
        for tdx,track_name in enumerate(track_names):
            note_df = pd.read_csv(os.path.join(file_path['converted_path'], piece_name, 'note', track_name+'.csv'))
            note_df = note_df[note_df['onset']<note_df['offset']]
            print(f'{tdx} {track_name} | {note_df.shape=}')
            all_note_df.append(note_df)

        ###### global statistical features: calculate statistical features at piece-lavel ######
        global_statistical_features_for_this_piece = {}
        # including all instrument parts/voices/tracks
        concat_all_note_df = pd.concat(all_note_df)  # all notes in this song
        global_statistical_features_for_this_piece['all'] = get_global_statistical_features(
            concat_all_note_df, measure_se)
        # single track
        for tdx, track_name in enumerate(track_names):
            global_statistical_features_for_this_piece[track_name] = get_global_statistical_features(
                all_note_df[tdx], measure_se)
            
        ###### local statistical features: calculate statistical features at bar-level ######
        # including all instrument parts/voices/tracks
        local_statistical_features_for_this_piece = {mn: {} for mn in measure_se.index}
        for mn in measure_se.index:
            local_statistical_features_for_this_piece[mn]['all'] = get_local_statistical_features(
                concat_all_note_df, measure_se, mn)
        # single track
        for tdx, track_name in enumerate(track_names):
            for mn in measure_se.index:
                local_statistical_features_for_this_piece[mn][track_name] = get_local_statistical_features(
                    all_note_df[tdx], measure_se, mn)
                local_statistical_features_for_this_piece[mn][track_name]['occupation_rate'] /= local_statistical_features_for_this_piece[mn]['all']['occupation_rate'] 

        ###### all track statistical features ######
        all_track_df = []
        for mn in measure_se.index:
            tmp_dct = {
                'measure_number': mn,
                'piece_name': piece_name,
                'time_signature': measure_se.loc[mn, 'time_signature'],
            }
            # all tracks, all bars
            tmp_dct.update({'all_track_glob-'+k:v for k,v in global_statistical_features_for_this_piece['all'].items() if k!= 'onset_synchrony'})
            # all tracks, a bar
            tmp_dct.update({'all_track_bar-onset_synchrony': global_statistical_features_for_this_piece['all']['onset_synchrony'][mn]['local_synchrony']})
            tmp_dct.update({'all_track_bar-'+k:v for k,v in local_statistical_features_for_this_piece[mn]['all'].items()})
            all_track_df.append(tmp_dct)
        all_track_df = pd.DataFrame(all_track_df)
        all_track_df.to_csv(os.path.join(
            save_at, f'{piece_name}_all_track.csv'), index=False)

        ###### each track statistical features ######
        if dataset_name=='orchestration':
            if piece_name=='beethoven_op125-1' or piece_name=='k550-1':  # these two pieces' tracks have dummy tracks
                new_role = np.zeros((role.shape[0], role.shape[1]-1, role.shape[2]))
                new_role[:,:5,:] += role[:,:5,:]
                new_role[:,4:,:] += role[:,5:,:]
                role = new_role
        for tdx, track_name in enumerate(track_names):
            track_df = []
            for mn in measure_se.index:
                tmp_dct = {
                    'measure_number': mn,
                    'piece_name': piece_name,
                    'time_signature': measure_se.loc[mn, 'time_signature'],
                    'track_name': track_name,
                    'is_mel': role[mn, tdx][0],
                    'is_rhythm': role[mn, tdx][1],
                    'is_harm': role[mn, tdx][2],
                    'main_label': int(bool(role[mn,tdx][0])) + int(bool(role[mn,tdx][1])) * 2 + int(bool(role[mn,tdx][2])) * 4,
                }
                # a track, all bars
                tmp_dct.update({'track_glob-'+k:v for k,v in global_statistical_features_for_this_piece[track_name].items() if k!= 'onset_synchrony'})
                # a track, a bar
                tmp_dct.update({'track_bar-onset_synchrony': global_statistical_features_for_this_piece[track_name]['onset_synchrony'][mn]['local_synchrony']})
                tmp_dct.update({'track_bar-'+k:v for k,v in local_statistical_features_for_this_piece[mn][track_name].items()})
                tmp_dct.update({inst_name: 0 for inst_name in INSTRUMENT_NAME})
                tmp_dct[INSTRUMENT_NAME[find_inst_index(track_name)]] = 1
                track_df.append(tmp_dct)
            track_df = pd.DataFrame(track_df)
            track_df.to_csv(os.path.join(
                save_at, f'{piece_name}_{track_name}.csv'), index=False)


# %%[markdown]
# ## Create statistical features and store them into csv files
#  for training Random Forest Classifiers

if __name__=='__main__':
    convert_dataset('s3')
    convert_dataset('orchestration')



