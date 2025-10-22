"""
File: statistical_features.py
Author: Zih-Syuan Lin (2025)
"""
from typing import Final

track_info_feature_cols = [
    'normalized_duration', 'number_of_note', 'occupation_rate', 'polyphony_rate']
pitch_feature_cols = [
    'highest_pitch', 'lowest_pitch', 'mean_pitch', 'std_pitch']
pitch_interval_feature_cols = [
    'number_of_different_intervals', 'largest_interval', 'smallest_interval', 'mean_interval',
    'std_interval']
note_dur_feature_cols = [
    'longest_duration', 'shortest_duration', 'mean_duration', 'std_duration']
syncopation_feature_cols = [
    'number_of_syncopated_note']
all_pitch_feature_cols = [
    'all_highest_pitch', 'all_lowest_pitch', 'all_mean_pitch', 'all_std_pitch']
all_pitch_interval_feature_cols = [
    'all_number_of_different_intervals', 'all_largest_interval', 'all_smallest_interval',
    'all_mean_interval', 'all_std_interval']
all_note_dur_feature_cols = [
    'all_longest_duration', 'all_shortest_duration', 'all_mean_duration', 'all_std_duration']
all_syncopation_feature_cols = [
    'all_number_of_syncopated_note']
global_pitch_feature_cols = [
    'global_highest_pitch', 'global_lowest_pitch', 'global_mean_pitch', 'global_std_pitch']
global_pitch_interval_feature_cols = [
    'global_number_of_different_intervals', 'global_largest_interval',
    'global_smallest_interval', 'global_mean_interval', 'global_std_interval']
global_note_dur_feature_cols = [
    'global_longest_duration', 'global_shortest_duration', 'global_mean_duration',
    'global_std_duration']
global_syncopation_feature_cols = [
    'global_number_of_syncopated_note']
statistical_feature_cols = track_info_feature_cols + pitch_feature_cols \
    + pitch_interval_feature_cols + note_dur_feature_cols + syncopation_feature_cols \
    + all_pitch_feature_cols+all_pitch_interval_feature_cols \
    + all_note_dur_feature_cols + all_syncopation_feature_cols
global_statistical_feature_cols = global_pitch_feature_cols \
    + global_pitch_interval_feature_cols + global_note_dur_feature_cols \
    + global_syncopation_feature_cols


STAT_FEATURE_COLS:Final[dict[str,list[str]]] = {
    'y': ['main_label'],
    'y_3': ['is_mel', 'is_rhythm', 'is_harm'],
    'basic_info': [
        'piece_name', 'time_signature', 'track_name'
    ],
    # a track, a bar
    'track_bar': [
        'track_bar-onset_synchrony', #
        'track_bar-normalized_duration',#
        # 'track_bar-number_of_note',
        # 'track_bar-number_of_new_note',
        'track_bar-occupation_rate',
        'track_bar-polyphony_rate', #
        'track_bar-highest_pitch',#
        'track_bar-lowest_pitch',#
        'track_bar-mean_pitch',#
        'track_bar-std_pitch',#
        'track_bar-longest_duration',#
        'track_bar-shortest_duration',#
        'track_bar-mean_duration',#
        'track_bar-std_duration',#
        'track_bar-number_of_different_intervals',#
        'track_bar-largest_interval',#
        'track_bar-smallest_interval',#
        'track_bar-mean_interval',#
        'track_bar-std_interval',#
        'track_bar-number_of_syncopated_note', #
    ],
    # a track, all bars
    'track': [
        'track_glob-global_highest_pitch',
        'track_glob-global_lowest_pitch',
        'track_glob-global_mean_pitch',
        'track_glob-global_std_pitch',
        'track_glob-global_longest_duration',
        'track_glob-global_shortest_duration',
        'track_glob-global_mean_duration',
        'track_glob-global_std_duration',
        'track_glob-global_number_of_different_intervals',
        'track_glob-global_largest_interval',
        'track_glob-global_smallest_interval',
        'track_glob-global_mean_interval',
        'track_glob-global_std_interval',
        'track_glob-global_number_of_syncopated_note',
        'track_glob-total_synchrony',
    ],
    # all track, a bar
    'all_track_bar': [
        'all_track_bar-onset_synchrony', 
        'all_track_bar-normalized_duration', 
        # 'all_track_bar-number_of_note',
        # 'all_track_bar-number_of_new_note',
        # 'all_track_bar-occupation_rate',
        'all_track_bar-polyphony_rate',
        'all_track_bar-highest_pitch',
        'all_track_bar-lowest_pitch',
        'all_track_bar-mean_pitch',
        'all_track_bar-std_pitch',
        'all_track_bar-longest_duration', 
        'all_track_bar-shortest_duration',
        'all_track_bar-mean_duration',
        'all_track_bar-std_duration',
        'all_track_bar-number_of_different_intervals',
        'all_track_bar-largest_interval',
        'all_track_bar-smallest_interval',
        'all_track_bar-mean_interval',
        'all_track_bar-std_interval',
        'all_track_bar-number_of_syncopated_note' 
    ],
    # all track, all bars
    'all_track': [
        'all_track_glob-global_highest_pitch',
        'all_track_glob-global_lowest_pitch',
        'all_track_glob-global_mean_pitch',
        'all_track_glob-global_std_pitch',
        'all_track_glob-global_longest_duration',
        'all_track_glob-global_shortest_duration',
        'all_track_glob-global_mean_duration',
        'all_track_glob-global_std_duration',
        'all_track_glob-global_number_of_different_intervals',
        'all_track_glob-global_largest_interval',
        'all_track_glob-global_smallest_interval',
        'all_track_glob-global_mean_interval',
        'all_track_glob-global_std_interval',
        'all_track_glob-global_number_of_syncopated_note',
        'all_track_glob-total_synchrony',
    ],
    'instrument': [
        'englishhorn', 'flute', 'piccolo', 'oboe',
        'clarinet', 'bassoon', 'contrabassoon', 'horn', 'piccolotrumpet',
        'trumpet', 'trombone', 'basstrombone', 'tuba', 'violin', 'viola',
        'violoncello', 'doublebass', 'timpani', 'triangle', 'cymbal', 'tamtam',
        'bassdrum', 'sopranosolo', 'sopranochoir', 'altosolo', 'altochoir',
        'tenorsolo', 'tenorchoir', 'baritonesolo', 'basschoir', 'none',
    ],
    'cnn_feature': [f'cnn_embd-{_}' for _ in range(360)],
}

