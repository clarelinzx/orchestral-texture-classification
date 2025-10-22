'''
File: PianoRollsDataset.py
Purpose:      
    For each dataset, select specific pieces, prepare 
    and load piano roll data from preprocessed dataset 
    to an object. Class PianoRoll and PianoRollDataset
    are based onChu's code (https://github.com/YaHsuanChu/orchestraTextureClassification).
Author: Zih-Syuan (2025)
'''

# %%[markdown]
# ## Imports
import cv2
import pypianoroll
import random
import time
import torch
import os
from functools import wraps
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

from settings.instruments import INSTRUMENT_LOOKUP_TABLE, INSTRUMENT_NAME, INSTRUMENT_NAME_VARIATION
from settings.orchestration_info import int_to_string as piece_name_in_str
from settings.s3_info import int_to_string as piece_name_in_str_s3

# %%[markdown]
# ## Functions
def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' took {elapsed_time:.4f} seconds to complete.")
        return result
    return wrapper

# %% [markdown]
# ## PianoRoll
class PianoRoll:
    def __init__(
            self, 
            npz_file=None, 
            n_bars_in_annotation=None, 
            time_signature=None,
            time_signature_for_beat=None,
            rest_fp=None,
            replaymtx_fp=None,
            rest_replaymtx_fp=None,
            add_rest=False,
            add_replaymtx=False,
            piece_name=None,
            piece_index=None,
            inst_list=None,
            ds='orchestration', # 'orchestration' or 's3'
            load_from_npz=False,
            ):
        if npz_file is not None and ds=='orchestration' or load_from_npz:
            self.multitrack = pypianoroll.load(npz_file)
            self.n_tracks = len(self.multitrack.tracks)
            self.inst_list = [track.name for track in self.multitrack.tracks]
            self.resolution = self.multitrack.resolution
            self.active_length = self.multitrack.get_length()
            self.max_length = self.multitrack.get_max_length()
        elif ds=='s3':
            self.multitrack = np.load(npz_file)
            self.inst_list = inst_list
            self.active_length = self.multitrack.shape[1]
            self.max_length = self.multitrack.shape[1]
            self.n_tracks = self.multitrack.shape[0]
            self.resolution = 24
        self.ds = ds
        self.piece_name = piece_name
        self.piece_index=piece_index
        self.n_bars_in_annotation = n_bars_in_annotation 
        self.time_signature = self.parse_time_signature(time_signature)
        self.time_signature_for_beat = self.parse_time_signature_for_beat(time_signature_for_beat)
        self.add_rest = add_rest
        self.add_replaymtx = add_replaymtx
        if self.add_rest:
            self.rest = np.load(rest_fp)
            if self.add_replaymtx:
                self.rest_replaymtx = np.load(rest_replaymtx_fp)
        if self.add_replaymtx:
            self.replaymtx = np.load(replaymtx_fp)
        if self.ds=='orchestration':
            self.check_theoretical_total_length()
    
    def parse_time_signature_for_beat(self, time_signature_for_beat):
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

    def parse_time_signature(self, string):
        '''parse time_signature information from metadata.csv and save to a list
        input format: 
            "beg_measure:end_measure=number_of_quarter_beats_per_bar, ..."
            example:
                "1:10=4,11:20=3" means:
                { measure 1 to 10: 4/4 or 2/2,
                  measure 11 20 20: 3/4 or 6/8 }
        return:
            a list of 3-tuple
            [ [begin of a segment, end of a segment, number of quarter beats per bar],
              [ ...                                               ],
              ...                                                   ]
            example:
                [ [1,10,4],
                  [11,20,3] ] a list shape (2, 3)'''

        time_signature_list = []
        raw_segs = string.split(',')
        for seg in raw_segs:
            n_quarter_beat = int(seg.split('=')[-1])
            interval = seg.split('=')[0]
            beg = int(interval.split(':')[0])
            end = int(interval.split(':')[1])
            time_signature_list.append( [beg, end, n_quarter_beat] )
        
        return time_signature_list
        
    def measure_to_time_step(self, measure):
        '''return the equivelant time step of the begining of a measure
        example.
            if measure = 1, return 0
            if measure = 10, and time signature = 4/4, return 9*4*resolution '''
        
        if measure > self.n_bars_in_annotation+1 or measure < 1:
            raise Exception('''Error: Measure number out of range !!!''')
        else:
            steps = 0
            for (beg, end, n_q_beat) in self.time_signature:
                if measure > end: # contain the whole segment
                    steps += (end-beg+1)*n_q_beat*self.resolution
                elif measure > beg: # break inside the segment
                    steps += (measure-beg)*n_q_beat*self.resolution
                else: # do not contain the segment
                    pass
            return steps

    def measure_interval_to_step_interval(self, beg_measure, end_measure):
        '''calculate the desired time steps range [beg_measure, end_measure]
        which is equivelant to [beg_time_step, end_time_step)
        return: 
            (beg_time_step, end_time_step)
        example.
            input: [beg_measure, end_measure] = [1, 10] and time_signature=4/4 for this segment
            return: (0, 40*resolution) '''
        
        return (self.measure_to_time_step(beg_measure), self.measure_to_time_step(end_measure+1))

    def get_trimmed_multitrack_by_measure(self, beg_measure, end_measure):
        '''the tracks of all instruments in range [beg_measure, end_measure]
        return an pypianoroll.Multitrack object with all tracks trimed'''
        
        beg, end = self.measure_interval_to_step_interval(beg_measure, end_measure)
        if self.ds=='orchestration':
            return self.multitrack.trim(start=beg, end=end)
        elif self.ds=='s3':
            return self.multitrack[beg:end, :]

    def get_trimmed_multitrack_by_time_step(self, beg_step, end_step):
        '''the tracks of all instrument in range [beg_step, end_step]
        return an pypianoroll.Multitrack object with all tracks trimmed'''

        if beg_step < 0 or beg_step >= self.max_length or end_step < 0 or end_step >= self.max_length:
            raise Exception('Error: time step index out of range!!!')
        elif end_step < beg_step:
            raise Exception('Error: beg_step > end_step')
        if self.ds=='orchestration':
            return self.multitrack.trim(start=beg_step, end=end_step)
        else:
            return self.multitrack[beg_step:end_step, :]

    def get_pianoroll_by_inst_and_measure(self, inst_index, measure):
        if measure<1 or measure>self.n_bars_in_annotation:
            raise Exception('Error:  measure number out of range !!!')
        if inst_index<0 or inst_index>=self.n_tracks:
            raise Exception('Error: instrument index out of range !!!')
        beg_step, end_step = self.measure_interval_to_step_interval( measure, measure )
        if self.ds=='orchestration':
            inst_pnoroll = self.multitrack.tracks[inst_index].pianoroll[beg_step:end_step]
        elif self.ds=='s3':
            inst_pnoroll = self.multitrack[inst_index][beg_step:end_step, :]
        if self.add_rest:
            new_pnoroll_slice = np.concatenate([
                    inst_pnoroll,
                    self.rest[inst_index][beg_step:end_step]
                ], axis=1)
            if self.add_replaymtx:
                new_replaymtx_slice = np.concatenate([
                    self.replaymtx[inst_index][beg_step:end_step],
                    self.rest_replaymtx[inst_index][beg_step:end_step]
                ], axis=1)
                return new_pnoroll_slice, new_replaymtx_slice
            else:
                return new_pnoroll_slice, None
        else:
            if self.add_replaymtx:
                return inst_pnoroll, self.replaymtx[inst_index][beg_step:end_step]
        
            else:
                return inst_pnoroll, None
    
    def get_pianoroll_blended_by_measure(self, measure):
        '''stack all the tracks and sum them up'''
        if measure<1 or measure>self.n_bars_in_annotation:
            raise Exception('Error:  measure number out of range !!!')
        beg_step, end_step = self.measure_interval_to_step_interval( measure, measure )
        
        stacked = []
        stacked_replaymtx = []
        if self.ds=='orchestration':
            tracks = self.multitrack.tracks
        elif self.ds=='s3':
            tracks = self.multitrack
        for tdx,track in enumerate(tracks):
            if self.add_rest:
                if self.add_replaymtx:
                    stacked_replaymtx.append(np.concatenate([
                        self.replaymtx[tdx][beg_step:end_step],
                        self.rest_replaymtx[tdx][beg_step:end_step]
                ], axis=1))
                if self.ds=='orchestration':
                    a_pnoroll = track.pianoroll[beg_step:end_step]
                elif self.ds=='s3':
                    a_pnoroll = track[beg_step:end_step, :]
                stacked.append(np.concatenate([
                    a_pnoroll,
                    self.rest[tdx][beg_step:end_step]
                ], axis=1))
            else:
                if self.add_replaymtx:
                    stacked_replaymtx.append(self.replaymtx[tdx][beg_step:end_step])
                if self.ds=='orchestration':
                    stacked.append(track.pianoroll[beg_step:end_step])
                elif self.ds=='s3':
                    stacked.append(track[beg_step:end_step, :])
        stacked = np.stack(stacked, axis=0).astype(np.float32)
        stacked = np.sum(stacked, axis=0)/(self.n_tracks)
        if self.add_replaymtx:
            stacked_replaymtx = np.stack(stacked_replaymtx, axis=0).astype(np.float32)
            stacked_replaymtx = np.sum(stacked_replaymtx, axis=0)/(self.n_tracks)
            return stacked, stacked_replaymtx
        else:
            return stacked, None

    ########## The code below is for manual data cleaning and debugging ###########
    def check_theoretical_total_length(self):
        '''calculate the theiretical length of the pianoroll using time signature imformation
        and compare to the real length of a pianoroll by get_max_length()'''
        
        theoretical_time_steps = 0
        for seg in self.time_signature:
            beg = seg[0]
            end = seg[1]
            n_q_beat = seg[2]
            theoretical_time_steps += (end-beg+1)*n_q_beat*self.resolution
        if self.max_length != theoretical_time_steps:
            raise Exception('Error: max length = {:<6d}, theoretical length = {:<6d}, difference = {:<4d}'.format(\
                                self.max_length, theoretical_time_steps, self.max_length-theoretical_time_steps))
    
    def check_piece_length_by_downbeat_count(self):
        '''see if downbeats number match the total number of bars in a piece'''
        if self.ds!='orchestration':
            raise Exception('Error: this function only works for orchestration dataset !!!')
        count_annot = self.n_bars_in_annotation
        count_pianoroll = self.multitrack.count_downbeat()
        if count_annot != count_pianoroll:
            print('number of down beats: {:<5d}, number of bars in annotation: {:<5d}, do not match!!!'.format(count_pianoroll, count_annot))
        else:
            print('Consistent number of bars')

# %% [markdown]
# ## PianoRollDataset

class PianoRollsDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            meta_csv_file, 
            test=False, 
            test_piece=[0, 4, 9], 
            context=0,
            other_inst=False, 
            blend='SUM', 
            k=5,
            add_inst=None,
            add_barlines=False,
            add_beats = False,
            converted_path = '',
            ds='orchestration',
            add_rest=False,
            add_replaymtx=False,
            load_from_npz=False
        ):
        # ''' configuration '''
        self.meta_df = pd.read_csv(meta_csv_file)
        self.test_data = test_piece
        self.context_measures = context
        self.other_inst = other_inst
        self.blend_mode = blend
        self.k = k
        self.add_inst = add_inst
        self.add_barlines = add_barlines
        self.add_beats = add_beats
        self.add_rest = add_rest
        self.add_replaymtx = add_replaymtx
        self.converted_path = converted_path
        self.ds = ds
        self.return_chose_k = False
        self.load_from_npz = load_from_npz
        if self.add_rest:
            self.pitch_count = 129
        else:
            self.pitch_count = 128
                
        # ''' define input channels '''
        if not other_inst: 
            self.n_input_channels = 1
            if self.add_replaymtx:
                self.n_input_channels += 1
        else:
            if self.blend_mode=='SUM': 
                self.n_input_channels = 2
                if self.add_replaymtx: 
                    self.n_input_channels += 2
            elif self.blend_mode=='COMB': 
                self.n_input_channels = k
                if self.add_replaymtx:
                    self.n_input_channels += k
        
        if self.add_inst=='target_inst':
            self.n_input_channels += 1
        elif self.add_inst=='all':
            if self.blend_mode=='COMB':
                self.n_input_channels += k
            elif self.blend_mode=='SUM':
                self.n_input_channels += 2
            else:
                self.n_input_channels += 1
        
        if self.add_barlines:
            self.n_input_channels += 1
        if self.add_beats:
            self.n_input_channels += 1
        
        # pad empty bar: all channels are empty
        self.empty = np.zeros(
            (self.n_input_channels, 96, self.pitch_count),
            dtype=np.float32
        )
        # give the bar line
        if self.add_barlines:
            self.a_barline = np.zeros((96, self.pitch_count), dtype=np.float32)
            self.a_barline[:1, :] = 1.0
        if self.add_beats:
            self.ts_fig = self.get_ts_fig()
        # give the beat line: related to time signature
        
        # ''' keep desired pieces only '''
        if test:                                            
            self.meta_df = self.meta_df.iloc[self.test_data]
            print('There are {} pieces in test set'.format(self.meta_df.shape[0]))
        else:
            self.meta_df = self.meta_df.drop(self.test_data, axis=0)
            print('There are {} pieces in train set'.format(self.meta_df.shape[0]))
        self.meta_df.reindex( range(self.meta_df.shape[0]), axis=0 )
        # ''' read into pianoRoll Objects '''
        self.pianoRoll_list = {}
        for piece in self.meta_df.index:
            if ds == 'orchestration':
                piece_name = piece_name_in_str[piece]
                self.pianoRoll_list.update({piece:(
                    PianoRoll(
                        npz_file=self.meta_df.loc[piece,'pianoroll'],
                        n_bars_in_annotation=self.meta_df.loc[piece,'n_measures'],
                        time_signature=self.meta_df.loc[piece,'time_signature'],
                        time_signature_for_beat=self.meta_df.loc[piece,'time_signature_beat'],
                        rest_fp=os.path.join(converted_path[ds]['converted_path'], piece_name, 'from_notecsv', 'rest_pnoroll.npy'),
                        replaymtx_fp=os.path.join(converted_path[ds]['converted_path'], piece_name, 'from_notecsv', 'replaymtx.npy'),
                        rest_replaymtx_fp=os.path.join(converted_path[ds]['converted_path'], piece_name, 'from_notecsv', 'rest_replaymtx.npy'),
                        add_replaymtx=self.add_replaymtx,
                        add_rest=self.add_rest,
                        piece_name=piece_name,
                        piece_index=piece,
                        ds='orchestration',
                ))})
            elif ds == 's3':
                piece_name = piece_name_in_str_s3[piece]
                if self.load_from_npz:
                    self.pianoRoll_list.update({piece:(
                        PianoRoll(
                            npz_file=os.path.join(converted_path[ds]['converted_path'], piece_name, 'pianoroll.npz'),
                            n_bars_in_annotation=self.meta_df.loc[piece,'n_measures'],
                            time_signature=self.meta_df.loc[piece,'time_signature'],
                            time_signature_for_beat=self.meta_df.loc[piece,'time_signature_beat'],
                            rest_fp=os.path.join(converted_path[ds]['converted_path'], piece_name, 'from_notecsv', 'rest_pnoroll.npy'),
                            replaymtx_fp=os.path.join(converted_path[ds]['converted_path'], piece_name, 'from_notecsv', 'replaymtx.npy'),
                            rest_replaymtx_fp=os.path.join(converted_path[ds]['converted_path'], piece_name, 'from_notecsv', 'rest_replaymtx.npy'),
                            add_replaymtx=self.add_replaymtx,
                        add_rest=self.add_rest,
                        piece_name=piece_name,
                        piece_index=piece,
                        ds='s3',
                        inst_list=self.meta_df.loc[piece,'track_names'].split(';'),
                        load_from_npz=self.load_from_npz,
                ))})
                else:
                    self.pianoRoll_list.update({piece:(
                        PianoRoll(
                            npz_file=os.path.join(converted_path[ds]['converted_path'], piece_name, 'from_notecsv', 'pnoroll.npy'),
                            n_bars_in_annotation=self.meta_df.loc[piece,'n_measures'],
                            time_signature=self.meta_df.loc[piece,'time_signature'],
                            time_signature_for_beat=self.meta_df.loc[piece,'time_signature_beat'],
                            rest_fp=os.path.join(converted_path[ds]['converted_path'], piece_name, 'from_notecsv', 'rest_pnoroll.npy'),
                            replaymtx_fp=os.path.join(converted_path[ds]['converted_path'], piece_name, 'from_notecsv', 'replaymtx.npy'),
                            rest_replaymtx_fp=os.path.join(converted_path[ds]['converted_path'], piece_name, 'from_notecsv', 'rest_replaymtx.npy'),
                            add_replaymtx=self.add_replaymtx,
                        add_rest=self.add_rest,
                        piece_name=piece_name,
                        piece_index=piece,
                        ds='s3',
                        inst_list=self.meta_df.loc[piece,'track_names'].split(';'),
                ))})
        # ''' read label arrays '''
        label_list = {}
        for piece in self.meta_df.index:
            if self.ds=='orchestration':
                label_list.update(
                    {piece: np.load(self.meta_df.loc[piece,'annot_npy'])}
                    )
            elif self.ds=='s3':
                if self.load_from_npz:
                    label_list.update(
                        {piece: np.load(os.path.join(converted_path[ds]['converted_path'], piece_name_in_str_s3[piece], 'role', 'role_new.npy'))}
                        )
                else:
                    label_list.update(
                        {piece: np.load(os.path.join(converted_path[ds]['converted_path'], piece_name_in_str_s3[piece], 'role', 'role.npy'))}
                        )

        # ''' select eligible data, and store as id=[piece_index, measure, instrment_index] '''
        # ''' for each piece:
        #         for each bar:
        #             for each instrument:
        #                 if label is not [False, Fasle, False]
        #                 x.append()
        #                 label.append()
        # '''       
        self.accessor = [] # id to access a data, not loading data here because of memory capacity limit
        self.label = []
        count_empty = 0
        for piece in self.meta_df.index:
            pianoroll = self.pianoRoll_list[piece]
            for measure in range(pianoroll.n_bars_in_annotation):
                for inst in range(pianoroll.n_tracks):
                    if np.any(label_list[piece][measure][inst]): #has at least one role
                        pnoroll, replaymtx = pianoroll.get_pianoroll_by_inst_and_measure(inst, measure+1)
                        if np.any(pnoroll[:,:128]): # if no notes played ,discard
                            self.accessor.append(np.array([piece, measure, inst]))
                            self.label.append(label_list[piece][measure][inst])
                        else: #an empty bar 
                            count_empty += 1
                                               
        
        self.accessor = np.array(self.accessor)
        self.label = torch.from_numpy(np.array(self.label)).float()
        print( 'There are {} (inst, measure) data in this dataset'.format(len(self.label)) )

    def get_ts_fig(self):
        ts_44 = np.zeros((96, self.pitch_count), dtype=np.float32)
        for _ in range(0, 96, int(96/4)): ts_44[_:_+1, :] = 1.0
        
        ts_22 = np.zeros((96, self.pitch_count), dtype=np.float32)
        for _ in range(0, 96, int(96/2)): ts_22[_:_+1, :] = 1.0
        
        ts_34 = np.zeros((96, self.pitch_count), dtype=np.float32)
        for _ in range(0, 96, int(96/3)): ts_34[_:_+1, :] = 1.0
        
        ts_68 = np.zeros((96, self.pitch_count), dtype=np.float32)
        for _ in range(0, 96, int(96/6)): ts_68[_:_+1, :] = 1.0
        
        ts_54 = np.zeros((96, self.pitch_count), dtype=np.float32)
        for _ in range(0, 96, int(96/5)): ts_54[_:_+1, :] = 1.0

        ts_12 = np.zeros((96, self.pitch_count), dtype=np.float32)
        for _ in range(0, 96, int(96/12)): ts_12[_:_+1, :] = 1.0

        time_sign_fig = {
            '4/4': ts_44,
            '3/4': ts_34,
            '2/2': ts_22,
            '2/4': ts_22,
            '6/8': ts_68,
            '3/2': ts_34,
            '6/4': ts_68,
            '4/8': ts_44,
            '5/4': ts_54,
            '12/8': ts_12,
        }
        return time_sign_fig
    
    def __len__(self):
        return len(self.label)

    def get_inst_value(self, inst_name):
        if self.ds=='s3' and not self.load_from_npz:
            return INSTRUMENT_LOOKUP_TABLE[inst_name] / len(INSTRUMENT_NAME)
        else:            
            return INSTRUMENT_LOOKUP_TABLE[INSTRUMENT_NAME_VARIATION[inst_name]] / len(INSTRUMENT_NAME)

    # @time_it
    def __getitem__(self, index):
        data_id = self.accessor[index]
        piece, predicting_measure, inst = data_id[0], data_id[1], data_id[2] 
        pianoroll = self.pianoRoll_list[piece]
        x = []
        chosen_k = []
        
        if self.add_inst is not None:
            target_inst_value = self.get_inst_value(pianoroll.inst_list[inst])
            target_inst_value = np.ones((96, self.pitch_count),dtype=np.float32) * target_inst_value
            # print(f"{target_inst_value.shape}")
        for getting_measure in range(
            predicting_measure-self.context_measures,
            predicting_measure+self.context_measures+1
            ):
            if getting_measure>=0 and getting_measure<pianoroll.n_bars_in_annotation:
                pnoroll, replaymtx = pianoroll.get_pianoroll_by_inst_and_measure(inst, getting_measure+1)
                stacked = [self.transform(pnoroll)]
                # print(f"{stacked[-1].shape}")
                if self.add_replaymtx:
                    target_replaymtx = self.transform_r(replaymtx)
                    stacked += [target_replaymtx]
                    # print(f"{stacked[-1].shape}")
                if self.other_inst and self.blend_mode=='SUM':
                    pnoroll, replaymtx = pianoroll.get_pianoroll_blended_by_measure(getting_measure+1)
                    if self.add_replaymtx:
                        stacked += [self.transform(pnoroll), self.transform_r(replaymtx)]
                        # print(f"{stacked[-1].shape}")
                    else:
                        stacked += [self.transform(pnoroll)]
                        # print(f"{stacked[-1].shape}")
                    if self.add_inst:
                        stacked = [target_inst_value] + stacked
                elif self.other_inst and self.blend_mode=='COMB':
                    n = pianoroll.n_tracks
                    pool = list(range(n))
                    pool.pop(inst)
                    chosen_k_minus_1 = random.sample(pool, self.k-1) #choose k-1 tracks from n-1 other tracks
                    other_tracks = []
                    chosen_k.append(chosen_k_minus_1)
                    # print(f"{chosen_k_minus_1=}")
                    for other_inst_index in chosen_k_minus_1:
                        p, r = pianoroll.get_pianoroll_by_inst_and_measure(
                            other_inst_index, getting_measure+1)
                        other_tracks.append(self.transform(p))
                        if self.add_replaymtx:
                            other_tracks.append(self.transform_r(r))
                    stacked += other_tracks
                    if self.add_inst:
                        if self.add_inst=='all':
                            other_tracks_inst_value = []
                            for other_inst_index in chosen_k_minus_1:
                                other_tracks_inst_value.append(
                                    np.ones((96, self.pitch_count),dtype=np.float32) * self.get_inst_value(pianoroll.inst_list[other_inst_index])
                                )
                            stacked = [target_inst_value] + stacked + other_tracks_inst_value
                        else:
                            stacked = [target_inst_value] + stacked
                    
                else: # other_inst=False <=> input single track
                    if self.add_inst:
                        stacked = [target_inst_value] + stacked
                
                if self.add_barlines:
                    stacked.append(self.a_barline)
                if self.add_beats:
                    stacked.append(
                        self.ts_fig[pianoroll.time_signature_for_beat[getting_measure+1]]
                    )
                x.append(np.stack(stacked, axis=0))

            else: #pad zero arrays
                x.append(self.empty)
                chosen_k.append([None, None, None, None])

        if len(x)==1: #no context
            x = x[0]
        else:
            x = np.concatenate(x, axis=-2)
        if self.return_chose_k:
            return x, self.label[index], chosen_k            
        return x, self.label[index]
    
    def transform(self, one_bar_pianoroll, normalized=127.0):
        '''resize and normalize 
        input:
            one_bar_pianoroll, shape = (48, 128) or (72, 128) or (96, 128)
        return:
            one_bar_pianoroll, shape = (96, 128), value between 0~1, dtype=np.float32
        '''
        one_bar_pianoroll = cv2.resize( one_bar_pianoroll, (self.pitch_count, 96), interpolation=cv2.INTER_AREA ) #NOTE.pass (n_cols, n_rows) as arg in cv2
        one_bar_pianoroll = one_bar_pianoroll.astype(np.float32)/normalized # normalize
        return one_bar_pianoroll

    # @time_it
    def transform_r(self, a_replaymtx):
        a_replaymtx = cv2.resize( a_replaymtx, (self.pitch_count, 96), interpolation=cv2.INTER_AREA ) #NOTE.pass (n_cols, n_rows) as arg in cv2
        a_replaymtx = a_replaymtx.astype(np.float32) # normalize
        return a_replaymtx

    def label_statistics(self, label):
        '''label shohuld be of shape (n_data, 3)
        invetigate the distrubution of data'''
        
        print('Labels statistics:')
        n_data = label.shape[0]
        role_counts = np.sum(label, axis=0)
        total_labels = np.sum(role_counts)
        label_count_per_data = np.sum(label, axis=1)
        n_label_distribution = np.array([np.sum(label_count_per_data==n_label) for n_label in range(4)])
        
        n_mel_rhythm = np.sum(np.sum(label==[1, 1, 0], axis=1 )==3)
        n_rhythm_harm = np.sum(np.sum(label==[0, 1, 1], axis=1 )==3)

        print('number of [mel, rhythm, harm] labels = ', role_counts, ', : ', role_counts.astype(np.float32)/total_labels )
        print('data with [0, 1, 2, 3] labels = ', n_label_distribution, ', : ', n_label_distribution.astype(np.float32)/n_data)
        print('number of mel+rhythm = ', n_mel_rhythm)
        print('number of rhythm+hram = ', n_rhythm_harm)

