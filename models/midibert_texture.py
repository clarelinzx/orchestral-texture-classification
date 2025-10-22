"""
File: midibert_texture.py
Description: 
    This is a pseudo code modifying Midi-BERT from https://github.com/wazenmai/MIDI-BERT.
Author: Zih-Syuan (2025)
"""

import datetime
import os
import sys
import shutil
import pickle
import random

import miditoolkit
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from transformers import AdamW
import torch.nn.functional as F

from models.utils import print_confusion_matrix
from data_preprocessing.utils import find_inst_index
from settings.evaluation import PERFORMANCE_DF_COLS_MIDIBERT_MEL, PERFORMANCE_DF_COLS_MIDIBERT_TEX
from settings.s3_info import file_path as s3_file_path
from settings.s3_info import int_to_string as piece_name_in_str_s3
from settings.s3_info import meta_csv_path as meta_csv_file_s3
from settings.orchestration_info import file_path as orchestration_file_path
from settings.orchestration_info import int_to_string as piece_name_in_str
from settings.orchestration_info import meta_csv_path as meta_csv_file

converted_path = {
    's3': s3_file_path,
    'orchestration': orchestration_file_path
}
print('python kernal:', sys.executable)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('torch version:', torch.__version__)
print('CPU/GPU device:', device)

texture2mel = {
    0: 0,
    1: 0,
    2: 1,
    3: 3,
    4: 2,
    5: 3,
    6: 2,
    7: 3,
    8: 2,
}

s3_path =  os.path.join('dataset', 'converted_dataset', 's3')
orchestration_path =  os.path.join('dataset', 'converted_dataset', 'orchestration')

Composer = {
    "Bethel": 0,
    "Clayderman": 1,
    "Einaudi": 2,
    "Hancock": 3,
    "Hillsong": 4,
    "Hisaishi": 5,
    "Ryuichi": 6,
    "Yiruma": 7,
    "Padding": 8,
}

Emotion = {
    "Q1": 0,
    "Q2": 1,
    "Q3": 2,
    "Q4": 3,
}


# parameters for input
DEFAULT_VELOCITY_BINS = np.array([ 0, 32, 48, 64, 80, 96, 128])     # np.linspace(0, 128, 32+1, dtype=np.int)
DEFAULT_FRACTION = 16
DEFAULT_DURATION_BINS = np.arange(60, 3841, 60, dtype=int)
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]

# parameters for output
DEFAULT_RESOLUTION = 480

# define "Item" for general storage
class Item(object):
    def __init__(self, name, start, end, velocity, pitch, Type, inst_voice, texture_label=-1):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch
        self.Type = Type
        self.inst_voice = inst_voice
        self.texture_label = texture_label

    def __repr__(self):
        return 'Item(name={}, start={}, end={}, velocity={}, pitch={}, Type={}, inst_voice={}, texture_label={})'.format(
            self.name, self.start, self.end, self.velocity, self.pitch, self.Type, self.inst_voice, self.texture_label)

DEFAULT_RESOLUTION = 480

# for piece_name in piece_names:
    # read_items_from_note_csv(piece_name)

useful_cols = ['onset', 'offset', 'duration', 'time_signature', 'measure',
       'measure_offset', 'onset_time_step', 'offset_time_step', 'midi_number',
       'pitch_name', 'velocity', 'inst_name', 'is_mel', 'is_rhythm',
       'is_harm']

def read_items_from_note_csv(ds, piece_name, midi_path):
    # create storage
    # track_names
    with open(os.path.join(converted_path[ds]['converted_path'], piece_name, 'role', 'track_order.txt'), 'r') as f:
        track_names = f.readline().split(';')
    # note
    note_items = []
    for tdx,track_name in enumerate(track_names):
        note_df = pd.read_csv(os.path.join(converted_path[ds]['converted_path'], piece_name, 'note_xy', track_name+'.csv'))
        note_df.sort_values('onset').sort_values('midi_number')
        for row in note_df[useful_cols].to_numpy():
            onset, offset, _, _, _, _, _, _, midi_number, _, velocity, _, is_mel, is_rhythm, is_harm = row
            if offset==onset:
                continue
            note_items.append(Item(
                name='Note',
                start=int(onset*DEFAULT_RESOLUTION),  # unit=ticks
                end=int(offset*DEFAULT_RESOLUTION),
                velocity=velocity,
                pitch=midi_number,
                Type= find_inst_index(track_name),
                inst_voice=track_name,
                texture_label=is_mel*1 + is_rhythm*2 + is_harm*4,
            ))
    # notes in the whole piece
    note_items.sort(key=lambda x: x.start)
    
    # tempo item from midi file
    midi_obj = miditoolkit.midi.parser.MidiFile(midi_path)
    # tempo
    tempo_items = []
    for tempo in midi_obj.tempo_changes:
        tempo_items.append(Item(
            name='Tempo',
            start=tempo.time,
            end=None,
            velocity=None,
            pitch=int(tempo.tempo),
            Type=-1,
            inst_voice='Tempo'))
    tempo_items.sort(key=lambda x: x.start)

    # expand to all beat
    max_tick = tempo_items[-1].start
    existing_ticks = {item.start: item.pitch for item in tempo_items}
    wanted_ticks = np.arange(0, max_tick+1, DEFAULT_RESOLUTION)
    output = []
    for tick in wanted_ticks:
        if tick in existing_ticks:
            output.append(Item(
                name='Tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=existing_ticks[tick],
                Type=-1,
                inst_voice='Tempo'))
                
        else:
            output.append(Item(
                name='Tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=output[-1].pitch,
                Type=-1,
                inst_voice='Tempo'))
    tempo_items = output
    return note_items, tempo_items


# read notes and tempo changes from midi (assume there is only one track)
def read_items(file_path):
    midi_obj = miditoolkit.midi.parser.MidiFile(file_path)
    # note
    note_items = []
    num_of_instr = len(midi_obj.instruments) 
    
    for i in range(num_of_instr):
        notes = midi_obj.instruments[i].notes
        notes.sort(key=lambda x: (x.start, x.pitch))

        for note in notes:
            note_items.append(Item(
                name='Note',
                start=note.start, 
                end=note.end, 
                velocity=note.velocity, 
                pitch=note.pitch,
                Type=i))
                
    note_items.sort(key=lambda x: x.start)
    
    # tempo
    tempo_items = []
    for tempo in midi_obj.tempo_changes:
        tempo_items.append(Item(
            name='Tempo',
            start=tempo.time,
            end=None,
            velocity=None,
            pitch=int(tempo.tempo),
            Type=-1))
    tempo_items.sort(key=lambda x: x.start)

    # expand to all beat
    max_tick = tempo_items[-1].start
    existing_ticks = {item.start: item.pitch for item in tempo_items}
    wanted_ticks = np.arange(0, max_tick+1, DEFAULT_RESOLUTION)
    output = []
    for tick in wanted_ticks:
        if tick in existing_ticks:
            output.append(Item(
                name='Tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=existing_ticks[tick],
                Type=-1))
        else:
            output.append(Item(
                name='Tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=output[-1].pitch,
                Type=-1))
    tempo_items = output
    return note_items, tempo_items



class Event(object):
    def __init__(self, name, time, value, text, Type, inst_voice, texture_label=-1):
        self.name = name
        self.time = time
        self.value = value
        self.text = text
        self.Type = Type
        self.inst_voice = inst_voice
        self.texture_label = texture_label
    def __repr__(self):
        return 'Event(name={}, time={}, value={}, text={}, Type={}, inst_voice={}, texture_label={})'.format(
            self.name, self.time, self.value, self.text, self.Type, self.inst_voice, self.texture_label)


def item2event(groups, task):
    events = []
    n_downbeat = 0
    for i in range(len(groups)):
        if 'Note' not in [item.name for item in groups[i][1:-1]]:
            continue
        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        new_bar = True
        
        for item in groups[i][1:-1]:
            if item.name != 'Note':
                continue
            note_tuple = []

            # Bar
            if new_bar:
                BarValue = 'New' 
                new_bar = False
            else:
                BarValue = "Continue"
            note_tuple.append(Event(
                name='Bar',
                time=None, 
                value=BarValue,
                text='{}'.format(n_downbeat),
                Type=-1,
                inst_voice=item.inst_voice,
                texture_label=item.texture_label))

            # Position
            flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
            index = np.argmin(abs(flags-item.start))
            note_tuple.append(Event(
                name='Position', 
                time=item.start,
                value='{}/{}'.format(index+1, DEFAULT_FRACTION),
                text='{}'.format(item.start),
                Type=-1,
                inst_voice=item.inst_voice,
                texture_label=item.texture_label))

            
            # Pitch
            velocity_index = np.searchsorted(DEFAULT_VELOCITY_BINS, item.velocity, side='right') - 1

            if task == 'melody' or task == 'texture':
                pitchType = item.Type  # track
            elif task == 'velocity':
                pitchType = velocity_index
            else:
                pitchType = -1
                
            note_tuple.append(Event(
                name='Pitch',
                time=item.start, 
                value=item.pitch,
                text='{}'.format(item.pitch),
                Type=pitchType,
                inst_voice=item.inst_voice,
                texture_label=item.texture_label))

            # Duration
            duration = item.end - item.start
            index = np.argmin(abs(DEFAULT_DURATION_BINS-duration))
            note_tuple.append(Event(
                name='Duration',
                time=item.start,
                value=index,
                text='{}/{}'.format(duration, DEFAULT_DURATION_BINS[index]),
                Type=-1,
                inst_voice=item.inst_voice,
                texture_label=item.texture_label))
            # print(item.inst_voice)
            events.append(note_tuple)
    return events


def quantize_items(items, ticks=120):
    grids = np.arange(0, items[-1].start, ticks, dtype=int)
    # process
    for item in items:
        index = np.argmin(abs(grids - item.start))
        shift = grids[index] - item.start
        item.start += shift
        item.end += shift
    return items      

# group_items 函數的主要作用是將音符和節拍（Item 類型的事件）組織成多個小組，
# 每個小組包含在一個條件範圍內的音符。具體來說，它將音符和節拍按照時間範圍進行
# 分組，這樣便於後續的處理和事件轉換。
def group_items(items, max_time, ticks_per_bar=DEFAULT_RESOLUTION*4):
    items.sort(key=lambda x: x.start)
    downbeats = np.arange(0, max_time+ticks_per_bar, ticks_per_bar)
    groups = []
    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
        insiders = []
        for item in items:
            if (item.start >= db1) and (item.start < db2):
                insiders.append(item)
        overall = [db1] + insiders + [db2]
        groups.append(overall)
    return groups



class CP(object):
    def __init__(self, dict):
        # load dictionary
        self.event2word, self.word2event = pickle.load(open(dict, 'rb'))
        # pad word: ['Bar <PAD>', 'Position <PAD>', 'Pitch <PAD>', 'Duration <PAD>']
        self.pad_word = [self.event2word[etype]['%s <PAD>' % etype] for etype in self.event2word]

    def extract_events_t(self, task, ds, piece_name, midi_path):
        note_items, tempo_items = read_items_from_note_csv(ds, piece_name, midi_path)
        if len(note_items) == 0:   # if the midi contains nothing
            return None
        note_items = quantize_items(note_items)
        max_time = note_items[-1].end
        items = tempo_items + note_items
        
        groups = group_items(items, max_time)
        events = item2event(groups, task)
        return events

    def extract_events(self, input_path, task):
        note_items, tempo_items = read_items(input_path)
        if len(note_items) == 0:   # if the midi contains nothing
            return None
        note_items = quantize_items(note_items)
        max_time = note_items[-1].end
        items = tempo_items + note_items
        
        groups = group_items(items, max_time)
        events = item2event(groups, task)
        return events

    def padding(self, data, max_len, ans, task):
        pad_len = max_len - len(data)
        for _ in range(pad_len):
            if not ans:
                data.append(self.pad_word)
            elif task=='texture':
                data.append([0,0])
            else:
                data.append(0)

        return data

    def prepare_data(self, piece_names, task, max_len, ds):
        all_words, all_ys = [], []

        for piece_name in tqdm(piece_names):
            if ds=='orchestration':
                midi_path = os.path.join('dataset', 'scores_midi', piece_name+'.mid')
            elif ds=='s3':
                midi_path = os.path.join('dataset_s3', piece_name, 'sheet.mid')
            else:
                raise ValueError(f"Invalid dataset {ds}, should be 'orchestration' or 's3'. ")
            # extract events
            events = self.extract_events_t(task, ds, piece_name, midi_path)
            # events = self.extract_events(path, task)
            if not events:  # if midi contains nothing
                print(f'skip {piece_name} because it is empty')
                continue
            # events to words
            words, ys = [], []
            for note_tuple in events:
                nts, to_class, texture_label = [], -1, -1
                for e in note_tuple:
                    e_text = '{} {}'.format(e.name, e.value)
                    if e_text not in self.event2word[e.name].keys():
                        nts.append(self.event2word[e.name][f'{e.name} <PAD>'])
                    else:
                        nts.append(self.event2word[e.name][e_text])
                    if e.name == 'Pitch':  # if melody, Pitch's type will record track #
                        to_class = e.Type
                        texture_label = e.texture_label
                words.append(nts)
                if task == 'melody' or task == 'velocity':  # if melody, let track # start at 1
                    ys.append(to_class+1)
                elif task == 'texture':
                    ys.append([to_class, texture_label])  # 1~8

            # slice to chunks so that max length = max_len (default: 512)
            slice_words, slice_ys = [], []
            for i in range(0, len(words), max_len):
                slice_words.append(words[i:i+max_len])
                # if task == "composer":
                #     name = path.split('/')[-2]
                #     slice_ys.append(Composer[name])
                # elif task == "emotion":
                #     name = path.split('/')[-1].split('_')[0]
                #     slice_ys.append(Emotion[name])
                # else:
                slice_ys.append(ys[i:i+max_len])
            
            # padding or drop
            # drop only when the task is 'composer' and the data length < max_len//2
            if len(slice_words[-1]) < max_len:
                if task == 'composer' and len(slice_words[-1]) < max_len//2:
                    slice_words.pop()
                    slice_ys.pop()
                else:
                    slice_words[-1] = self.padding(slice_words[-1], max_len, ans=False, task=task)

            if (task == 'melody' or task == 'velocity') and len(slice_ys[-1]) < max_len:
                slice_ys[-1] = self.padding(slice_ys[-1], max_len, ans=True, task=task)
            elif task == 'texture' and len(slice_ys[-1]) < max_len:
                slice_ys[-1] = self.padding(slice_ys[-1], max_len, ans=True, task=task)
            
            all_words = all_words + slice_words
            all_ys = all_ys + slice_ys
        
        all_words = np.array(all_words)
        all_ys = np.array(all_ys)
        

        return all_words, all_ys


class FinetuneDataset(Dataset):
    """
    Expected data shape: (data_num, data_len)
    """
    def __init__(self, X, y, dataset=None, task='melody'):
        self.dataset = dataset
        self.label_dict = {
            # 0: {'label': [-1, -1, -1], 'meaning': '<PAD>ForNoLabel'},
            0: {'label': [0, 0, 0], 'meaning': 'None'},
            1: {'label': [1, 0, 0], 'meaning': 'Melody'},
            2: {'label': [0, 1, 0], 'meaning': 'Rhythm'},
            3: {'label': [1, 1, 0], 'meaning': 'Melody+Rhythm'},
            4: {'label': [0, 0, 1], 'meaning': 'Harmony'},
            5: {'label': [1, 0, 1], 'meaning': 'Melody+Harmony'},
            6: {'label': [0, 1, 1], 'meaning': 'Rhythm+Harmony'},
            7: {'label': [1, 1, 1], 'meaning': 'All'}
        }

        self.label_list = [self.label_dict[i]['meaning'] for i in range(8)]
        self.task = task
        self.texture2mel = {
            0: 0,
            1: 0,
            2: 1,
            3: 3,
            4: 2,
            5: 3,
            6: 2,
            7: 3,
            8: 2,
        }
        self.data = X 
        self.label = y
    
    def get_track(self, index):
        return torch.tensor(self.label[index][:,0], dtype=torch.long)

    def __len__(self):
        return(len(self.data))
    
    def __getitem__(self, index):
        if self.dataset is None:
            return torch.tensor(self.data[index]), torch.tensor(self.label[index])
        elif self.task=='melody':
            return torch.tensor(self.data[index], dtype=torch.long),\
                torch.tensor([self.texture2mel[y] for y in self.label[index][:,1]], dtype=torch.long)
        else:
            return torch.tensor(self.data[index], dtype=torch.long),\
                torch.tensor(self.label[index][:,1], dtype=torch.long)

class TokenClassification(nn.Module):
    def __init__(self, midibert, class_num, hs):
        super().__init__()
        
        self.midibert = midibert
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hs, 256),
            nn.ReLU(),
            nn.Linear(256, class_num)
        )
    
    def forward(self, y, attn, layer):
        # feed to bert 
        y = self.midibert(y, attn, output_hidden_states=True)
        #y = y.last_hidden_state         # (batch_size, seq_len, 768)
        y = y.hidden_states[layer]
        return self.classifier(y)


class SequenceClassification(nn.Module):
    def __init__(self, midibert, class_num, hs, da=128, r=4):
        super(SequenceClassification, self).__init__()
        self.midibert = midibert
        self.attention = SelfAttention(hs, da, r)
        self.classifier = nn.Sequential(
            nn.Linear(hs*r, 256),
            nn.ReLU(),
            nn.Linear(256, class_num)
        )

    def forward(self, x, attn, layer):             # x: (batch, 512, 4)
        x = self.midibert(x, attn, output_hidden_states=True)   # (batch, 512, 768)
        #y = y.last_hidden_state         # (batch_size, seq_len, 768)
        x = x.hidden_states[layer]
        attn_mat = self.attention(x)        # attn_mat: (batch, r, 512)
        m = torch.bmm(attn_mat, x)          # m: (batch, r, 768)
        flatten = m.view(m.size()[0], -1)   # flatten: (batch, r*768)
        res = self.classifier(flatten)      # res: (batch, class_num)
        return res


class SelfAttention(nn.Module):
    def __init__(self, input_dim, da, r):
        '''
        Args:
            input_dim (int): batch, seq, input_dim
            da (int): number of features in hidden layer from self-attn
            r (int): number of aspects of self-attn
        '''
        super(SelfAttention, self).__init__()
        self.ws1 = nn.Linear(input_dim, da, bias=False)
        self.ws2 = nn.Linear(da, r, bias=False)

    def forward(self, h):
        attn_mat = F.softmax(self.ws2(torch.tanh(self.ws1(h))), dim=1)
        attn_mat = attn_mat.permute(0,2,1)
        return attn_mat
    




def three_bool_to_eight_class(three_bool):
    if three_bool.shape[-1]!=3: 
        raise Exception('Invalid bool array to convert to classes')
    cls = three_bool[:,0]*1 + three_bool[:,1]*2 + three_bool[:,2]*4
    return cls

class FinetuneTrainer:
    def __init__(self, midibert, train_dataloader, valid_dataloader, test_dataloader, layer, 
                lr, class_num, hs, testset_shape, cpu, 
                cuda_devices=None, model=None, SeqClass=False, freeze=False, task='',
                label_num=4, label_dict=None, class_weight=None,
                do_save_current_stage=True, return_perf=False, save_at='', early_stopping=None,
                epoch_gap=5, epoch_beg=0, epoch_end=20,
                stage=None, piece_name=None, skip_first=None,is_debug=False, reset_patient=True,
                max_patient=1000, threshold=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() and not cpu else 'cpu')
        print('   device:',self.device)
        self.midibert = midibert
        self.SeqClass = SeqClass
        self.layer = layer
        self.task = task
        self.label_num = label_num
        self.label_dict=label_dict
        self.class_weight=class_weight
        self.performance_list = []
        self.do_save_current_stage = do_save_current_stage
        self.return_performance = return_perf
        self.save_at = save_at
        self.epoch = 0
        self.early_stopping = early_stopping 
        self.epoch_gap = epoch_gap
        self.epoch_beg = epoch_beg
        self.epoch_end = epoch_end
        self.stage = stage
        self.piece_name = piece_name
        self.skip_first=skip_first
        self.performance_df_cols = PERFORMANCE_DF_COLS_MIDIBERT_TEX if self.task=='texture' else PERFORMANCE_DF_COLS_MIDIBERT_MEL
        self.is_debug=is_debug
        self.reset_patient = reset_patient
        self.max_patient = max_patient
        self.threshold = threshold
        if model != None:    # load model
            print('load a fine-tuned model')
            self.model = model.to(self.device)
        else:
            print('init a fine-tune model, sequence-level task?', SeqClass)
            if SeqClass:
                self.model = SequenceClassification(self.midibert, class_num, hs).to(self.device)
            else:
                self.model = TokenClassification(self.midibert, class_num, hs).to(self.device)

        # freeze midibert params
        if freeze:
            for name, param in self.model.named_parameters():
                if 'midibert.bert' in name:
                        param.requires_grad = False
                print(name, param.requires_grad)


        if torch.cuda.device_count() > 1 and not cpu:
            print("Use %d GPUS" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.train_data = train_dataloader
        self.valid_data = valid_dataloader
        self.test_data = test_dataloader
        
        self.optim = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        if self.task=='texture':
            # self.loss_func = nn.BCEWithLogitsLoss(ignore_index=0, reduction='none')
            self.loss_func = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.loss_func = nn.CrossEntropyLoss(reduction='none')

        self.testset_shape = testset_shape
    
    def compute_loss(self, predict, target, loss_mask, seq):  # yhat, y, attn
        if self.task=='texture':
            target_decoded = torch.stack([
                (target & 1) > 0,         # 第 0 維（二進位最低位）
                (target & 2) > 0,         # 第 1 維
                (target & 4) > 0          # 第 2 維
            ], dim=1).float()  
            loss = self.loss_func(predict, target_decoded)
            loss = loss * loss_mask.unsqueeze(1)
            # loss = self.loss_func(predict, target * loss_mask)
        else:
            loss = self.loss_func(predict, target)
        
        if self.task=='texture':
            loss = torch.sum(loss) / torch.sum(loss_mask) / target.shape[-1]  # /3
        elif not seq:
            loss = loss * loss_mask
            loss = torch.sum(loss) / torch.sum(loss_mask)
        else:
            loss = torch.sum(loss)/loss.shape[0]
        return loss

    def train(self):
        if self.epoch < 3:  # 前3個 epoch 再 freeze
            for name, param in self.model.named_parameters():
                if 'midibert.bert' in name:
                    param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = True
                
        self.model.train()
        self.stage = 'train'
        print(f"{self.epoch=} | {self.stage=}")
        return self.iteration(self.train_data, 0, self.SeqClass)

    def valid(self):
        self.model.eval()
        self.stage = 'valid'
        print(f"{self.epoch=} | {self.stage=}")
        return self.iteration(self.valid_data, 1, self.SeqClass)

    def test(self):
        self.model.eval()
        self.stage = 'test'
        print(f"{self.epoch=} | {self.stage=}")
        return self.iteration(self.test_data, 2, self.SeqClass)
    

    def inference(self, x, y):
        self.model.eval()
        seq = self.SeqClass
        total_acc, total_cnt, total_loss = 0, 0, 0
        cnt = 0
        # record results:
        # for overall accuracy
        correct_label_wise = 0.0
        correct_data_wise = 0.0
        # for precision and recall of each label
        pred_label_count = np.zeros(3, dtype=np.float32)
        gt_label_count = np.zeros(3, dtype=np.float32)
        TP = np.zeros(3, dtype=np.float32)
        # for predicted labels analysis including confusion matrix
        predicted_class = []
        gt_class = []
        x, y = x.to(self.device), y.to(self.device)     # seq: (batch, 512, 4), (batch) / token: , (batch, 512)

        # avoid attend to pad word
        if self.task=='texture':
            attn = (y.sum(axis=-1) != -3).float().to(self.device)
        elif not seq:
            attn = (y != 0).float().to(self.device)   # (batch,512)
        else:   
            attn = torch.ones((1, 512)).to(self.device)     # attend each of them

        y_hat = self.model.forward(x, attn, self.layer)     # seq: (batch, class_num) / token: (batch, 512, class_num)

        # get multi-label result
        if self.task=='texture':
            output = (nn.Sigmoid()(y_hat).cpu().detach().numpy() > 0.5)
        # get the most likely choice with max
        else:
            output = np.argmax(y_hat.cpu().detach().numpy(), axis=-1)
        output = torch.from_numpy(output).to(self.device)
        print(f"{output[:3, :5]=}")
        if not seq:
            acc = torch.sum((y == output).float() * attn)
            total_acc += acc
            total_cnt += torch.sum(attn).item()
        else:
            acc = torch.sum((y == output).float())
            total_acc += acc
            total_cnt += y.shape[0]

        # calculate losses
        if not seq:
            y_hat = y_hat.permute(0,2,1)
        loss = self.compute_loss(y_hat, y, attn, seq)
        total_loss += loss.item()

        return round(loss.item(),4), round(total_acc.item()/total_cnt,4), output


    def iteration(self, training_data, mode, seq):
        pbar = tqdm(training_data, disable=False)

        total_acc, total_cnt, total_loss = 0, 0, 0

        if mode == 2: # testing
            all_output = torch.empty(self.testset_shape)
            cnt = 0

        performance = {
            'epoch': self.epoch,
            'stage': self.stage,
            'piece_name': self.piece_name,
            'loss': 0, 
            'valid_bar_num': 0,
            # texture
            'correct_texture': 0,
            'count_texture': 0,
            'texture_obj': np.zeros((self.label_num, 4)),  # TP, FP, FN, TN
            'texture_obj_aprf': np.zeros((self.label_num,4)),
            'cm_texture': np.zeros((self.label_num, self.label_num)),
        }

        for x, y in pbar:  # (batch, 512, 768)
            batch = x.shape[0]
            x, y = x.to(self.device), y.to(self.device)     # seq: (batch, 512, 4), (batch) / token: , (batch, 512)
            # print(np.unique(y.detach().cpu().numpy()))
            # print(y.shape)
            # print(y.sum(axis=-1).shape)
            # print(y.sum(axis=-1))
            # avoid attend to pad word
            # if self.task=='texture':
                # attn = (y.sum(axis=-1) != -3).float().to(self.device)
            if not seq:
            # elif not seq:
                attn = (y != 0).float().to(self.device)   # (batch,512)
            else:   
                attn = torch.ones((batch, 512)).to(self.device)     # attend each of them
            # print(f'{attn.shape=}')
            y_hat = self.model.forward(x, attn, self.layer)     # seq: (batch, class_num) / token: (batch, 512, class_num)
            # print(f'{y_hat.shape=}')

            # get multi-label result
            if self.task=='texture':
                output = (nn.Sigmoid()(y_hat).cpu().detach().numpy() > 0.5)
            # get the most likely choice with max
            else:
                output = np.argmax(y_hat.cpu().detach().numpy(), axis=-1)
            output = torch.from_numpy(output).to(self.device)
            if mode == 2:
                if self.task=='texture':
                    tmp = (
                        output[:, :, 0] * 1 +
                        output[:, :, 1] * 2 +
                        output[:, :, 2] * 4
                    )
                    all_output[cnt : cnt+tmp.shape[0]] = tmp
                else:
                    all_output[cnt : cnt+output.shape[0]] = output
                cnt += batch

            # accuracy
            if not seq:
                if self.task=='texture':
                    y_pred = output > self.threshold
                    # print(y_pred.shape)
                    y_pred = y_pred[:,:,0] + y_pred[:,:,1]*2 + y_pred[:,:,2]*4
                    # print(y.shape, y_pred.shape)
                    acc = torch.sum((y == y_pred).float() * attn)
                else:
                    acc = torch.sum((y == output).float() * attn)
                total_acc += acc
                total_cnt += torch.sum(attn).item()
            else:
                acc = torch.sum((y == output).float())
                total_acc += acc
                total_cnt += y.shape[0]

            # calculate losses
            if not seq:
                y_hat = y_hat.permute(0,2,1)
            # print(f"{y_hat.shape=}, {y.shape=}, {attn.shape=}, {seq=}")

            loss = self.compute_loss(y_hat, y, attn, seq)
            total_loss += loss.item()

            # udpate only in train
            if mode == 0:
                self.model.zero_grad()
                loss.backward()
                self.optim.step()
            
            performance = self.record_performance_for_a_batch(
                y,
                output,
                performance
            )
            if self.is_debug:
                break

        performance = self.get_performance(performance)
        self.performance = performance

        # if mode == 2:
        #     return round(total_loss/len(training_data),4), round(total_acc.item()/total_cnt,4), all_output
        # My
        for i in range(self.label_num):
            performance[f'class_{i}_count'] = self.class_count[str(i)]
            performance[f'class_{i}_weight'] = self.class_weight[str(i)]

        if self.do_save_current_stage:
            # save result and model
            self.performance_list.append([
                performance[_] for _ in self.performance_df_cols
            ])
            if self.epoch%self.epoch_gap==0 or self.epoch==self.epoch_end-1:
                self.save_current_stage(performance)
            if self.epoch==self.epoch_end-1:
                torch.save(self.model.state_dict(), os.path.join(self.save_at, 'model', f"epoch{str(performance['epoch'])}.pt")) 


        if self.return_performance:
            if mode==2:
                return performance['loss'], performance['accuracy'], performance, all_output
            return performance['loss'], performance['accuracy'], performance
        else:
            if mode==2:
                return performance['loss'], performance['accuracy'], all_output
            return performance['loss'], performance['accuracy']

        # return round(total_loss/len(training_data),4), round(total_acc.item()/total_cnt,4)
    
    def record_performance_for_a_batch(
            self, 
            y_true,
            y_pred,
            performance
        ):
        if self.task=='texture':
            # y_pred_bin = (torch.sigmoid(y_pred) > self.threshold) 
            y_pred_8 = (
                y_pred[:, :, 0] * 1 +
                y_pred[:, :, 1] * 2 +
                y_pred[:, :, 2] * 4
            )  # shape: [B, T]
            y_pred_8 = y_pred_8.detach().cpu().numpy().astype(int).reshape(-1)
            y_true_8 = y_true.detach().cpu().numpy().astype(int).reshape(-1)
        else:
            y_true_8 = y_true.detach().cpu().numpy().astype(int).reshape(-1)
            y_pred_8 = y_pred.detach().cpu().numpy().astype(int).reshape(-1)
        # print(f"{y_pred_8.shape=}, {y_true_8.shape=}, {self.label_num=}")
        # print('\n\n y true')
        # print(y_true_8[:5])
        # print('\n\n y pred')
        # print(y_pred_8[:5])
        cm = confusion_matrix(
            y_true_8,
            y_pred_8,
            labels=list(range(self.label_num))
            )
        for _index in range(self.label_num):
            tp = cm[_index, _index]
            fp = cm[:, _index].sum() - tp
            fn = cm[_index, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            performance['texture_obj'][_index] += np.array([tp, fp, fn, tn])
        performance['cm_texture'] += cm
        v, count = np.unique(y_true_8, return_counts=True)
        performance['valid_bar_num'] += len(y_true_8)
        # performance['valid_bar_num'] += count[1:].sum() if self.skip_first else count[:].sum()

        return performance

    def save_checkpoint(self, epoch, train_acc, valid_acc, 
                        valid_loss, train_loss, is_best, filename):
        state = {
            'epoch': epoch + 1,
            # 'state_dict': self.model.module.state_dict(), 
            'state_dict': self.model.state_dict(), 
            'valid_acc': valid_acc,
            'valid_loss': valid_loss,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'optimizer' : self.optim.state_dict()
        }
        torch.save(state, filename)

        best_mdl = filename.split('.')[0]+'_best.ckpt'
        
        if is_best:
            shutil.copyfile(filename, best_mdl)

    def get_performance(self, performance):
        label_num = self.label_num
        label_dict = self.label_dict
        class_weight = self.class_weight

        performance['correct_texture'] = 0
        print(f"{' '*20} | {'acc':>6} | {'prec':>6} | {'recall':>6} | {'f1':>6}")
        # 8 classes
        print("--"*30)
        for _ in range(self.label_num):
            tp, fp, fn, tn = performance['texture_obj'][_]
            t_acc = (tp+tn) / (tp+fp+fn+tn)  # data-wise acc
            t_precision = tp / (tp+fp) if (tp+fp)>0 else None
            t_recall = tp / (tp+fn) if (tp+fn)>0 else None
            t_f1 =  None if ((t_precision is None) or (t_recall is None)) else 2*t_precision*t_recall / (t_precision + t_recall)
            performance['texture_obj_aprf'][_] += [t_acc, t_precision if t_precision is not None else 0, t_recall if t_recall is not None else 0, t_f1 if t_f1 is not None else 0]
            # support: The number of occurrences of each label in y_true
            label_name = label_dict[_]['meaning']
            performance.update({
                f'acc_{label_name}': t_acc,
                f'precision_{label_name}': t_precision, 
                f'recall_{label_name}': t_recall,
                f'f1_{label_name}': t_f1,
                f'cmtx_{label_name}': (tp, fp, fn, tn),
            })
            print(f"{label_dict[_]['meaning']:<20} | {t_acc if t_acc is not None else -1:>6.2f} | {t_precision if t_precision is not None else -1:>6.2f} | {t_recall if t_recall is not None else -1:>6.2f} | {t_f1 if t_f1 is not None else -1:>6.2f}")
            # claculate A for all 
        tp, fp, fn, tn = performance['texture_obj'][1:,:].sum(axis=0) if self.skip_first else performance['texture_obj'][:,:].sum(axis=0)
        # accuracy = tp / performance['valid_bar_num']
        if self.skip_first:
            accuracy = performance['texture_obj'][1:,0].sum(axis=0) / performance['cm_texture'][1:,1:].sum()
        else:
            accuracy = performance['texture_obj'][:,0].sum(axis=0) / performance['cm_texture'][:,:].sum()

        performance['accuracy'] = accuracy
        print(f"{'average':<20} | {accuracy:>6.2f} | {performance['valid_bar_num']}")

        # micro average
        tp, fp, fn, tn = performance['texture_obj'][:,:].sum(axis=0)  # 可計算 None 列，因為已經用 msk 遮掉無音符的區域了
        t_acc = (tp+tn) / (tp+fp+fn+tn)  # data-wise acc
        t_precision = tp / (tp+fp) if (tp+fp)>0 else None
        t_recall = tp / (tp+fn) if (tp+fn)>0 else None
        t_f1 =  None if ((t_precision is None) or (t_recall is None)) else 2*t_precision*t_recall / (t_precision + t_recall)
        performance.update({
            f'texture_acc_micro': t_acc,
            f'texture_precision_micro': t_precision, 
            f'texture_recall_micro': t_recall,
            f'texture_f1_micro': t_f1,
            f'texture_cmtx_micro': (tp, fp, fn, tn),
        })
        print('--'*30)
        print(f"{'micro-average':<20} | {accuracy:>6.2f} | {t_precision if t_precision is not None else -1:>6.2f} | {t_recall if t_recall is not None else -1:>6.2f} | {t_f1 if t_f1 is not None else -1:>6.2f}")
        # macro-average: each class has same weight
        macro_7 = performance['texture_obj_aprf'][1:,:].sum(axis=0) / (label_num-1)
        macro_8 = performance['texture_obj_aprf'][:,:].sum(axis=0) / label_num
        class_weight_int = [class_weight[str(k)] for k in range(label_num)]
        tp = sum(performance['texture_obj'][:,0] * class_weight_int)
        fp = sum(performance['texture_obj'][:,1] * class_weight_int)
        fn = sum(performance['texture_obj'][:,2] * class_weight_int)
        tn = sum(performance['texture_obj'][:,3] * class_weight_int)
        t_acc = (tp+tn) / (tp+fp+fn+tn) 
        t_precision = tp / (tp+fp) if (tp+fp)>0 else 0
        t_recall = tp / (tp+fn) if (tp+fn)>0 else 0
        t_f1 =  0 if ((t_precision == 0) or (t_recall == 0)) else 2*t_precision*t_recall / (t_precision + t_recall)
        class_weight_int = [class_weight[str(k)] for k in range(1,label_num)]
        tp = sum(performance['texture_obj'][1:,0] * class_weight_int)
        fp = sum(performance['texture_obj'][1:,1] * class_weight_int)
        fn = sum(performance['texture_obj'][1:,2] * class_weight_int)
        tn = sum(performance['texture_obj'][1:,3] * class_weight_int)
        t_acc3 = (tp+tn) / (tp+fp+fn+tn) 
        t_precision3 = tp / (tp+fp) if (tp+fp)>0 else 0
        t_recall3 = tp / (tp+fn) if (tp+fn)>0 else 0
        t_f13 =  0 if ((t_precision == 0) or (t_recall == 0)) else 2*t_precision*t_recall / (t_precision + t_recall)
                
        performance.update({
            # divide to label_num-1
            f'texture_acc_macro_7': macro_7[0],
            f'texture_precision_macro_7': macro_7[1], 
            f'texture_recall_macro_7': macro_7[2],
            f'texture_f1_macro_7': 2*macro_7[1]*macro_7[2]/(macro_7[1]+macro_7[2]) if macro_7[1]+macro_7[2]>0 else None,
            # divide to label_num
            f'texture_acc_macro_8': macro_8[0],
            f'texture_precision_macro_8': macro_8[1],
            f'texture_recall_macro_8': macro_8[2],
            f'texture_f1_macro_8': 2*macro_8[1]*macro_8[2]/(macro_8[1]+macro_8[2]) if macro_8[1]+macro_8[2]>0 else None,
            # calculate by weight
            f'texture_acc_macro_weight': t_acc,
            f'texture_precision_macro_weight': t_precision,
            f'texture_recall_macro_weight': t_recall,
            f'texture_f1_macro_weight': t_f1,
            # calculate by weight (skip first)
            f'texture_acc_macro_weight3': t_acc3,
            f'texture_precision_macro_weight3': t_precision3,
            f'texture_recall_macro_weight3': t_recall3,
            f'texture_f1_macro_weight3': t_f13,
        })
        print(f"{'macro-average 7':<20} | {performance['texture_acc_macro_7']:>6.2f} | {performance['texture_precision_macro_7']:>6.2f} | {performance['texture_recall_macro_7']:>6.2f} | {performance['texture_f1_macro_7'] if performance['texture_f1_macro_7'] is not None else -1:>6.2f}")
        print(f"{'macro-average 8':<20} | {performance['texture_acc_macro_8']:>6.2f} | {performance['texture_precision_macro_8']:>6.2f} | {performance['texture_recall_macro_8']:>6.2f} | {performance['texture_f1_macro_8'] if performance['texture_f1_macro_8'] is not None else -1:>6.2f}")
        print(f"{'macro-average weight':<20} | {performance['texture_acc_macro_weight']:>6.2f} | {performance['texture_precision_macro_weight']:>6.2f} | {performance['texture_recall_macro_weight']:>6.2f} | {performance['texture_f1_macro_weight'] if performance['texture_f1_macro_weight'] is not None else -1:>6.2f}")
        print(f"{'macro-average weight3':<20} | {performance['texture_acc_macro_weight3']:>6.2f} | {performance['texture_precision_macro_weight3']:>6.2f} | {performance['texture_recall_macro_weight3']:>6.2f} | {performance['texture_f1_macro_weight3'] if performance['texture_f1_macro_weight3'] is not None else -1:>6.2f}")
        
        if self.task=='texture':
            # 3 classes
            cmtx = performance['cm_texture']
            # mel: gt 1,3,5,7
            index_for_true = [1,3,5,7]
            index_for_false = [0,2,4,6]
            tp = cmtx[index_for_true][:,index_for_true].sum()
            fp = cmtx[index_for_false][:,index_for_true].sum()
            fn = cmtx[index_for_true][:,index_for_false].sum()
            tn = cmtx[index_for_false][:,index_for_false].sum()
            performance['correct_texture'] = tp+tn
            t_acc = (tp+tn) / (tp+fp+fn+tn)  # data-wise acc
            t_precision = tp / (tp+fp) if (tp+fp)>0 else None
            t_recall = tp / (tp+fn) if (tp+fn)>0 else None
            t_f1 =  None if ((t_precision is None) or (t_recall is None)) else 2*t_precision*t_recall / (t_precision + t_recall)
            performance.update({
                'mel_acc': t_acc,
                'mel_precision': t_precision, 
                'mel_recall': t_recall,
                'mel_f1': t_f1,
                'mel_cmtx': (tp, fp, fn, tn),
            })
            # rhythm: gt 2,3,6,7
            index_for_true = [2,3,6,7]
            index_for_false = [0,1,4,5]
            tp = cmtx[index_for_true][:,index_for_true].sum()
            fp = cmtx[index_for_false][:,index_for_true].sum()
            fn = cmtx[index_for_true][:,index_for_false].sum()
            tn = cmtx[index_for_false][:,index_for_false].sum()
            performance['correct_texture'] = tp+tn
            t_acc = (tp+tn) / (tp+fp+fn+tn)  # data-wise acc
            t_precision = tp / (tp+fp) if (tp+fp)>0 else None
            t_recall = tp / (tp+fn) if (tp+fn)>0 else None
            t_f1 =  None if ((t_precision is None) or (t_recall is None)) else 2*t_precision*t_recall / (t_precision + t_recall)
            performance.update({
                'rhythm_acc': t_acc,
                'rhythm_precision': t_precision, 
                'rhythm_recall': t_recall,
                'rhythm_f1': t_f1,
                'rhythm_cmtx': (tp, fp, fn, tn),
            })
            # harm: gt 4,5,6,7
            index_for_true = [4,5,6,7]
            index_for_false = [0,1,2,3]
            tp = cmtx[index_for_true][:,index_for_true].sum()
            fp = cmtx[index_for_false][:,index_for_true].sum()
            fn = cmtx[index_for_true][:,index_for_false].sum()
            tn = cmtx[index_for_false][:,index_for_false].sum()
            performance['correct_texture'] = tp+tn
            t_acc = (tp+tn) / (tp+fp+fn+tn)  # data-wise acc
            t_precision = tp / (tp+fp) if (tp+fp)>0 else None
            t_recall = tp / (tp+fn) if (tp+fn)>0 else None
            t_f1 =  None if ((t_precision is None) or (t_recall is None)) else 2*t_precision*t_recall / (t_precision + t_recall)
            performance.update({
                'harm_acc': t_acc,
                'harm_precision': t_precision, 
                'harm_recall': t_recall,
                'harm_f1': t_f1,
                'harm_cmtx': (tp, fp, fn, tn),
            })
            # label-wise accuracy
            label_wise_acc = performance['correct_texture'] / ((performance['valid_bar_num'])*3)
            performance.update({
                'label_wise_acc': label_wise_acc,
            })
        return performance

    def save_current_stage(self, performance):
        df = pd.DataFrame(self.performance_list, columns=self.performance_df_cols)
        df.to_csv(f'{self.save_at}/performance{self.epoch}.csv')
        # 8 class cm
        np.save(f"{self.save_at}/confusion_matrix/epoch{str(performance['epoch'])}_{performance['stage']}_texture.npy", performance['cm_texture'])
        performance['cm_texture_new'] = performance['cm_texture'] / performance['cm_texture'].sum(axis=1, keepdims=True)
        print_confusion_matrix(performance['cm_texture_new'], f"epoch{str(performance['epoch'])}_{performance['stage']}_texture",
                                f'{self.save_at}/fig', 
                                show=False, label_list=[self.label_dict[_]['meaning'] for _ in range(self.label_num)]
                                )
        # torch.save(self.model.state_dict(), os.path.join(self.save_at, 'model', f"epoch{str(performance['epoch'])}.pt")) 


    def check_early_stop(self, compared_with):
        if self.early_stopping['rule'] == 'min':
            if compared_with <= self.early_stopping['criteria']:
                self.early_stopping.update({'criteria': compared_with, 'beat_epoch': self.epoch})
                torch.save(self.model.state_dict(), f'{self.save_at}/model/epoch{str(self.epoch)}-best.pt') 
                if self.epoch%self.epoch_gap!=0 or self.epoch!=self.epoch_end:
                    self.save_current_stage(self.performance)
                if self.reset_patient:
                    self.early_stopping['patient'] = self.max_patient
            else:
                self.early_stopping['patient'] -= 1
        elif self.early_stopping['rule'] == 'max':
            if compared_with >= self.early_stopping['criteria']:
                self.early_stopping.update({'criteria': compared_with, 'beat_epoch': self.epoch})
                torch.save(self.model.state_dict(), f'{self.save_at}/model/epoch{str(self.epoch)}-best.pt') 
                if self.epoch%self.epoch_gap!=0 or self.epoch!=self.epoch_end:
                    self.save_current_stage(self.performance)
                if self.reset_patient:
                    self.early_stopping['patient'] = self.max_patient
            else:
                # torch.save(self.model.state_dict(), f'{self.save_at}/model/epoch{str(self.epoch)}.pt') 
                self.early_stopping['patient'] -= 1
        if self.early_stopping['patient'] == 0:
            print(f'early stopping at epoch {self.epoch} with {self.early_stopping}, best_epoch is {self.early_stopping["beat_epoch"]}')
            return True
        return False

def get_perf(y_true, y_pred, 
    output_dir, label_dict, label_num, skip_first=True,
    task='melody'):
    class_count = {str(_):0 for _ in range(label_num)}
    for _ in y_true:
        class_count[str(int(_))] += 1
    class_weight = {k: v/sum(class_count.values()) for k,v in class_count.items()}

    performance = {
        'texture_obj': np.zeros((label_num,4)),
        'valid_bar_num': 0,
        'texture_obj_aprf': np.zeros((label_num,4))
    }
    
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(label_num))
        )
    for _index in range(label_num):
        tp = cm[_index, _index]
        fp = cm[:, _index].sum() - tp
        fn = cm[_index, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        performance['texture_obj'][_index] += np.array([tp, fp, fn, tn])
    performance['cm_texture'] = cm
    v, count = np.unique(y_true, return_counts=True)
    performance['valid_bar_num'] = count[1:].sum() if skip_first else count[:].sum()

    performance['correct_texture'] = 0
    print(f"{' '*20} | {'acc':>6} | {'prec':>6} | {'recall':>6} | {'f1':>6}")
    # 8 classes
    print("--"*30)
    for _ in range(label_num):
        tp, fp, fn, tn = performance['texture_obj'][_]
        t_acc = (tp+tn) / (tp+fp+fn+tn)  # data-wise acc
        t_precision = tp / (tp+fp) if (tp+fp)>0 else None
        t_recall = tp / (tp+fn) if (tp+fn)>0 else None
        t_f1 =  None if ((t_precision is None) or (t_recall is None)) else 2*t_precision*t_recall / (t_precision + t_recall)
        performance['texture_obj_aprf'][_] += [t_acc, t_precision if t_precision is not None else 0, t_recall if t_recall is not None else 0, t_f1 if t_f1 is not None else 0]
        # support: The number of occurrences of each label in y_true
        print(f"{label_dict[_]['meaning']:<20} | {t_acc:>6.2f} | {t_precision if t_precision is not None else -1:>6.2f} | {t_recall if t_recall is not None else -1:>6.2f} | {t_f1 if t_f1 is not None else -1:>6.2f}")
        label_name = label_dict[_]['meaning']
        performance.update({
            f'acc_{label_name}': t_acc,
            f'precision_{label_name}': t_precision, 
            f'recall_{label_name}': t_recall,
            f'f1_{label_name}': t_f1,
            f'cmtx_{label_name}': (tp, fp, fn, tn),
        })
        # claculate A for all 
    tp, fp, fn, tn = performance['texture_obj'][1:,:].sum(axis=0) if skip_first else performance['texture_obj'][:,:].sum(axis=0)
    if skip_first:
        accuracy = performance['texture_obj'][1:,0].sum(axis=0) / performance['cm_texture'][1:,1:].sum()
    else:
        accuracy = performance['texture_obj'][:,0].sum(axis=0) / performance['cm_texture'][:,:].sum()

    # accuracy = tp / performance['valid_bar_num']
    performance['accuracy'] = accuracy
    print(f"{'average':<20} | {accuracy:>6.2f} | {performance['valid_bar_num']}")

    # micro average
    tp, fp, fn, tn = performance['texture_obj'][:,:].sum(axis=0)  # 可計算 None 列，因為已經用 msk 遮掉無音符的區域了
    t_acc = (tp+tn) / (tp+fp+fn+tn)  # data-wise acc
    t_precision = tp / (tp+fp) if (tp+fp)>0 else None
    t_recall = tp / (tp+fn) if (tp+fn)>0 else None
    t_f1 =  None if ((t_precision is None) or (t_recall is None)) else 2*t_precision*t_recall / (t_precision + t_recall)
    performance.update({
        f'texture_acc_micro': t_acc,
        f'texture_precision_micro': t_precision, 
        f'texture_recall_micro': t_recall,
        f'texture_f1_micro': t_f1,
        f'texture_cmtx_micro': (tp, fp, fn, tn),
    })
    print('--'*30)
    print(f"{'micro-average':<20} | {accuracy:>6.2f} | {t_precision if t_precision is not None else -1:>6.2f} | {t_recall if t_recall is not None else -1:>6.2f} | {t_f1 if t_f1 is not None else -1:>6.2f}")
    # macro-average: each class has same weight
    macro_7 = performance['texture_obj_aprf'][1:,:].sum(axis=0) / (label_num-1)
    macro_8 = performance['texture_obj_aprf'][:,:].sum(axis=0) / label_num
    class_weight_int = [class_weight[str(k)] for k in range(label_num)]
    tp = sum(performance['texture_obj'][:,0] * class_weight_int)
    fp = sum(performance['texture_obj'][:,1] * class_weight_int)
    fn = sum(performance['texture_obj'][:,2] * class_weight_int)
    tn = sum(performance['texture_obj'][:,3] * class_weight_int)
    t_acc = (tp+tn) / (tp+fp+fn+tn) 
    t_precision = tp / (tp+fp) if (tp+fp)>0 else 0
    t_recall = tp / (tp+fn) if (tp+fn)>0 else 0
    t_f1 =  0 if ((t_precision == 0) or (t_recall == 0)) else 2*t_precision*t_recall / (t_precision + t_recall)
    class_weight_int = [class_weight[str(k)] for k in range(1,label_num)]
    tp = sum(performance['texture_obj'][1:,0] * class_weight_int)
    fp = sum(performance['texture_obj'][1:,1] * class_weight_int)
    fn = sum(performance['texture_obj'][1:,2] * class_weight_int)
    tn = sum(performance['texture_obj'][1:,3] * class_weight_int)
    t_acc3 = (tp+tn) / (tp+fp+fn+tn) 
    t_precision3 = tp / (tp+fp) if (tp+fp)>0 else 0
    t_recall3 = tp / (tp+fn) if (tp+fn)>0 else 0
    t_f13 =  0 if ((t_precision == 0) or (t_recall == 0)) else 2*t_precision*t_recall / (t_precision + t_recall)
            
    performance.update({
        # divide to label_num-1
        f'texture_acc_macro_7': macro_7[0],
        f'texture_precision_macro_7': macro_7[1], 
        f'texture_recall_macro_7': macro_7[2],
        f'texture_f1_macro_7': 2*macro_7[1]*macro_7[2]/(macro_7[1]+macro_7[2]) if macro_7[1]+macro_7[2]>0 else None,
        # divide to label_num
        f'texture_acc_macro_8': macro_8[0],
        f'texture_precision_macro_8': macro_8[1],
        f'texture_recall_macro_8': macro_8[2],
        f'texture_f1_macro_8': 2*macro_8[1]*macro_8[2]/(macro_8[1]+macro_8[2]) if macro_8[1]+macro_8[2]>0 else None,
        # calculate by weight
        f'texture_acc_macro_weight': t_acc,
        f'texture_precision_macro_weight': t_precision,
        f'texture_recall_macro_weight': t_recall,
        f'texture_f1_macro_weight': t_f1,
        # calculate by weight (skip first)
        f'texture_acc_macro_weight3': t_acc3,
        f'texture_precision_macro_weight3': t_precision3,
        f'texture_recall_macro_weight3': t_recall3,
        f'texture_f1_macro_weight3': t_f13,
    })
    print(f"{'macro-average 7':<20} | {performance['texture_acc_macro_7']:>6.2f} | {performance['texture_precision_macro_7']:>6.2f} | {performance['texture_recall_macro_7']:>6.2f} | {performance['texture_f1_macro_7'] if performance['texture_f1_macro_7'] is not None else -1:>6.2f}")
    print(f"{'macro-average 8':<20} | {performance['texture_acc_macro_8']:>6.2f} | {performance['texture_precision_macro_8']:>6.2f} | {performance['texture_recall_macro_8']:>6.2f} | {performance['texture_f1_macro_8'] if performance['texture_f1_macro_8'] is not None else -1:>6.2f}")
    print(f"{'macro-average weight':<20} | {performance['texture_acc_macro_weight']:>6.2f} | {performance['texture_precision_macro_weight']:>6.2f} | {performance['texture_recall_macro_weight']:>6.2f} | {performance['texture_f1_macro_weight'] if performance['texture_f1_macro_weight'] is not None else -1:>6.2f}")
    print(f"{'macro-average weight3':<20} | {performance['texture_acc_macro_weight3']:>6.2f} | {performance['texture_precision_macro_weight3']:>6.2f} | {performance['texture_recall_macro_weight3']:>6.2f} | {performance['texture_f1_macro_weight3'] if performance['texture_f1_macro_weight3'] is not None else -1:>6.2f}")
    
    if task=='texture':
        # 3 classes
        cmtx = performance['cm_texture']
        # mel: gt 1,3,5,7
        index_for_true = [1,3,5,7]
        index_for_false = [0,2,4,6]
        tp = cmtx[index_for_true][:,index_for_true].sum()
        fp = cmtx[index_for_false][:,index_for_true].sum()
        fn = cmtx[index_for_true][:,index_for_false].sum()
        tn = cmtx[index_for_false][:,index_for_false].sum()
        performance['correct_texture'] = tp+tn
        t_acc = (tp+tn) / (tp+fp+fn+tn)  # data-wise acc
        t_precision = tp / (tp+fp) if (tp+fp)>0 else None
        t_recall = tp / (tp+fn) if (tp+fn)>0 else None
        t_f1 =  None if ((t_precision is None) or (t_recall is None)) else 2*t_precision*t_recall / (t_precision + t_recall)
        performance.update({
            'mel_acc': t_acc,
            'mel_precision': t_precision, 
            'mel_recall': t_recall,
            'mel_f1': t_f1,
            'mel_cmtx': (tp, fp, fn, tn),
        })
        # rhythm: gt 2,3,6,7
        index_for_true = [2,3,6,7]
        index_for_false = [0,1,4,5]
        tp = cmtx[index_for_true][:,index_for_true].sum()
        fp = cmtx[index_for_false][:,index_for_true].sum()
        fn = cmtx[index_for_true][:,index_for_false].sum()
        tn = cmtx[index_for_false][:,index_for_false].sum()
        performance['correct_texture'] = tp+tn
        t_acc = (tp+tn) / (tp+fp+fn+tn)  # data-wise acc
        t_precision = tp / (tp+fp) if (tp+fp)>0 else None
        t_recall = tp / (tp+fn) if (tp+fn)>0 else None
        t_f1 =  None if ((t_precision is None) or (t_recall is None)) else 2*t_precision*t_recall / (t_precision + t_recall)
        performance.update({
            'rhythm_acc': t_acc,
            'rhythm_precision': t_precision, 
            'rhythm_recall': t_recall,
            'rhythm_f1': t_f1,
            'rhythm_cmtx': (tp, fp, fn, tn),
        })
        # harm: gt 4,5,6,7
        index_for_true = [4,5,6,7]
        index_for_false = [0,1,2,3]
        tp = cmtx[index_for_true][:,index_for_true].sum()
        fp = cmtx[index_for_false][:,index_for_true].sum()
        fn = cmtx[index_for_true][:,index_for_false].sum()
        tn = cmtx[index_for_false][:,index_for_false].sum()
        performance['correct_texture'] = tp+tn
        t_acc = (tp+tn) / (tp+fp+fn+tn)  # data-wise acc
        t_precision = tp / (tp+fp) if (tp+fp)>0 else None
        t_recall = tp / (tp+fn) if (tp+fn)>0 else None
        t_f1 =  None if ((t_precision is None) or (t_recall is None)) else 2*t_precision*t_recall / (t_precision + t_recall)
        performance.update({
            'harm_acc': t_acc,
            'harm_precision': t_precision, 
            'harm_recall': t_recall,
            'harm_f1': t_f1,
            'harm_cmtx': (tp, fp, fn, tn),
        })
        # label-wise accuracy
        label_wise_acc = performance['correct_texture'] / ((performance['valid_bar_num'])*3)
        performance.update({
            'label_wise_acc': label_wise_acc,
        })

    return performance




def exp_20_s3(midibert, lr, class_num, seq_class, hs, 
              model, cp_model, output_dir,
              csv_name = '2025-06-09_s3_all',
              task='melody',
              y_pred_npy='2025-06-09_s3_all_pred.npy',
            X_s3_fp=None, y_s3_fp=None,
            batch_size=12, num_workers=0,
            skip_first=None, max_len=512,is_debug=False):
    
    if X_s3_fp is None:
        output_dir = os.path.join('D:\\','backup', '2025.4.18','MIDI-BERT-CP','Data','CP_data', 'tmp')
        X_s3_fp=os.path.join(output_dir, f's3_all.npy'),
    if y_s3_fp is None:
        output_dir = os.path.join('D:\\','backup', '2025.4.18','MIDI-BERT-CP','Data','CP_data', 'tmp')
        y_s3_fp=os.path.join(output_dir, f's3_all-ans.npy')
    if not os.path.exists(X_s3_fp):
        create_and_store_dataset(cp_model, [piece_name_in_str_s3[_] for _ in range(16)], output_dir, 's3_all', ds='s3')
    if not os.path.exists(y_s3_fp):
        create_and_store_dataset(cp_model, [piece_name_in_str_s3[_] for _ in range(16)], output_dir, 's3_all', ds='s3')
    X_s3 = np.load(X_s3_fp, allow_pickle=True)
    y_s3 = np.load(y_s3_fp, allow_pickle=True)

    testset_s3 = FinetuneDataset(X=X_s3, y=y_s3, dataset='mine', task=task) 
    test_loader_s3 = DataLoader(testset_s3, batch_size=batch_size, num_workers=num_workers)
    print("   len of test_loader_s3",len(test_loader_s3))

    index_layer = int(12)-13
    print("\nCreating Finetune Trainer using index layer", index_layer)
    
    if task=='melody':
        label_dict = {
            0: {'meaning': 'Pad'},
            1: {'meaning': 'Mel'},
            2: {'meaning': 'SubMel'},
            3: {'meaning': 'Background'},
        }
        label_num = 4
    else:
        label_dict = {
            # 0: {'label': [-1, -1, -1], 'meaning': '<PAD>ForNoLabel'},
            0: {'label': [0, 0, 0], 'meaning': 'None'},
            1: {'label': [1, 0, 0], 'meaning': 'Melody'},
            2: {'label': [0, 1, 0], 'meaning': 'Rhythm'},
            3: {'label': [1, 1, 0], 'meaning': 'Melody+Rhythm'},
            4: {'label': [0, 0, 1], 'meaning': 'Harmony'},
            5: {'label': [1, 0, 1], 'meaning': 'Melody+Harmony'},
            6: {'label': [0, 1, 1], 'meaning': 'Rhythm+Harmony'},
            7: {'label': [1, 1, 1], 'meaning': 'All'}
        }
        label_num = 8
    class_count = {str(_):0 for _ in range(20)}
    for x,y in testset_s3:
        for _ in y:
            class_count[str(int(_))] += 1
    class_weight = {k: v/sum(class_count.values()) for k,v in class_count.items()}
    print(class_weight)
    trainer = FinetuneTrainer(midibert, None, None, test_loader_s3, 
                            index_layer, lr, class_num,
                            hs, y_s3.shape[:2], False, '0', model, seq_class,
                            skip_first=skip_first,label_dict=label_dict, label_num=label_num,class_weight=class_weight,
                            is_debug=is_debug, do_save_current_stage=False, return_perf=False, task=task,freeze=freeze)
    
    print("\nTraining Start")
    filename = os.path.join(output_dir, 'model.ckpt')
    print("   save model at {}".format(filename))


    all_epoch = [int(_.split('.pt')[0].split('-best')[0].split('epoch')[-1]) for _ in os.listdir(os.path.join(output_dir, 'model')) if _.startswith('epoch')]
    best_epoch = [int(_.split('-best.pt')[0].split('epoch')[-1]) for _ in os.listdir(os.path.join(output_dir, 'model')) if _.endswith('best.pt')]
    if best_epoch:
        best_epoch = max(best_epoch)
        ename = f"epoch{str(best_epoch)}-best.pt"
    elif all_epoch:
        best_epoch = max(all_epoch)
        ename = f"epoch{str(best_epoch)}.pt"
    else:
        best_epoch = None

        # best_epoch = epoch_end - 5
        # ename = f"epoch{str(best_epoch)}.pt"
    if best_epoch is not None:
        trainer.model.load_state_dict(torch.load(os.path.join(
            output_dir, 'model', ename
        )))
        # epoch_beg = max(epoch_beg, best_epoch)


    trainer.class_count = class_count
    test_loss, test_acc, all_output = trainer.test()
    print('test loss: {}, test_acc: {}'.format(test_loss, test_acc))
    np.save(os.path.join(output_dir, y_pred_npy), all_output)

    if task=='melody':
        _y = np.vectorize(testset_s3.texture2mel.get)(y_s3[:, :, 1])
        _n = 3+1
    else:
        _y = y_s3
        _n = 8
    _output = all_output.detach().cpu().numpy().astype(int)
    _output = _output.reshape(-1,1)
    _y = _y.reshape(-1,1)

    cm = confusion_matrix(_y, _output, labels=list(range(_n)))[1:, 1:]
    _title = 's3_all_mba_cm'
    np.save(os.path.join(output_dir, _title+'.npy'), cm)
    print_confusion_matrix(cm, 
        _title,
        output_dir, show=False, label_list=['Mel', 'SubMel', 'Background'])

    cm2 = cm / cm.sum(axis=1, keepdims=True)
    _title = 's3_all_mba_cm2'
    print_confusion_matrix(cm2, 
        _title,
        output_dir, show=False, label_list=['Mel', 'SubMel', 'Background'])

    # get_perf    
    y = np.load(y_s3_fp, allow_pickle=True)
    _y = np.vectorize(texture2mel.get)(y[:, :, 1]).astype(int)
    y_pred = np.load(os.path.join(output_dir, y_pred_npy)).astype(int)
    y_true = _y.reshape(-1)
    y_pred = y_pred.reshape(-1)
    print(y_true.shape, y_pred.shape)
    if task=='melody':
        label_dict = {
            0: {'meaning': 'Pad'},
            1: {'meaning': 'Mel'},
            2: {'meaning': 'SubMel'},
            3: {'meaning': 'Background'},
        }
        label_num=4
        skip_first=True
    else:
        from settings.annotations import LABEL_DCT as label_dict
        label_num = 8
        skip_first=False

    performance = get_perf(y_true, y_pred, output_dir,
        label_dict, label_num, skip_first=skip_first, task=task)
    pd.DataFrame([performance]).to_csv(
        os.path.join(output_dir, csv_name+'.csv')
        )
    

    
def create_and_store_dataset(
        cp_model, piece_lst, output_dir, dataset_name, 
        max_len=512, ds='orchestration', task='texture'
        ):
    segments, ans = cp_model.prepare_data(piece_lst, task, int(max_len), ds)
    output_file = os.path.join(output_dir, f'{dataset_name}.npy')
    np.save(output_file, segments)
    print(f'Data shape: {segments.shape}, saved at {output_file}')
    ans_file = os.path.join(output_dir, f'{dataset_name}-ans.npy')
    np.save(ans_file, ans)
    print(f'Answer shape: {ans.shape}, saved at {ans_file}')
    return

def exp_50(freeze, is_debug, is_kfold, skip_first, epoch_beg, epoch_end, PATIENT, task, exp_date, batch_size):
        # set seed
    seed = 2021
    torch.manual_seed(seed)             # cpu
    torch.cuda.manual_seed(seed)        # current gpu
    torch.cuda.manual_seed_all(seed)    # all gpu
    np.random.seed(seed)
    random.seed(seed)

    # load model
    midibert_root = os.path.join('D:\\','backup', '2025.4.18','MIDI-BERT-CP')
    dict_file = 'data_creation/prepare_data/dict/CP.pkl'
    dict_file = os.path.join(midibert_root, dict_file)
    max_seq_len = 512
    num_workers= 0  # default=5
    epochs = 10
    lr = 2e-5
    hs = 768  # hidden size

    print("Loading Dictionary")
    with open(dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

    from transformers import BertConfig
    from MidiBERT.model import MidiBert
    configuration = BertConfig(max_position_embeddings=max_seq_len,
                                    position_embedding_type='relative_key_query',
                                    hidden_size=hs)
    midibert = MidiBert(bertConfig=configuration, e2w=e2w, w2e=w2e)
    # task = 'melody'

    best_mdl = ''
    ckpt = os.path.join(
        midibert_root,
        'result/pretrain/default/model_best.ckpt'
        )
    best_mdl = ckpt
    print("   Loading pre-trained model from", best_mdl.split('/')[-1])
    checkpoint = torch.load(best_mdl, map_location='cpu')
    midibert.load_state_dict(checkpoint['state_dict'])

    print("\nLoading Dataset") 
    if task == 'melody' or task == 'velocity':
        dataset = 'pop909'
        class_num = 4
        model = TokenClassification(midibert, class_num, hs)
        seq_class = False
    elif task == 'composer' or task == 'emotion':
        dataset = task
        class_num = 3
        model = SequenceClassification(midibert, class_num, hs)
        seq_class = True
    else:
        model = None
        class_num = 3
        seq_class = False

    prepare_data_dict = os.path.join('D:\\','backup', '2025.4.18','MIDI-BERT-CP','data_creation','prepare_data','dict','CP.pkl')
    cp_model = CP(dict=prepare_data_dict)
    if task=='melody':
        label_dict = {
            0: {'meaning': 'Pad'},
            1: {'meaning': 'Mel'},
            2: {'meaning': 'SubMel'},
            3: {'meaning': 'Background'},
        }
        label_num = 4
    else:
        label_dict = {
            # 0: {'label': [-1, -1, -1], 'meaning': '<PAD>ForNoLabel'},
            0: {'label': [0, 0, 0], 'meaning': 'None'},
            1: {'label': [1, 0, 0], 'meaning': 'Melody'},
            2: {'label': [0, 1, 0], 'meaning': 'Rhythm'},
            3: {'label': [1, 1, 0], 'meaning': 'Melody+Rhythm'},
            4: {'label': [0, 0, 1], 'meaning': 'Harmony'},
            5: {'label': [1, 0, 1], 'meaning': 'Melody+Harmony'},
            6: {'label': [0, 1, 1], 'meaning': 'Rhythm+Harmony'},
            7: {'label': [1, 1, 1], 'meaning': 'All'}
        }
        label_num = 8
    # k-fold on orch, 18 pieces
    if not is_kfold:
        exp_name = task + f'{exp_date}_50-t049'
        exp_name = exp_name+'-test' if is_debug else exp_name
        output_dir = os.path.join('D:\\','backup', '2025.4.18','MIDI-BERT-CP','Data','CP_data', exp_name)
        os.makedirs(output_dir, exist_ok=True)
        save_at = output_dir
        os.makedirs(os.path.join(output_dir, 'confusion_matrix'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'fig'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'model'), exist_ok=True)

        orch_test_pieces = [piece_name_in_str[test_index] for test_index in [0,4,9]]
        # valid_index = 1
        # orch_valid_pieces = [piece_name_in_str[valid_index]]
        orch_train_pieces = [piece_name_in_str[_] for _ in range(18) if _ not in [0,4,9]]

        test_index = '049'
        dataset_name = f'mba_freeze-orch_test_{test_index}'
        if not os.path.exists(os.path.join(output_dir, f'{dataset_name}.npy')):
            create_and_store_dataset(cp_model, orch_test_pieces, output_dir, dataset_name)
        # dataset_name = f'mba_freeze-orch_valid_{test_index}'
        # if not os.path.exists(os.path.join(output_dir, f'{dataset_name}.npy')):
        #     create_and_store_dataset(cp_model, orch_valid_pieces, output_dir, dataset_name)
        dataset_name = f'mba_freeze-orch_train_{test_index}'
        if not os.path.exists(os.path.join(output_dir, f'{dataset_name}.npy')):
            create_and_store_dataset(cp_model, orch_train_pieces, output_dir, dataset_name)

        # load data
        dataset_name = f'mba_freeze-orch_train_{test_index}'
        X_train = np.load(os.path.join(output_dir, f'{dataset_name}.npy'), allow_pickle=True)
        y_train = np.load(os.path.join(output_dir, f'{dataset_name}-ans.npy'), allow_pickle=True)
        train_set = FinetuneDataset(X=X_train, y=y_train, dataset='mine', task=task) 
        # train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)

        train_size = int(0.8 * len(train_set))  # 80% 作為訓練集
        valid_size = len(train_set) - train_size  # 剩下 20% 作為驗證集
        train_subset, valid_subset = random_split(train_set, [train_size, valid_size])
        train_loader = DataLoader(train_subset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        valid_loader = DataLoader(valid_subset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        # print(np.unique(y_train))
        # dataset_name = f'mba_freeze-orch_valid_{test_index}'
        # X_valid = np.load(os.path.join(output_dir, f'{dataset_name}.npy'), allow_pickle=True)
        # y_valid = np.load(os.path.join(output_dir, f'{dataset_name}-ans.npy'), allow_pickle=True)
        # valid_set = FinetuneDataset(X=X_valid, y=y_valid, dataset='mine', task=task) 
        # valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=num_workers)
        
        dataset_name = f'mba_freeze-orch_test_{test_index}'
        X_test = np.load(os.path.join(output_dir, f'{dataset_name}.npy'), allow_pickle=True)
        y_test = np.load(os.path.join(output_dir, f'{dataset_name}-ans.npy'), allow_pickle=True)
        test_set = FinetuneDataset(X=X_test, y=y_test, dataset='mine', task=task) 
        test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        
        # train, valid, test
        # task = 'melody'
        index_layer = int(12)-13
        print("\nCreating Finetune Trainer using index layer", index_layer)

        class_count = {str(_):0 for _ in range(label_num)}
        for x,y in train_set:
            for _ in y:
                class_count[str(int(_))] += 1
        class_weight = {k: v/sum(class_count.values()) for k,v in class_count.items()}
        print(class_count, class_weight)
        trainer = FinetuneTrainer(
            midibert, train_loader, valid_loader, test_loader, 
            index_layer, lr, class_num, hs, 
            testset_shape=y_test.shape[:2], 
            cpu=False,
            cuda_devices='0', model=model, SeqClass=seq_class,
            freeze=freeze, 
            label_num=label_num,
            label_dict=label_dict, class_weight=class_weight,
            do_save_current_stage=True,
            return_perf=True, save_at=output_dir,
            early_stopping={
                'patient': PATIENT,
                'criteria': float("-inf"),
                'beat_epoch': 0,
                'rule': "max"
            },
            epoch_gap=5, epoch_beg=epoch_beg, epoch_end=epoch_end,
            stage='', piece_name=exp_name,skip_first=skip_first,
            task=task, is_debug=is_debug
        )
        trainer.class_count = class_count

        print("\nTraining Start")
        filename = os.path.join(output_dir, 'model.ckpt')
        print("   save model at {}".format(filename))


        all_epoch = [int(_.split('.pt')[0].split('-best')[0].split('epoch')[-1]) for _ in os.listdir(os.path.join(save_at, 'model')) if _.startswith('epoch')]
        best_epoch = [int(_.split('-best.pt')[0].split('epoch')[-1]) for _ in os.listdir(os.path.join(save_at, 'model')) if _.endswith('best.pt')]
        if best_epoch:
            best_epoch = max(best_epoch)
            ename = f"epoch{str(best_epoch)}-best.pt"
        elif all_epoch:
            best_epoch = max(all_epoch)
            ename = f"epoch{str(best_epoch)}.pt"
        else:
            best_epoch = None

            # best_epoch = epoch_end - 5
            # ename = f"epoch{str(best_epoch)}.pt"
        if best_epoch is not None:
            trainer.model.load_state_dict(torch.load(os.path.join(
                save_at, 'model', ename
            )))
            epoch_beg = max(epoch_beg, best_epoch)

        time_spend = []
        t0 = time.time()

        for epoch in range(epoch_beg, epoch_end):
            print('*'*10, f'Epoch {epoch:4d}', '*'*10)
            trainer.epoch = epoch
            t2 = time.time()
            current_train_loss, current_train_acc, _train_performance = trainer.train()
            t3 = time.time()
            current_valid_loss, current_valid_acc, _valid_performance = trainer.valid()
            t4 = time.time()
            current_test_loss, current_test_acc, _test_performance, _all_output = trainer.test()

            # record time spend
            t5 = time.time()
            time_spend.append(f'Epoch {epoch} spends {datetime.timedelta(seconds=t5-t2)}: training {datetime.timedelta(seconds=t3-t2)}, validing {datetime.timedelta(seconds=t4-t3)}, testing {datetime.timedelta(seconds=t5-t4)}\n')
            t1 = time.time()
            time_spend.append(f'\n\nRunning all code: {datetime.timedelta(seconds=t1-t0)}')
            with open(os.path.join(save_at, 'time_spend.txt'), 'w') as f:
                f.write('\n'.join(time_spend))
            trainer.check_early_stop(current_valid_acc)
            
    else:
        for test_index in range(18):
            exp_name = task + f'{exp_date}_50-k{test_index}'
            exp_name = exp_name+'-test' if is_debug else exp_name
            skip_first = True
            output_dir = os.path.join('D:\\','backup', '2025.4.18','MIDI-BERT-CP','Data','CP_data', exp_name)
            os.makedirs(output_dir, exist_ok=True)
            save_at = output_dir
            os.makedirs(os.path.join(output_dir, 'confusion_matrix'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'fig'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'model'), exist_ok=True)

            orch_test_pieces = [piece_name_in_str[test_index]]
            # valid_index = test_index + 1 if test_index!=17 else 0
            # orch_valid_pieces = [piece_name_in_str[valid_index]]
            orch_train_pieces = [piece_name_in_str[_] for _ in range(18) if _ not in [test_index]]

            dataset_name = f'mba_freeze-orch_test_{test_index}'
            if not os.path.exists(os.path.join(output_dir, f'{dataset_name}.npy')):
                create_and_store_dataset(cp_model, orch_test_pieces, output_dir, dataset_name)
            # dataset_name = f'mba_freeze-orch_valid_{test_index}'
            # if not os.path.exists(os.path.join(output_dir, f'{dataset_name}.npy')):
            #     create_and_store_dataset(cp_model, orch_valid_pieces, output_dir, dataset_name)
            dataset_name = f'mba_freeze-orch_train_{test_index}'
            if not os.path.exists(os.path.join(output_dir, f'{dataset_name}.npy')):
                create_and_store_dataset(cp_model, orch_train_pieces, output_dir, dataset_name)

            # load data
            dataset_name = f'mba_freeze-orch_test_{test_index}'
            X_train = np.load(os.path.join(output_dir, f'{dataset_name}.npy'), allow_pickle=True)
            y_train = np.load(os.path.join(output_dir, f'{dataset_name}-ans.npy'), allow_pickle=True)
            train_set = FinetuneDataset(X=X_train, y=y_train, dataset='mine',task=task) 
            # train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)
            train_size = int(0.8 * len(train_set))  # 80% 作為訓練集
            valid_size = len(train_set) - train_size  # 剩下 20% 作為驗證集
            train_subset, valid_subset = random_split(train_set, [train_size, valid_size])
            train_loader = DataLoader(train_subset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
            valid_loader = DataLoader(valid_subset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        
            # dataset_name = f'mba_freeze-orch_valid_{test_index}'
            # X_valid = np.load(os.path.join(output_dir, f'{dataset_name}.npy'), allow_pickle=True)
            # y_valid = np.load(os.path.join(output_dir, f'{dataset_name}-ans.npy'), allow_pickle=True)
            # valid_set = FinetuneDataset(X=X_valid, y=y_valid, dataset='mine') 
            # valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=num_workers)
            
            dataset_name = f'mba_freeze-orch_train_{test_index}'
            X_test = np.load(os.path.join(output_dir, f'{dataset_name}.npy'), allow_pickle=True)
            y_test = np.load(os.path.join(output_dir, f'{dataset_name}-ans.npy'), allow_pickle=True)
            test_set = FinetuneDataset(X=X_test, y=y_test, dataset='mine',task=task)  
            test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
            print(np.unique(y_train))
            # train, valid, test
            # task = 'melody'
            index_layer = int(12)-13
            print("\nCreating Finetune Trainer using index layer", index_layer)
            
            class_count = {str(_):0 for _ in range(len(label_dict))}
            for x,y in train_set:
                for _ in y:
                    class_count[str(int(_))] += 1
            class_weight = {k: v/sum(class_count.values()) for k,v in class_count.items()}
            trainer = FinetuneTrainer(
                midibert, train_loader, valid_loader, test_loader, 
                index_layer, lr, class_num, hs, 
                testset_shape=y_test.shape[:2], 
                cpu=False,
                cuda_devices='0', model=model, SeqClass=seq_class,
                freeze=freeze, task='',
                label_num=4,
                label_dict=label_dict, class_weight=class_weight,
                do_save_current_stage=True,
                return_perf=True, save_at=output_dir,
                early_stopping={
                    'patient': PATIENT,
                    'criteria': float("-inf"),
                    'beat_epoch': 0,
                    'rule': "max"
                },
                epoch_gap=5, epoch_beg=epoch_beg, epoch_end=epoch_end,
                stage='', piece_name=exp_name,skip_first=skip_first
            )
            trainer.class_count = class_count

            print("\nTraining Start")
            filename = os.path.join(output_dir, 'model.ckpt')
            print("   save model at {}".format(filename))

            best_acc, best_epoch = 0, 0
            bad_cnt = 0

            time_spend = []
            t0 = time.time()

            performance = pd.DataFrame(
                columns=['epoch', 'train_acc_l', 'test_acc_l', 'train_acc_d', 'test_acc_d', 
                'train_precision_mel', 'train_recall_mel', 'test_precision_mel', 'test_recall_mel',
                'train_precision_rhythm', 'train_recall_rhythm', 'test_precision_rhythm', 'test_recall_rhythm',
                'train_precision_harm', 'train_recall_harm', 'test_precision_harm', 'test_recall_harm']
            )
            for epoch in range(epoch_beg, epoch_end):
                print('*'*10, f'Epoch {epoch:4d}', '*'*10)
                trainer.epoch = epoch
                t2 = time.time()
                current_train_loss, current_train_acc, _train_performance = trainer.train()
                t3 = time.time()
                current_valid_loss, current_valid_acc, _valid_performance = trainer.valid()
                t4 = time.time()
                current_test_loss, current_test_acc, _test_performance, _all_output = trainer.test()

                # record time spend
                t5 = time.time()
                time_spend.append(f'Epoch {epoch} spends {datetime.timedelta(seconds=t5-t2)}: training {datetime.timedelta(seconds=t3-t2)}, validing {datetime.timedelta(seconds=t4-t3)}, testing {datetime.timedelta(seconds=t5-t4)}\n')
                t1 = time.time()
                time_spend.append(f'\n\nRunning all code: {datetime.timedelta(seconds=t1-t0)}')
                with open(os.path.join(save_at, 'time_spend.txt'), 'w') as f:
                    f.write('\n'.join(time_spend))
                trainer.check_early_stop(current_valid_acc)

            break
    
    return


def exp_50_test_s3(freeze, is_debug, is_kfold, skip_first, epoch_beg, epoch_end, PATIENT, task, exp_date, batch_size):
        # set seed
    seed = 2021
    torch.manual_seed(seed)             # cpu
    torch.cuda.manual_seed(seed)        # current gpu
    torch.cuda.manual_seed_all(seed)    # all gpu
    np.random.seed(seed)
    random.seed(seed)

    # load model
    midibert_root = os.path.join('D:\\','backup', '2025.4.18','MIDI-BERT-CP')
    dict_file = 'data_creation/prepare_data/dict/CP.pkl'
    dict_file = os.path.join(midibert_root, dict_file)
    max_seq_len = 512
    num_workers= 0  # default=5
    # class_num = 8
    epochs = 10
    lr = 2e-5
    hs = 768  # hidden size

    print("Loading Dictionary")
    with open(dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

    from transformers import BertConfig
    from MidiBERT.model import MidiBert
    configuration = BertConfig(max_position_embeddings=max_seq_len,
                                    position_embedding_type='relative_key_query',
                                    hidden_size=hs)
    midibert = MidiBert(bertConfig=configuration, e2w=e2w, w2e=w2e)
    # task = 'melody'

    best_mdl = ''
    ckpt = os.path.join(
        midibert_root,
        'result/pretrain/default/model_best.ckpt'
        )
    
    best_mdl = ckpt
    print("   Loading pre-trained model from", best_mdl.split('/')[-1])
    checkpoint = torch.load(best_mdl, map_location='cpu')
    midibert.load_state_dict(checkpoint['state_dict'])
    print("\nLoading Dataset") 
    if task == 'melody' or task == 'velocity':
        dataset = 'pop909'
        model = TokenClassification(midibert, class_num, hs)
        class_num = 4
        seq_class = False
    elif task == 'composer' or task == 'emotion':
        dataset = task
        model = SequenceClassification(midibert, class_num, hs)
        class_num = 3
        seq_class = True
    else:
        model = None
        class_num = 3
        seq_class = False

    # Start new exp
    # freeze = True
    # is_kfold = False
    # is_debug = False
    # epoch_beg, epoch_end = 0, 2 if is_debug else 50

    prepare_data_dict = os.path.join('D:\\','backup', '2025.4.18','MIDI-BERT-CP','data_creation','prepare_data','dict','CP.pkl')
    cp_model = CP(dict=prepare_data_dict)
    
    # k-fold on orch, 18 pieces
    if not is_kfold:
        exp_name = task + f'{exp_date}_50-t049'
        exp_name = exp_name+'-test' if is_debug else exp_name
        skip_first = True
        output_dir = os.path.join('D:\\','backup', '2025.4.18','MIDI-BERT-CP','Data','CP_data', exp_name)
        os.makedirs(output_dir, exist_ok=True)
        save_at = output_dir

        # train, valid, test
        # task = 'melody'
        index_layer = int(12)-13
        print("\nCreating Finetune Trainer using index layer", index_layer)
        print("\nTraining Start")
        filename = os.path.join(output_dir, 'model.ckpt')
        print("   save model at {}".format(filename))
        # After training, load the best model and test on each mvt 
        # best_epoch = [int(_.split('-best.pt')[0].split('epoch')[-1]) for _ in os.listdir(os.path.join(save_at, 'model')) if _.endswith('best.pt')]
        # if best_epoch:
        #     best_epoch = max(best_epoch)
        # else:
        #     best_epoch = None
        #     # best_epoch = 0 
        # if best_epoch is not None:
        #      model.load_state_dict(torch.load(os.path.join(
        #         # save_at, 'model', f"epoch{str(best_epoch)}.pt"
        #         save_at, 'model', f"epoch{str(best_epoch)}-best.pt"
        #     )))

        # test on s3
        
        exp_20_s3(midibert, lr, class_num, seq_class, hs, 
                model, cp_model, output_dir,
            csv_name = exp_name+'_s3_all',
            task=task,
            y_pred_npy=exp_name+'_y_pred.npy',
            X_s3_fp=os.path.join(output_dir, 's3_all.npy'),
            y_s3_fp=os.path.join(output_dir, 's3_all-ans.npy'),
            batch_size=batch_size, num_workers=num_workers, is_debug=is_debug
        )

    else:    
        # k-fold on orch, 18 pieces
        for test_index in range(18):
            exp_name = task + f'{exp_date}_50-k{test_index}'
            exp_name = exp_name+'-test' if is_debug else exp_name
            # PATIENT = 10
            # skip_first = True

            prepare_data_dict = os.path.join('D:\\','backup', '2025.4.18','MIDI-BERT-CP','data_creation','prepare_data','dict','CP.pkl')
            cp_model = CP(dict=prepare_data_dict)
            output_dir = os.path.join('D:\\','backup', '2025.4.18','MIDI-BERT-CP','Data','CP_data', exp_name)
            save_at = output_dir
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'confusion_matrix'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'fig'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'model'), exist_ok=True)

            # train, valid, test
            # task = 'melody'
            index_layer = int(12)-13
            print("\nCreating Finetune Trainer using index layer", index_layer)

            print("\nTraining Start")
            filename = os.path.join(output_dir, 'model.ckpt')
            print("   save model at {}".format(filename))

            # test on s3
            exp_20_s3(midibert, lr, class_num, seq_class, hs, 
                    model, cp_model, output_dir,
                csv_name = exp_name+'_s3_all',
                task=task,
                y_pred_npy=exp_name+'_y_pred.npy',
                X_s3_fp=os.path.join(output_dir, 's3_all.npy'),
                y_s3_fp=os.path.join(output_dir, 's3_all-ans.npy'),
                batch_size=batch_size, num_workers=num_workers, skip_first=skip_first
            )


if __name__=='__main__':
    # Start new exp
    freeze = True
    is_debug = False
    is_kfold = False
    skip_first = True
    epoch_beg, epoch_end = 0, 2 if is_debug else 150
    PATIENT = 10
    task = 'texture'
    exp_date = 'texture2025-06-10'
    batch_size = 6

    exp_50(freeze, is_debug, is_kfold, skip_first, epoch_beg, epoch_end, PATIENT, task, exp_date, batch_size)
    exp_50_test_s3(freeze, is_debug, is_kfold, skip_first, epoch_beg, epoch_end, PATIENT, task, exp_date, batch_size)