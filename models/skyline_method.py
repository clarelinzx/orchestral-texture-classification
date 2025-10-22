"""
File: skyline_method.py
Author: Zih-Syuan Lin (2025)
"""
# %%[markdown]
# ## Imports
import os
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from models.utils import load_a_piece, print_confusion_matrix
from settings.annotations import LABEL_DCT, LABEL_LST

# %%[markdown]
# ## Imports
def load_note_a_piece(save_at):
    
    with open(os.path.join(
        save_at, 'role',f'track_order.txt'), 'r') as f:
        track_names = f.readline().strip().split(';')
    a_piece = {
        'track_names': track_names,
    }
    print(f'{save_at=} | {track_names=}')
    for track_name in track_names:
        a_piece[track_name] = pd.read_csv(os.path.join(save_at, 'note_xy', f"{track_name}.csv"))
    return a_piece

def get_performance(cm, valid_bar_num, class_weight, stage, piece_name, use_model):
    performance = {
        'stage': stage,
        'piece_name': piece_name,
        'use_model': use_model,
        'texture_obj': np.zeros((8, 4)),
        'valid_bar_num': valid_bar_num,
        'cm': cm,
        'texture_obj_aprf': np.zeros((8,4)),
    }
    # get tp,fp,fn,tn for each class
    for _index in range(8):
        tp = cm[_index, _index]
        fp = cm[:, _index].sum() - tp
        fn = cm[_index, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        performance['texture_obj'][_index] += np.array([tp, fp, fn, tn])

    # claculate APRF for each class
    for _index in range(8):
            tp, fp, fn, tn = performance['texture_obj'][_index]
            t_acc = (tp+tn) / (tp+fp+fn+tn)  # data-wise acc
            t_precision = tp / (tp+fp) if (tp+fp)>0 else None
            t_recall = tp / (tp+fn) if (tp+fn)>0 else None
            t_f1 =  None if ((t_precision is None) or (t_recall is None)) else 2*t_precision*t_recall / (t_precision + t_recall)
            performance['texture_obj_aprf'][_index] += [t_acc, t_precision if t_precision is not None else 0, t_recall if t_recall is not None else 0, t_f1 if t_f1 is not None else 0]
            # support: The number of occurrences of each label in y_true
            print(f"{LABEL_DCT[_index]['meaning']:<20} | {t_acc:>6.2f} | {t_precision if t_precision is not None else -1:>6.2f} | {t_recall if t_recall is not None else -1:>6.2f} | {t_f1 if t_f1 is not None else -1:>6.2f}")
            label_name = LABEL_DCT[_index]['meaning']
            performance.update({
                f'acc_{label_name}': t_acc,
                f'precision_{label_name}': t_precision, 
                f'recall_{label_name}': t_recall,
                f'f1_{label_name}': t_f1,
                f'cmtx_{label_name}': (tp, fp, fn, tn),
            })

    # claculate A for all 
    tp, fp, fn, tn = performance['texture_obj'][:,:].sum(axis=0)  
    accuracy = tp / performance['valid_bar_num']
    performance['accuracy'] = accuracy
    print(f"{'average':<20} | {accuracy:>6.2f} | {performance['valid_bar_num']}")
          
    # micro average: each bar has same weight
    tp, fp, fn, tn = performance['texture_obj'][:,:].sum(axis=0) 
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
    print(f"{'micro-average':<20} | {t_acc:>6.2f} | {t_precision if t_precision is not None else -1:>6.2f} | {t_recall if t_recall is not None else -1:>6.2f} | {t_f1 if t_f1 is not None else -1:>6.2f}")
    
    # macro-average: each class has same weight
    class_weight_int = [class_weight[str(k)] for k in range(8)]
    macro = np.zeros((3,4))  # [[APRF], [APRF], [APRF]]
    for _ in range(8):
        label_name = LABEL_DCT[_]['meaning']
        if _>0:
            macro[0][0] += performance[f'acc_{label_name}'] /7 if performance[f'acc_{label_name}'] is not None else 0
            macro[0][1] += performance[f'precision_{label_name}']/7 if performance[f'precision_{label_name}'] is not None else 0
            macro[0][2] += performance[f'recall_{label_name}']/7 if performance[f'recall_{label_name}'] is not None else 0
            macro[0][3] += performance[f'f1_{label_name}']/7 if performance[f'f1_{label_name}'] is not None else 0
        macro[1][0] += performance[f'acc_{label_name}'] /8 if performance[f'acc_{label_name}'] is not None else 0
        macro[1][1] += performance[f'precision_{label_name}'] /8 if performance[f'precision_{label_name}'] is not None else 0
        macro[1][2] += performance[f'recall_{label_name}'] /8 if performance[f'recall_{label_name}'] is not None else 0
        macro[1][3] += performance[f'f1_{label_name}'] /8 if performance[f'f1_{label_name}'] is not None else 0
        macro[2][0] += performance[f'acc_{label_name}'] * class_weight_int[_] if performance[f'acc_{label_name}'] is not None else 0
        macro[2][1] += performance[f'precision_{label_name}']* class_weight_int[_] if performance[f'precision_{label_name}'] is not None else 0
        macro[2][2] += performance[f'recall_{label_name}']* class_weight_int[_] if performance[f'recall_{label_name}'] is not None else 0
        macro[2][3] += performance[f'f1_{label_name}']* class_weight_int[_] if performance[f'f1_{label_name}'] is not None else 0
                
    performance.update({
            f'texture_acc_macro_7': macro[0,0],
            f'texture_precision_macro_7': macro[0,1],
            f'texture_recall_macro_7': macro[0,2],
            f'texture_f1_macro_7':macro[0,3],
            # divide to 8
            f'texture_acc_macro_8': macro[1,0],
            f'texture_precision_macro_8': macro[1,1],
            f'texture_recall_macro_8': macro[1,2],
            f'texture_f1_macro_8': macro[1,3],
            # calculate by weight
            f'texture_acc_macro_weight': macro[2,0],
            f'texture_precision_macro_weight': macro[2,1],
            f'texture_recall_macro_weight': macro[2,2],
            f'texture_f1_macro_weight': macro[2,3],
    })
    print(f"{'macro-average 7':<20} | {performance['texture_acc_macro_7']:>6.2f} | {performance['texture_precision_macro_7']:>6.2f} | {performance['texture_recall_macro_7']:>6.2f} | {performance['texture_f1_macro_7'] if performance['texture_f1_macro_7'] is not None else -1:>6.2f}")
    print(f"{'macro-average 8':<20} | {performance['texture_acc_macro_8']:>6.2f} | {performance['texture_precision_macro_8']:>6.2f} | {performance['texture_recall_macro_8']:>6.2f} | {performance['texture_f1_macro_8'] if performance['texture_f1_macro_8'] is not None else -1:>6.2f}")
    print(f"{'macro-average weight':<20} | {performance['texture_acc_macro_weight']:>6.2f} | {performance['texture_precision_macro_weight']:>6.2f} | {performance['texture_recall_macro_weight']:>6.2f} | {performance['texture_f1_macro_weight'] if performance['texture_f1_macro_weight'] is not None else -1:>6.2f}")
    # 3 classes

    cmtx = performance['cm']
    # mel: gt 1,3,5,7
    index_for_true = [1,3,5,7]
    index_for_false = [0,2,4,6]
    tp = cmtx[index_for_true][:,index_for_true].sum()
    fp = cmtx[index_for_false][:,index_for_true].sum()
    fn = cmtx[index_for_true][:,index_for_false].sum()
    tn = cmtx[index_for_false][:,index_for_false].sum()
    performance['correct_texture_mel'] = tp+tn
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
    performance['correct_texture_rhythm'] = tp+tn
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
    performance['correct_texture_harm'] = tp+tn
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
    performance.update({
        'label_wise_acc(mel)': performance['correct_texture_mel'] / (performance['valid_bar_num']*3),
        'label_wise_acc(rhythm)': performance['correct_texture_rhythm'] / (performance['valid_bar_num']*3),
        'label_wise_acc(harm)': performance['correct_texture_harm'] / (performance['valid_bar_num']*3),
    })

    return performance



# %%[markdown]
# ## 
if __name__=='__main__':
    dataset, notes_df = {}, {}
    from ..settings.orchestration_info import piece_names, file_path
    dataset['orchestration'] = {
            piece_name: load_a_piece(
            os.path.join('converted_dataset', f'rf','orchestration'),
            piece_name,
            'orchestration',
            file_path
            ) for piece_name in piece_names
    }
    al_piece_names = piece_names
    notes_df['orchestration']={
            piece_name: load_note_a_piece(
            os.path.join('.', 'converted_dataset', f'orchestration', piece_name)
            ) for piece_name in file_path
    }

    from ..settings.s3_info import piece_names, file_path
    dataset['s3'] = {
            piece_name: load_a_piece(
            os.path.join('converted_dataset', f'rf', 's3'),
            piece_name,
            's3',
            file_path
            ) for piece_name in piece_names
    }
    notes_df['s3'] = {
            piece_name: load_note_a_piece(
            os.path.join('converted_dataset', f's3', piece_name),
            ) for piece_name in file_path
        }
    s3_piece_names = piece_names
    result_orch =  []
    ds = 's3'
    piece_names = al_piece_names if ds=='orchestration' else s3_piece_names
    for pdx, piece_name in enumerate(piece_names):
        all_note_df_for_this_piece = pd.concat([notes_df[ds][piece_name][track_name] for track_name in notes_df[ds][piece_name]['track_names']])
        mn_lst = list(all_note_df_for_this_piece.groupby('measure').groups.keys())
        for mn in mn_lst:
            sorted_mn_notes = all_note_df_for_this_piece.groupby('measure').get_group(mn).sort_values('midi_number', ascending=False)
            diff_pitch = np.unique(sorted_mn_notes.midi_number)
            max_pitch = sorted_mn_notes.midi_number.max()
            sorted_mn_notes['skyline'] = sorted_mn_notes['midi_number']==max_pitch

            for track_name in sorted_mn_notes.groupby('inst_name').groups.keys():
                track_bar = sorted_mn_notes.groupby('inst_name').get_group(track_name)
                note_num, _ = track_bar.shape
                y_true = dataset[ds][piece_name][track_name].loc[mn, 'main_label']
                if note_num == 1:
                    y_pred = track_bar['skyline'].values[0]
                elif note_num > 1:
                    is_skyline_value, is_skyline_count = np.unique(track_bar['skyline'].values, return_counts=True)
                    if is_skyline_count.shape[0]==1:
                        y_pred = is_skyline_value[0]
                    else:
                        true_count = is_skyline_count[list(is_skyline_value).index(True)]
                        false_count = is_skyline_count[list(is_skyline_value).index(False)]
                        if true_count - false_count >=0:
                            y_pred = True
                        else:
                            y_pred = False
                result_orch.append({
                    'ds': ds,
                    'piece_name': piece_name,
                    'track_name': track_name,
                    'mn': mn,
                    'y_true': y_true,
                    'y_pred': 1 if y_pred else 0
                })

    orch = pd.read_csv(os.path.join('result', 'skyline', '2025-06-18_skyline_orch.csv'), index_col=0)
    s3 = pd.read_csv(os.path.join('result', 'skyline', '2025-06-18_skyline_s3.csv'), index_col=0)

    orch = orch[orch['y_true']!=0]
    orch['y_pred'] = orch['y_pred'].apply(lambda x: 1)
    k,v = np.unique(orch['y_true'].values, return_counts=True)
    class_count = {k:0 for k in range(8)}
    class_count.update(dict(zip(k,v)))
    total_count = sum(class_count.values())
    class_weight = {str(k):class_count[k]/total_count for k in range(8)}
    valid_bar_num = orch.shape[0]
    cm = confusion_matrix(orch['y_true'].astype(int), orch['y_pred'], labels=list(range(8)))
    s3 = s3[s3['y_true']!=0]
    s3['y_pred'] = s3['y_pred'].apply(lambda x: 1)
    k,v = np.unique(s3['y_true'].values, return_counts=True)
    class_count = {k:0 for k in range(8)}
    class_count.update(dict(zip(k,v)))
    total_count = sum(class_count.values())
    class_weight = {str(k):class_count[k]/total_count for k in range(8)}
    valid_bar_num = s3.shape[0]
    cm_s3 = confusion_matrix(s3['y_true'].astype(int), s3['y_pred'], labels=list(range(8)))
    perf_s3 = get_performance(cm, valid_bar_num, class_weight, '', '', '')

    print_confusion_matrix(cm, 'skyline orchestration recall 1 ', 'result/skyline', True)
    print_confusion_matrix(cm_s3, 'skyline s3 recall mel 1 ', 'result/skyline', True)
# %%
