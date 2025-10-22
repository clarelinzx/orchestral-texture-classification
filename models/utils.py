"""
File: utils.py
Author: Zih-Syuan (2025)
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

from settings.instruments import INSTRUMENT_LOOKUP_TABLE, INSTRUMENT_NAME, INST_CLASSES
from settings.annotations import LABEL_LST, LABEL_DCT

def three_bool_to_eight_class(three_bool):
    '''
    input: shpae=(n, 3)
    convert to three bool value [is_mel, is_rhythm, is_harm] to 2^3=8 classes, classed number is from 0 to 7
    example.
        [0, 0, 0] => class 0: no roles
        [1, 0, 0] => class 1: melody only
        [0, 1, 1] => class 6: rhythm and harmony
    '''
    if three_bool.shape[-1]!=3: 
        raise Exception('Invalid bool array to convert to classes')
    cls = three_bool[:,0]*1 + three_bool[:,1]*2 + three_bool[:,2]*4
    return cls


def update_performance_dataframe(original_df, train, test, epoch):
    train_p_r = train['precision_recall_of_role'] #precision and recall metrics of [mel, rhythm, harm], refer to model.py for details
    test_p_r = test['precision_recall_of_role']
    new_data = pd.DataFrame([[epoch, train['label_wise_acc'], test['label_wise_acc'],\
                                train['data_wise_acc'], test['data_wise_acc'],\
                                train_p_r[0], train_p_r[1], test_p_r[0], test_p_r[1],\
                                train_p_r[2], train_p_r[3], test_p_r[2], test_p_r[3],\
                                train_p_r[4], train_p_r[5], test_p_r[4], test_p_r[5]]],\
                                columns=['epoch', 'train_acc_l', 'test_acc_l', 'train_acc_d', 'test_acc_d', \
                            'train_precision_mel', 'train_recall_mel', 'test_precision_mel', 'test_recall_mel',\
                            'train_precision_rhythm', 'train_recall_rhythm', 'test_precision_rhythm', 'test_recall_rhythm',\
                            'train_precision_harm', 'train_recall_harm', 'test_precision_harm', 'test_recall_harm'] )
    return pd.concat([original_df, new_data], ignore_index=True)

def print_confusion_matrix(confusion_matrix, title, fig_at, show=False, label_list=LABEL_LST):
    fig = px.imshow(
        confusion_matrix,
        text_auto='.2f',
        aspect='auto',
        x=label_list,
        y=label_list,
        color_continuous_scale='Blues',
        title=title,
        height=500,
        width=700,
    )
    fig.update_xaxes(side='top')
    fig.write_image(
        os.path.join(fig_at, f"{title}.png")
    )
    if show:
        fig.show()

def get_class_weight(data_set):
    class_count = {str(_):0 for _ in range(8)}
    for x,y in data_set:
        y = y[0] + y[1]*2 + y[2]*4
        class_count[str(int(y))] += 1
    class_weight = {k: sum(class_count.values())/(v+1) for k,v in class_count.items()}
    return class_count, class_weight


def get_class_weight_for_perf(data_set):
    class_count = {str(_):0 for _ in range(8)}
    for x,y in data_set:
        y = y[0] + y[1]*2 + y[2]*4
        class_count[str(int(y))] += 1
    class_weight = {k: v/sum(class_count.values()) for k,v in class_count.items()}
    return class_count, class_weight

def find_inst_class(inst_name):
    for inst_class in INST_CLASSES:
        if inst_name in INST_CLASSES[inst_class]:
            return inst_class
    return None



def get_num_params(ll, params_table, total_params, total_params_nograd, ll_name=''):
    for name, param in ll.named_parameters():
        if param.requires_grad:
            params_table.append({
                'layer_name': ll_name,
                'name': name,
                'num_params': param.numel(),
                'require_grad': True,
            })
            total_params += param.numel()
        else:
            total_params_nograd += param.numel()
            params_table.append({
                'layer_name': ll_name,
                'name': name,
                'num_params': param.numel(),
                'require_grad': False,
            })
    return params_table, total_params, total_params_nograd


def draw_3_fig(save_at, figsize=(18,6), csv_name='performance.csv'):
    performance_path = os.path.join(save_at, csv_name)
    save_at = os.path.join(save_at, 'curve')
    os.makedirs(save_at, exist_ok=True)
    df = pd.read_csv(performance_path)
    print(df.columns.tolist())
    df["loss"].head()
    train_df = df[df["stage"] == "train"]
    test_df = df[df["stage"] == "test"]
    fig = plt.figure(figsize=figsize)

    # Loss Curve
    fig.add_subplot(131)
    plt.title("Loss Curve")
    plt.plot(train_df["epoch"], train_df["loss"], label="Train Loss")
    plt.plot(test_df["epoch"], test_df["loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(True)

    # Micro
    fig.add_subplot(132)
    plt.title("Curve: Micro Average")
    plt.plot(train_df["epoch"], train_df["accuracy"], label="Train Acc", color='#0096FF')  # color
    plt.plot(test_df["epoch"], test_df["accuracy"], label="Test Acc", color='#FF5733')
    plt.plot(train_df["epoch"], train_df["texture_precision_micro"], label="Train Precision", linestyle="dashdot", color='#9fc5e8')
    plt.plot(test_df["epoch"], test_df["texture_precision_micro"], label="Test Precision", linestyle="dashdot", color='#f9cb9c')
    plt.plot(train_df["epoch"], train_df["texture_recall_micro"], label="Train Recall",linestyle="dotted", color='#6fa8dc')  # color
    plt.plot(test_df["epoch"], test_df["texture_recall_micro"], label="Test Recall",linestyle="dotted", color='#f6b26b')
    plt.plot(train_df["epoch"], train_df["texture_f1_micro"], label="Train F1", linestyle="dashed", color='#0000FF')
    plt.plot(test_df["epoch"], test_df["texture_f1_micro"], label="Test F1", linestyle="dashed", color='#C70039')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(-.1, 1.1)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(True)

    # Macro
    fig.add_subplot(133)
    plt.title("Curve: Macro Average")
    plt.plot(train_df["epoch"], train_df["accuracy"], label="Train Acc", color='#0096FF')  # color
    plt.plot(test_df["epoch"], test_df["accuracy"], label="Test Acc", color='#FF5733')
    plt.plot(train_df["epoch"], train_df["texture_precision_macro_weight"], label="Train Precision", linestyle="dashdot", color='#9fc5e8')
    plt.plot(test_df["epoch"], test_df["texture_precision_macro_weight"], label="Test Precision", linestyle="dashdot", color='#f9cb9c')
    plt.plot(train_df["epoch"], train_df["texture_recall_macro_weight"], label="Train Recall",linestyle="dotted", color='#6fa8dc')  # color
    plt.plot(test_df["epoch"], test_df["texture_recall_macro_weight"], label="Test Recall",linestyle="dotted", color='#f6b26b')
    plt.plot(train_df["epoch"], train_df["texture_f1_macro_weight"], label="Train F1", linestyle="dashed", color='#0000FF')
    plt.plot(test_df["epoch"], test_df["texture_f1_macro_weight"], label="Test F1", linestyle="dashed", color='#C70039')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(-.1, 1.1)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_at, 'all_fig.png'))
    plt.close()


def draw_10_fig(save_at, figsize=(18,18), figsize2=(6,4), csv_name='performance.csv'):
    performance_path = os.path.join(save_at, csv_name)
    save_at = os.path.join(save_at, 'curve')
    os.makedirs(save_at, exist_ok=True)
    df = pd.read_csv(performance_path)

    df["loss"].head()

    train_df = df[df["stage"] == "train"]
    test_df = df[df["stage"] == "test"]

    fig = plt.figure(figsize=figsize)

    for _ in range(8):
        fig.add_subplot(4,2,_+1)
        cat_name = LABEL_DCT[_]['meaning']
        plt.plot(train_df["epoch"], train_df[f"precision_{cat_name}"], label="Train Precision", linestyle="dashed", color='#89CFF0')
        plt.plot(train_df["epoch"], train_df[f"recall_{cat_name}"], label="Train Recall", linestyle="dotted", color='#0096FF')
        plt.plot(train_df["epoch"], train_df[f"f1_{cat_name}"], label="Train F1", linestyle="dashdot", color='#0000FF')
        plt.plot(test_df["epoch"], test_df[f"precision_{cat_name}"], label="Test Precision", linestyle="dashed", color='#FFC300')
        plt.plot(test_df["epoch"], test_df[f"recall_{cat_name}"], label="Test Recall", linestyle="dotted", color='#FF5733')
        plt.plot(test_df["epoch"], test_df[f"f1_{cat_name}"], label="Test F1", linestyle="dashdot", color='#C70039')
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"{cat_name} Metrics Over Epochs")
        plt.ylim(-.1, 1.1)
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_at, 'all_fig_by_class.png'))
    plt.close()

    plt.figure(figsize=figsize2)
    plt.plot(train_df["epoch"], train_df[f"texture_precision_micro"], label="Train acc", linestyle="dashed", color='#89CFF0')
    plt.plot(train_df["epoch"], train_df[f"texture_recall_micro"], label="Train Recall", linestyle="dotted", color='#0096FF')
    plt.plot(train_df["epoch"], train_df[f"texture_f1_micro"], label="Train F1", linestyle="dashdot", color='#0000FF')
    plt.plot(test_df["epoch"], test_df[f"texture_precision_micro"], label="Test Precision", linestyle="dashed", color='#FFC300')
    plt.plot(test_df["epoch"], test_df[f"texture_recall_micro"], label="Test Recall", linestyle="dotted", color='#FF5733')
    plt.plot(test_df["epoch"], test_df[f"texture_f1_micro"], label="Test F1", linestyle="dashdot", color='#C70039')
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"Overall Metrics Over Epochs")
    plt.ylim(-.1, 1.1)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_at, 'all_fig_by_class_overall.png'))
    plt.close()


def draw_test_on(save_at, piece_names, ds='s3', piece_name_in_str=None):
    performance_path = os.path.join(save_at, 'test_on', f'{ds}_performance.csv')
    save_at = os.path.join(save_at, 'curve')
    os.makedirs(save_at, exist_ok=True)
    df = pd.read_csv(performance_path)
    acc_values = df["accuracy"]
    mean_label_acc = np.mean(acc_values)
    std_label_acc = np.std(acc_values)
    plt.figure(figsize=(12, 6))
    plt.plot(piece_names, acc_values, marker='o', label="Acc")
    plt.axhline(mean_label_acc, color='blue', linestyle='--', alpha=0.5, label=f"Mean Label wise acc ({mean_label_acc:.2f})")
    plt.axhline(mean_label_acc + std_label_acc, color='blue', linestyle=':', alpha=0.3, label=f"Label wise acc +1 STD ({mean_label_acc + std_label_acc:.2f})")
    plt.axhline(mean_label_acc - std_label_acc, color='blue', linestyle=':', alpha=0.3, label=f"Label wise acc -1 STD ({mean_label_acc - std_label_acc:.2f})")
    plt.title(f"[{ds}] Accuracy Metrics across Different Pieces")
    plt.xlabel("Piece Name")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.ylim(0, 1)
    if piece_name_in_str is not None:
        new_x_tick = [piece_name_in_str[_] for _ in piece_names]
        plt.xticks(piece_names, labels=new_x_tick)
    plt.tight_layout()
    plt.savefig(os.path.join(save_at, f'test_on_{ds}.png'))
    plt.close()



def load_a_piece(save_at, piece_name, ds, ALL_PATH):
    a_piece = {}
    all_track_df = pd.read_csv(os.path.join(
        'converted_dataset', f'rf-2025-06-17', ds, f"{piece_name}_all_track.csv"
        )).set_index('measure_number')
    with open(os.path.join(
        ALL_PATH[ds]['converted_path'], piece_name, 'role',f'track_order.txt'), 'r') as f:
        track_names = f.readline().strip().split(';')
    print(f'{piece_name=} | {track_names=}')
    a_piece['track_names'] = track_names
    for tdx, track_name in enumerate(track_names):
        a_piece[track_name] = load_a_track(
            save_at, piece_name, track_name, all_track_df)
    return a_piece

def load_a_track(save_at, piece_name, track_name, all_track_df):
    track_df = pd.read_csv(os.path.join(
            save_at, f'{piece_name}_{track_name}.csv'
            )).set_index('measure_number')
    if all_track_df is not None:
        track_df = track_df.join(all_track_df, rsuffix='_all')
    track_df = track_df.reset_index()
    return track_df

