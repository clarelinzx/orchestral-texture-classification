"""
File: random_forest.py
Author: Zih-Syuan Lin (2025)
"""
# %%[markdown]
# ## Imports
import joblib
import os
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from settings.annotations import LABEL_DCT, LABEL_LST
from settings.evaluation import PERFORMANCE_DF_COLS
from settings.statistical_features import STAT_FEATURE_COLS
from models.utils import load_a_piece, load_a_track
# %%[markdown]
# ## Functions

def get_combine_df(piece_names, dataset, skip_0=False, al_piece_names=None, s3_piece_names=None, skip_labels=[], convert_to_2_class=False):
    df = []
    for piece_name in piece_names:
        if piece_name in al_piece_names:
            for track_name in dataset['orchestration'][piece_name]['track_names']:
                df.append(
                    dataset['orchestration'][piece_name][track_name]
                )
        elif piece_name in s3_piece_names:
            for track_name in dataset['s3'][piece_name]['track_names']:
                df.append(
                    dataset['s3'][piece_name][track_name]
                )
        else:
            raise ValueError(f"Piece {piece_name} not found in dataset!")
    
    df = pd.concat(df, ignore_index=True)
    # df = df[df['onset'] < df['offset']]
    if skip_0:
        df = df[df['main_label']!=0]
    for _l in skip_labels:
        df = df[df['main_label'] != _l]
    if convert_to_2_class:
        convert_dct = {
            # 0:0, 1:1, 2:2, 3:2, 4:2, 5:2, 6:2, 7:2, # mel /acc
            # 0:0, 1:1, 2:3, 3:2, 4:3, 5:2, 6:3, 7:2  # mel / submel / acc
            # 0:0, 1:2, 2:2, 3:2, 4:4, 5:2, 6:2, 7:7    # rhythm (include mel) / harm / all
            # 0:0, 1:1, 2:2, 3:0, 4:4, 5:0, 6:0, 7:7    # mel / rhythm / harm / all
            0:0, 1:1, 2:2, 3:1, 4:2, 5:1, 6:2, 7:7, # MB
            # 0:0, 1:0, 2:2, 3:0, 4:4, 5:0, 6:6, 7:7  # RH2
        }
        df['main_label'] = df['main_label'].apply(lambda x:convert_dct[x])
    if piece_name in al_piece_names:
        filter_df = pd.read_csv('converted_dataset/cnn-orch-2025-06-20.csv')
        df = pd.merge(df, filter_df, on=['piece_name', 'measure_number', 'track_name'], how='inner', suffixes=('_rf','_cnn'))
    else:
        filter_df = pd.read_csv('converted_dataset/cnn-s3-2025-06-20.csv')
        df = pd.merge(df, filter_df, on=['piece_name', 'measure_number', 'track_name'], how='inner', suffixes=('_rf','_cnn'))

    return df

def get_x_y_df(df, x_cols, y_cols):
    x_df = df[x_cols].fillna(-1)
    y_df = df[y_cols].astype(int).astype(str)
    return x_df, y_df

def get_class_weight(y_df):
    class_weight = {str(_): 0 for _ in range(8)}
    k,v = np.unique(y_df, return_counts=True)
    for _k,_v in dict(zip(k,v)).items():
        class_weight[str(_k)] += _v
    total_count = sum(class_weight.values())
    class_weight_new = {str(k): total_count/(v+1) for k,v in class_weight.items()}
    return class_weight, class_weight_new

def get_class_weight_for_perf(y_df):
    class_weight = {str(_): 0 for _ in range(8)}
    k,v = np.unique(y_df, return_counts=True)
    for _k,_v in dict(zip(k,v)).items():
        class_weight[str(_k)] += _v
    total_count = sum(class_weight.values())
    class_weight_new = {str(k): v/total_count for k,v in class_weight.items()}
    return class_weight, class_weight_new

def create_model(*, use_model='rf', random_state=42, class_weight=None):
    if use_model == 'rf':
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=random_state, 
            class_weight=class_weight)
    elif use_model == 'xgb':
        model = GradientBoostingClassifier(random_state = random_state)
    elif use_model == 'fastxgb':
        model = HistGradientBoostingClassifier(random_state=random_state)
    elif use_model == 'knn':
        model = KNeighborsClassifier()
    else:
        raise ValueError("Invalid model type. Choose 'rf' or 'xgb'.")
    return model

def create_model_and_train(
        x_train, y_train, 
        *, class_weight, save_at, model_name,
        use_model='rf', random_state=42
        ):
    model = create_model(use_model=use_model, random_state=random_state, class_weight=class_weight)
    model.fit(x_train, y_train)
    joblib.dump(model, os.path.join(save_at, 'model', f'{model_name}.joblib'))
    return model

def load_model(
    save_at, model_name, use_model='rf', random_state=42,
    ):
    model = create_model(use_model=use_model, random_state=random_state)
    model = joblib.load(os.path.join(save_at, 'model', f'{model_name}.joblib'))
    if model is None:
        raise ValueError(f"Model {model_name} not found in {save_at}/model/")
    else:
        print(f"Model {model_name} loaded from {save_at}/model/")
        
    return model


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

def predict_and_evaluate(
        x_test, y_test, 
        *, save_at, model_name, use_model='rf', 
        random_state=42, model=None, stage='', piece_name='', show_cm=False):
    if model is None:
        model = load_model(
            save_at=save_at,
            model_name=model_name,
            use_model=use_model,
            random_state=random_state
        )
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred, labels=[str(i) for i in range(8)])
    class_count, class_weight = get_class_weight_for_perf(y_test)
    performance = get_performance(cm, len(y_test), class_weight, stage, piece_name, use_model)
    
    for i in range(8):
        performance[f'class_{i}_count'] = class_count[str(i)]
        performance[f'class_{i}_weight'] = class_weight[str(i)]

    cm2 = cm / cm.sum(axis=1)[:, np.newaxis]
    print_confusion_matrix(
        cm2,
        f'{model_name}-{use_model}-{stage}-{piece_name}',
        fig_at=os.path.join(save_at, 'fig'), 
        show=show_cm, 
        label_list=LABEL_LST
    )
    return cm, performance

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
    macro_7 = performance['texture_obj_aprf'][1:,:].sum(axis=0)  / 7
    macro_8 = performance['texture_obj_aprf'][:,:].sum(axis=0) / 8
    class_weight_int = [class_weight[str(k)] for k in range(8)]
    tp = sum(performance['texture_obj'][:,0] * class_weight_int)
    fp = sum(performance['texture_obj'][:,1] * class_weight_int)
    fn = sum(performance['texture_obj'][:,2] * class_weight_int)
    tn = sum(performance['texture_obj'][:,3] * class_weight_int)
    t_acc = (tp+tn) / (tp+fp+fn+tn) 
    t_precision = tp / (tp+fp) if (tp+fp)>0 else None
    t_recall = tp / (tp+fn) if (tp+fn)>0 else None
    t_f1 =  None if ((t_precision is None) or (t_recall is None)) else 2*t_precision*t_recall / (t_precision + t_recall)
            
    performance.update({
        # divide to 7
        f'texture_acc_macro_7': macro_7[0],
        f'texture_precision_macro_7': macro_7[1], 
        f'texture_recall_macro_7': macro_7[2],
        f'texture_f1_macro_7': 2*macro_7[1]*macro_7[2]/(macro_7[1]+macro_7[2]) if macro_7[1]+macro_7[2]>0 else None,
        # divide to 8
        f'texture_acc_macro_8': macro_8[0],
        f'texture_precision_macro_8': macro_8[1],
        f'texture_recall_macro_8': macro_8[2],
        f'texture_f1_macro_8': 2*macro_8[1]*macro_8[2]/(macro_8[1]+macro_8[2]) if macro_8[1]+macro_8[2]>0 else None,
        # calculate by weight
        f'texture_acc_macro_weight': t_acc,
        f'texture_precision_macro_weight': t_precision,
        f'texture_recall_macro_weight': t_recall,
        f'texture_f1_macro_weight': t_f1,
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


def exp(
    ds, model_name, train_pieces, test_pieces,
    dataset,
    x_cols, y_cols, al_piece_names, s3_piece_names,
    use_model='rf', skip_0=False, skip_labels=[]
    ):
    if ds not in ['orchestration', 's3', 'all']:
        raise ValueError(f"Invalid ds {ds}!")

    save_at = os.path.join('result', 'rf-2025-06-24', f'{use_model}-{model_name}')
    os.makedirs(save_at, exist_ok=True)
    os.makedirs(os.path.join(save_at, 'model'), exist_ok=True)
    os.makedirs(os.path.join(save_at, 'fig'), exist_ok=True)

    train_df = get_combine_df(train_pieces, dataset, skip_0, al_piece_names, s3_piece_names, skip_labels=skip_labels)
    # train_df = get_combine_df(train_pieces, dataset, skip_0, al_piece_names, s3_piece_names, skip_labels=skip_labels)
    # 隨機減少 20% 的訓練資料
    print('train', train_df.shape)
    ori_amount = train_df.shape[0]
    train_df = train_df.sample(n=int(ori_amount*0.8), random_state=42)
    print('train', train_df.shape)
    x_train, y_train = get_x_y_df(train_df, x_cols, y_cols)
    _, class_weight = get_class_weight(y_train)
    # class_weight = None
    print(f"X | Class weight: {class_weight} | {_}")

    test_df = get_combine_df(test_pieces, dataset, skip_0, al_piece_names, s3_piece_names, skip_labels=skip_labels)
    print('test', test_df.shape)
    x_test, y_test = get_x_y_df(test_df, x_cols, y_cols)
    class_weight = None
    print(f"Y | Class weight: {get_class_weight(y_test)}")
    model = create_model_and_train(
        x_train, y_train, 
        class_weight=class_weight,
        save_at=save_at,
        model_name=model_name,
        use_model=use_model,
        random_state=42,
    )
    
    # Record all results
    all_result = []
    
    # test set
    cm, performance = predict_and_evaluate(
        x_test, y_test, 
        save_at=save_at,
        model_name=model_name,
        use_model=use_model,
        random_state=42,
        stage='test',
        piece_name='all',
        show_cm=False,
    )
    all_result.append(performance)

    # train set
    cm, performance = predict_and_evaluate(
        x_train, y_train,
        save_at=save_at,
        model_name=model_name,
        use_model=use_model,
        random_state=42,
        stage='train',
        piece_name='all',
        show_cm=False,
    )
    all_result.append(performance)

    # test on orch each piece
    for piece in al_piece_names + s3_piece_names:
        df = get_combine_df([piece], dataset, skip_0, al_piece_names, s3_piece_names, skip_labels=skip_labels)
        print(f"{piece=}|{df.shape=}")
        x, y = get_x_y_df(df, x_cols, y_cols)
        cm, performance = predict_and_evaluate(
            x, y,     
            save_at=save_at,
            model_name=model_name,
            use_model=use_model,
            random_state=42,
            stage='train' if piece in train_pieces else 'test',
            piece_name=piece,
            show_cm=False,
        )
        all_result.append(performance)
    # save all results   
    pd.DataFrame(all_result, columns=PERFORMANCE_DF_COLS).to_csv(
        os.path.join(save_at, f'{model_name}-{use_model}-performance.csv'))
    
    # test on all s3
    df = get_combine_df(s3_piece_names, dataset, skip_0, al_piece_names, s3_piece_names, skip_labels=skip_labels)
    x, y = get_x_y_df(df, x_cols, y_cols)
    cm, performance = predict_and_evaluate(
        x, y,     
        save_at=save_at,
        model_name=model_name,
        use_model=use_model,
        random_state=42,
        stage='train' if piece in train_pieces else 'test',
        piece_name='s3_all',
        show_cm=False,
    )
    all_result.append(performance)
    # test on all orch
    df = get_combine_df(al_piece_names, dataset, skip_0, al_piece_names, s3_piece_names, skip_labels=skip_labels)
    x, y = get_x_y_df(df, x_cols, y_cols)
    cm, performance = predict_and_evaluate(
        x, y,     
        save_at=save_at,
        model_name=model_name,
        use_model=use_model,
        random_state=42,
        stage='train' if piece in train_pieces else 'test',
        piece_name='orch_all',
        show_cm=False,
    )
    all_result.append(performance)
    # save all results   
    pd.DataFrame(all_result, columns=PERFORMANCE_DF_COLS).to_csv(
        os.path.join(save_at, f'{model_name}-{use_model}-performance.csv'))



def main():
    experiment_name = '2025-07-01_'
    k_fold = False
    skip_0 = True
    is_debug = False
    use_model = 'rf'
    exp_dct = {
        0: {
            'model_name': 'RF(alone)', 
            'x_cols': STAT_FEATURE_COLS['track_bar'] + STAT_FEATURE_COLS['all_track_bar'] + STAT_FEATURE_COLS['instrument'],
            'y_cols': STAT_FEATURE_COLS['y'],
            },
        1: {
            'model_name': 'CNN+RF', 
            'x_cols': STAT_FEATURE_COLS['track_bar'] + STAT_FEATURE_COLS['all_track_bar'] + STAT_FEATURE_COLS['instrument']+ STAT_FEATURE_COLS['cnn_feature'],
            'y_cols': STAT_FEATURE_COLS['y'],
            },
    }
    skip_labels = []
    if not k_fold:
        ds = 'orchestration'  # train on this dataset
        train_pieces = [
            'beethoven_op36-1', 'beethoven_op55-1', 'beethoven_op60-1', 'beethoven_op67-1',
            'beethoven_op68-1', 'beethoven_op92-1', 'beethoven_op93-1', 'beethoven_op125-1',
            'hob100-1', 'hob101-1', 'hob103-1', 'hob104-1',
            'k551-1', 'k543-1', 'k550-1',
        ]
        test_pieces= [
            'k504-1', 'hob099-1', 'beethoven_op21-1',
        ]
        dataset = {}
        from ..settings.s3_info import piece_names, string_to_int, int_to_string, file_path
        dataset['s3'] = {
            piece_name: load_a_piece(
            os.path.join('converted_dataset', f'rf', 's3'),
            piece_name,
            's3',
            file_path
            ) for piece_name in piece_names
        }
        piece_names_s3 = piece_names
        file_path_s3 = file_path
        from ..settings.orchestration_info import piece_names, string_to_int, int_to_string, file_path
        dataset['orchestration'] = {
            piece_name: load_a_piece(
            os.path.join('converted_dataset', f'rf','orchestration'),
            piece_name,
            'orchestration',
            file_path
            ) for piece_name in piece_names
        }
        piece_names_orchestration = piece_names
        file_path_orchestration = file_path
        
        for exp_index, exp_d in exp_dct.items():
            model_name = experiment_name + exp_d['model_name']
            x_cols = exp_d['x_cols']
            y_cols = exp_d['y_cols']
            print(f"Exp {exp_index} | Model {use_model} | Train {train_pieces} | Test {test_pieces}")
            exp(
                ds,
                model_name, 
                train_pieces, 
                test_pieces,
                dataset,
                x_cols,
                y_cols,
                piece_names_orchestration,
                piece_names_s3,
                use_model,
                skip_0,
                skip_labels
            )
            if is_debug:
                break           
    else:
        # left one piece out
        for exp_index, exp_d in exp_dct.items():
            x_cols = exp_d['x_cols']
            y_cols = exp_d['y_cols']
            for test_index, test_piece in enumerate(piece_names_orchestration):
                # load data
                EXP_DATE = '2025-06-19'
                dataset = {
                    'orchestration':{
                        piece_name: load_a_piece(
                        os.path.join('converted_dataset', f'cnn_rf-{EXP_DATE}-yc40-kfold', f'fold-{test_index}','orchestration'),
                        piece_name,
                        'orchestration',
                        file_path_orchestration
                        ) for piece_name in piece_names_orchestration
                    },
                    's3':{
                        piece_name: load_a_piece(
                        os.path.join('converted_dataset', f'cnn_rf-{EXP_DATE}-yc40-kfold', f'fold-{test_index}', 's3'),
                        piece_name,
                        's3',
                        file_path_s3
                        ) for piece_name in file_path_s3
                    }
                }
                model_name = experiment_name + exp_d['model_name'] + f'-fold_{test_index}' + "-yc40"
                train_pieces = file_path_orchestration[:test_index] + file_path_orchestration[test_index+1:]
                test_pieces = [test_piece]
                print(f"Exp {exp_index} | Model {use_model} | Train {train_pieces} | Test {test_pieces}")
                exp(
                    ds,
                    model_name, 
                    train_pieces, 
                    test_pieces,
                    dataset,
                    x_cols,
                    y_cols,
                    file_path_orchestration,
                    file_path_s3,
                    use_model,
                    skip_0,
                    skip_labels
                )
                if is_debug:
                    break           
            if is_debug:
                break

if __name__=='__main__':
    main()

