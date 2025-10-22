"""
File: evaluation.py
Author: Zih-Syuan Lin (2025)
"""
PERFORMANCE_DF_COLS = [
    'epoch', 'loss',
    'stage', 'piece_name', 'valid_bar_num', 'accuracy',
    'texture_precision_macro_weight', 'texture_recall_macro_weight', 'texture_f1_macro_weight',
    'acc_None', 'precision_None', 'recall_None', 'f1_None',
    'acc_Melody', 'precision_Melody', 'recall_Melody', 'f1_Melody',
    'acc_Rhythm', 'precision_Rhythm', 'recall_Rhythm', 'f1_Rhythm', 
    'acc_Melody+Rhythm', 'precision_Melody+Rhythm', 'recall_Melody+Rhythm', 'f1_Melody+Rhythm', 
    'acc_Harmony','precision_Harmony', 'recall_Harmony', 'f1_Harmony',
    'acc_Melody+Harmony','precision_Melody+Harmony', 'recall_Melody+Harmony', 'f1_Melody+Harmony',
    'acc_Rhythm+Harmony', 'precision_Rhythm+Harmony', 'recall_Rhythm+Harmony', 'f1_Rhythm+Harmony', 
    'acc_All','precision_All', 'recall_All', 'f1_All',

    'mel_acc', 'mel_precision', 'mel_recall', 'mel_f1',
    'rhythm_acc', 'rhythm_precision', 'rhythm_recall', 'rhythm_f1',
    'harm_acc', 'harm_precision', 'harm_recall', 'harm_f1',
    
    'texture_acc_micro', 'texture_precision_micro', 'texture_recall_micro', 'texture_f1_micro',

    'texture_acc_macro_7', 'texture_precision_macro_7', 'texture_recall_macro_7', 'texture_f1_macro_7',
    'texture_acc_macro_8', 'texture_precision_macro_8', 'texture_recall_macro_8', 'texture_f1_macro_8',
    'texture_acc_macro_weight', 'texture_precision_macro_weight', 'texture_recall_macro_weight', 'texture_f1_macro_weight',
    # 'label_wise_acc(mel)',
    # 'label_wise_acc(rhythm)',
    # 'label_wise_acc(harm)',
    'class_0_count', 'class_0_weight',
    'class_1_count', 'class_1_weight',
    'class_2_count', 'class_2_weight',
    'class_3_count', 'class_3_weight',
    'class_4_count', 'class_4_weight',
    'class_5_count', 'class_5_weight',
    'class_6_count', 'class_6_weight',
    'class_7_count', 'class_7_weight',
]

PERFORMANCE_DF_COLS_MIDIBERT_TEX = [
    'epoch', 'loss',
    'stage', 'piece_name', 'valid_bar_num', 'accuracy',
    'texture_precision_macro_weight', 'texture_recall_macro_weight', 'texture_f1_macro_weight',
    'acc_None', 'precision_None', 'recall_None', 'f1_None',
    'acc_Melody', 'precision_Melody', 'recall_Melody', 'f1_Melody',
    'acc_Rhythm', 'precision_Rhythm', 'recall_Rhythm', 'f1_Rhythm', 
    'acc_Melody+Rhythm', 'precision_Melody+Rhythm', 'recall_Melody+Rhythm', 'f1_Melody+Rhythm', 
    'acc_Harmony','precision_Harmony', 'recall_Harmony', 'f1_Harmony',
    'acc_Melody+Harmony','precision_Melody+Harmony', 'recall_Melody+Harmony', 'f1_Melody+Harmony',
    'acc_Rhythm+Harmony', 'precision_Rhythm+Harmony', 'recall_Rhythm+Harmony', 'f1_Rhythm+Harmony', 
    'acc_All','precision_All', 'recall_All', 'f1_All',

    'mel_acc', 'mel_precision', 'mel_recall', 'mel_f1',
    'rhythm_acc', 'rhythm_precision', 'rhythm_recall', 'rhythm_f1',
    'harm_acc', 'harm_precision', 'harm_recall', 'harm_f1',
    
    'texture_acc_micro', 'texture_precision_micro', 'texture_recall_micro', 'texture_f1_micro',

    'texture_acc_macro_7', 'texture_precision_macro_7', 'texture_recall_macro_7', 'texture_f1_macro_7',
    'texture_acc_macro_8', 'texture_precision_macro_8', 'texture_recall_macro_8', 'texture_f1_macro_8',
    'texture_acc_macro_weight', 'texture_precision_macro_weight', 'texture_recall_macro_weight', 'texture_f1_macro_weight',
    'texture_acc_macro_weight3', 'texture_precision_macro_weight3', 'texture_recall_macro_weight3', 'texture_f1_macro_weight3',
    # 'label_wise_acc(mel)',
    # 'label_wise_acc(rhythm)',
    # 'label_wise_acc(harm)',
    'class_0_count', 'class_0_weight',
    'class_1_count', 'class_1_weight',
    'class_2_count', 'class_2_weight',
    'class_3_count', 'class_3_weight',
    'class_4_count', 'class_4_weight',
    'class_5_count', 'class_5_weight',
    'class_6_count', 'class_6_weight',
    'class_7_count', 'class_7_weight',
    'cm_texture'
]

PERFORMANCE_DF_COLS_MIDIBERT_MEL = [
    'epoch', 'loss',
    'stage', 'piece_name', 'valid_bar_num', 'accuracy',
    'texture_precision_macro_weight', 'texture_recall_macro_weight', 'texture_f1_macro_weight',
    'acc_Pad', 'precision_Pad', 'recall_Pad', 'f1_Pad',
    'acc_Mel', 'precision_Mel', 'recall_Mel', 'f1_Mel',
    'acc_SubMel', 'precision_SubMel', 'recall_SubMel', 'f1_SubMel', 
    'acc_Background', 'precision_Background', 'recall_Background', 'f1_Background', 
    'texture_acc_micro', 'texture_precision_micro', 'texture_recall_micro', 'texture_f1_micro',
    'texture_acc_macro_7', 'texture_precision_macro_7', 'texture_recall_macro_7', 'texture_f1_macro_7',
    'texture_acc_macro_8', 'texture_precision_macro_8', 'texture_recall_macro_8', 'texture_f1_macro_8',
    'texture_acc_macro_weight', 'texture_precision_macro_weight', 'texture_recall_macro_weight', 'texture_f1_macro_weight',
    'texture_acc_macro_weight3', 'texture_precision_macro_weight3', 'texture_recall_macro_weight3', 'texture_f1_macro_weight3',
    'class_0_count', 'class_0_weight',
    'class_1_count', 'class_1_weight',
    'class_2_count', 'class_2_weight',
    'class_3_count', 'class_3_weight',
    'cm_texture', 'texture_obj'
]