"""
File: trainer.py
Description: 
    Simple deel models (CNN, LSTM, CRNN, and Transformer) are defined here.
Author: Zih-Syuan (2025)
"""

import tqdm
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix

from settings.annotations import LABEL_DCT
from settings.evaluation import PERFORMANCE_DF_COLS
from models.utils import print_confusion_matrix

class MySingleClassTrainer:
    def __init__(self, 
                *,
                model, 
                optimizer,
                scheduler, 
                epoch_beg,
                epoch_end,
                loss_func,
                save_at, 
                is_debug, 
                early_stopping, 
                device='cpu', 
                threshold=0.5, 
                max_patient=1000, 
                reset_patient=False,
                freeze=False,
                do_save_current_stage = True,
                compute_loss_with_msk = True,
                return_performance=False,
                piece_name='',
                class_weight={str(_):1/8 for _ in range(8)},
                class_count={str(_):0 for _ in range(8)},
                ):
        self.model = model
        self.epoch_beg = epoch_beg
        self.epoch_end = epoch_end
        self.device = device
        self.loss_func = loss_func
        self.save_at = save_at
        self.is_debug = is_debug
        self.threshold = threshold
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.epoch = 0
        self.performance_list = []
        self.max_patient = max_patient
        self.reset_patient = reset_patient
        self.window_bar = 1
        self.do_save_current_stage = do_save_current_stage
        self.freeze = freeze
        self.compute_loss_with_msk = compute_loss_with_msk
        self.return_performance= return_performance
        self.class_weight = class_weight
        self.piece_name = piece_name
        self.class_count = class_count
        self.epoch_gap = 5

    def train(self, loader, epoch):
        if self.freeze:
            self.model.eval()
        else:
            self.model.train()
        return self.iteration(loader, epoch, 'train')
    
    def test(self, loader, epoch):
        self.model.eval()
        return self.iteration(loader, epoch, 'test')

    def valid(self, loader, epoch):
        self.model.eval()
        return self.iteration(loader, epoch, 'valid')
    

    def iteration(self, loader, epoch, stage='train'):
        pbar = tqdm.tqdm(loader, desc='Starting...')
        performance = {
            'epoch': epoch,
            'stage': stage,
            'piece_name': self.piece_name,
            'loss': 0, 
            'valid_bar_num': 0,
            # texture
            'correct_texture': 0,
            'count_texture': 0,
            'texture_obj': np.zeros((8, 4)),  # TP, FP, FN, TN
            'texture_obj_aprf': np.zeros((8,4)),
            'cm_texture': np.zeros((8, 8)),
        }
        for i, a_batch in enumerate(pbar):
            x, y_true = a_batch
            this_batch_size = x.size(0)
            x, y_true = x.to(self.device), y_true.to(self.device)
            y_pred = self.model(x)
            loss = self.loss_func(y_pred, y_true)
            if not self.freeze:
                if stage=='train':
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            performance['loss'] += loss.item()
            performance['valid_bar_num'] += this_batch_size
            performance = self.record_performance_for_a_batch(
                y_true,
                y_pred,
                performance
            )
            pbar.set_description(f'{i+1:>8}/{len(loader):>8} | loss {loss.item():>.6f}]')
            if self.is_debug and i==2:
                break 
        info_dict=None
        performance = self.get_performance(performance, self.class_weight)
        self.performance = performance
        for i in range(8):
            performance[f'class_{i}_count'] = self.class_count[str(i)]
            performance[f'class_{i}_weight'] = self.class_weight[str(i)]

        if self.do_save_current_stage:
            # save result and model
            self.performance_list.append([
                performance[_] for _ in PERFORMANCE_DF_COLS
            ])
            if self.epoch%self.epoch_gap==0 or self.epoch==self.epoch_end:
                self.save_current_stage(performance)
            if self.epoch>=self.epoch_end-1:
                torch.save(self.model.state_dict(), f'{self.save_at}/model/epoch{str(self.epoch)}.pt') 

        if self.return_performance:
            return performance['loss'], performance['accuracy'], performance, info_dict
        else:
            return performance['loss'], performance['accuracy']

    def record_performance_for_a_batch(
            self, 
            y_true,
            y_pred,
            performance
        ):
        y_pred_3 = y_pred > self.threshold
        y_pred_8 = y_pred_3[:,0] + y_pred_3[:,1]*2 + y_pred_3[:,2]*4
        y_true_8 = y_true[:,0] + y_true[:,1]*2 + y_true[:,2]*4
        y_pred_8 = y_pred_8.detach().cpu().numpy().astype(int)
        y_true_8 = y_true_8.detach().cpu().numpy().astype(int)
        # store result
        cm = confusion_matrix(
            y_true_8,
            y_pred_8,
            labels=list(range(8))
            )
        for _index in range(8):
            tp = cm[_index, _index]
            fp = cm[:, _index].sum() - tp
            fn = cm[_index, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            performance['texture_obj'][_index] += np.array([tp, fp, fn, tn])
        performance['cm_texture'] += cm
        return performance

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
                self.early_stopping['patient'] -= 1
        if self.early_stopping['patient'] == 0:
            print(f'early stopping at epoch {self.epoch} with {self.early_stopping}, best_epoch is {self.early_stopping["beat_epoch"]}')
            torch.save(self.model.state_dict(), f'{self.save_at}/model/epoch{str(self.epoch)}.pt') 
            return True
        return False

    def get_performance(self, performance, class_weight):
        performance['correct_texture'] = 0
        # 8 classes
        print(f"{' '*20} | {'acc':>6} | {'prec':>6} | {'recall':>6} | {'f1':>6}")
        print("--"*30)
        for _ in range(8):
            tp, fp, fn, tn = performance['texture_obj'][_]
            t_acc = (tp+tn) / (tp+fp+fn+tn)  # data-wise acc
            t_precision = tp / (tp+fp) if (tp+fp)>0 else None
            t_recall = tp / (tp+fn) if (tp+fn)>0 else None
            t_f1 =  None if ((t_precision is None) or (t_recall is None)) else 2*t_precision*t_recall / (t_precision + t_recall)
            performance['texture_obj_aprf'][_] += [t_acc, t_precision if t_precision is not None else 0, t_recall if t_recall is not None else 0, t_f1 if t_f1 is not None else 0]
            # support: The number of occurrences of each label in y_true
            print(f"{LABEL_DCT[_]['meaning']:<20} | {t_acc:>6.2f} | {t_precision if t_precision is not None else -1:>6.2f} | {t_recall if t_recall is not None else -1:>6.2f} | {t_f1 if t_f1 is not None else -1:>6.2f}")
            label_name = LABEL_DCT[_]['meaning']
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
        # # macro-average: each class has same weight
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
            # divide to 7
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
        df = pd.DataFrame(self.performance_list, columns=PERFORMANCE_DF_COLS)
        df.to_csv(f'{self.save_at}/performance_2025-06-16.csv')
        # 8 class cm
        np.save(f"{self.save_at}/confusion_matrix/epoch{str(performance['epoch'])}_{performance['stage']}_texture.npy", performance['cm_texture'])
        performance['cm_texture_new'] = performance['cm_texture'] / performance['cm_texture'].sum(axis=1, keepdims=True)
        print_confusion_matrix(performance['cm_texture_new'], f"epoch{str(performance['epoch'])}_{performance['stage']}_texture", f'{self.save_at}/fig', show=False)

    def print_confusion_matrix_simple(self, epoch, stage, 
            performance, name, show=False, label=['1', '0']):
        tp, fp, fn, tn = performance[name]
        cmtx = np.array([[tp/(tp+fp), fp/(tp+fp)], [fn/(fn+tn), tn/(fn+tn)]])
        tp, fp, fn, tn = cmtx.ravel()

        labels = np.array(
            [
                ["TP\n{:.2f}".format(tp), "FP\n{:.2f}".format(fp)], 
                ["FN\n{:.2f}".format(fn), "TN\n{:.2f}".format(tn)]
        ])

        # Confusion Matrix
        fig, ax = plt.subplots(figsize=(5,5))
        cax = ax.matshow(cmtx, cmap='Blues')
        norm = plt.Normalize(vmin=cmtx.min(), vmax=cmtx.max())
        for i in range(2):
            for j in range(2):
                text_color = 'white' if norm(cmtx[i, j]) > 0.5 else 'black'
                ax.text(j, i, labels[i, j], va='center', ha='center', fontsize=12, color=text_color)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(label, fontsize=10)
        ax.set_yticklabels(label, fontsize=10)
        ax.set_xlabel('Predicted Labels', fontsize=10)
        ax.set_ylabel('True Labels', fontsize=10)
        ax.set_title(f'{name} {epoch} {stage}', fontsize=12)
        plt.colorbar(cax)
        plt.tight_layout()
        plt.savefig(f'{self.save_at}/fig/{epoch}_{stage}_{name}.png')
        if show:
            plt.show()
        plt.close()


