"""
File: deep_models.py
Description: 
    Simple deel models (CNN, LSTM, CRNN, and Transformer) are defined here.
    They are basically adopted from Chu's CNN model (https://github.com/YaHsuanChu/orchestraTextureClassification).
Author: Zih-Syuan (2025)
"""
import math
import torch
import numpy as np
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import Dropout
from sklearn.metrics import confusion_matrix

class CnnModel(torch.nn.Module):
    def __init__(self, input_channels=2, context_measures=0, conv1_channels=15, conv2_channels=15, conv3_channels=10, hidden=15):
        super().__init__()
        '''
            total_measure = 1+context_measures*2
            input shape = (96*total_measure, 128)
            output = (is_mel, is_rhythm, is_harm) 
        '''
        total_measures = 1+context_measures*2
        
        '''shape (in_channels, 96*total_measures, 128) -> (conv1_channels, 24*total_measures, 32)'''
        self.conv1 = torch.nn.Sequential(
                Conv2d(in_channels=input_channels, out_channels=conv1_channels, kernel_size=(3, 3), stride=(2, 2), padding=1),
                ReLU(),
                MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) ) 
        
        '''shape (conv1_channels, 24*total_measures, 32) -> (conv2_channels, 12*total_measures, 8)'''
        self.conv2 = torch.nn.Sequential(
                Conv2d(in_channels=conv1_channels, out_channels=conv2_channels, kernel_size=(3, 3), stride=(1, 2), padding=1),
                ReLU(),
                MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) )
       
        '''shape (conv2_channels, 12*total_measures, 8) -> (conv3_channels, 6*total_measures, 2)'''
        self.conv3 = torch.nn.Sequential( 
                Conv2d(in_channels=conv2_channels, out_channels=conv3_channels, kernel_size=(3, 3), stride=(1, 2), padding=1),
                ReLU(),
                MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) )
        
        '''flatten -> shape = (12*conv3_channels*total_measures,)'''
        '''shape (12*conv3_channels*total_measures, ) -> (3, )'''
        self.out = torch.nn.Sequential(
            Linear(12*conv3_channels*total_measures, hidden),
            ReLU(),
            Linear(hidden, 3),
            torch.nn.Sigmoid() )

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)
        out = self.out(x)

        return out



class CrnnModel(torch.nn.Module):
    def __init__(self, input_channels=2, context_measures=0, conv1_channels=15, conv2_channels=15, conv3_channels=10, hidden=15):
        super().__init__()

        total_measures = 1+context_measures*2
        
        '''shape (in_channels, 96*total_measures, 128) -> (conv1_channels, 24*total_measures, 32)'''
        self.conv1 = torch.nn.Sequential(
                Conv2d(in_channels=input_channels, out_channels=conv1_channels, kernel_size=(3, 3), stride=(2, 2), padding=1),
                ReLU(),
                MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) ) 
        
        '''shape (conv1_channels, 24*total_measures, 32) -> (conv2_channels, 12*total_measures, 8)'''
        self.conv2 = torch.nn.Sequential(
                Conv2d(in_channels=conv1_channels, out_channels=conv2_channels, kernel_size=(3, 3), stride=(1, 2), padding=1),
                ReLU(),
                MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) )
       
        '''shape (conv2_channels, 12*total_measures, 8) -> (conv3_channels, 6*total_measures, 2)'''
        self.conv3 = torch.nn.Sequential( 
                Conv2d(in_channels=conv2_channels, out_channels=conv3_channels, kernel_size=(3, 3), stride=(1, 2), padding=1),
                ReLU(),
                MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) )
        
        self.bilstm = torch.nn.LSTM(
            input_size=conv3_channels*2,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.out = torch.nn.Sequential(
            Linear(hidden*2, 3),
            torch.nn.Sigmoid())

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0,2,1,3)                   # (batch size, measure dim,  channel num, pitch dim)
        x = x.reshape(x.size(0), x.size(1), -1)  # (batcj size, measure dim, channel num * pitch dim)
        x,( _h, _c) = self.bilstm(x)
        x = x[:, 0, :]
        # pick the first vector in x for each element, so x will be (batch size, hidden dim)
        out = self.out(x)

        return out

class BilstmModel(torch.nn.Module):
    def __init__(self, input_channels=2, context_measures=0, hidden=15,
                 dropout=0.0, num_layers=1):
        super().__init__()

        total_measures = 1+context_measures*2
        input_dim = 128 * input_channels
        reduced_dim = 8 * input_channels
        self.input_proj = torch.nn.Linear(input_dim, reduced_dim)  # e.g. 256 → 64
        self.bilstm = torch.nn.LSTM(
            input_size=reduced_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.out = torch.nn.Sequential(
            Linear(hidden*2, 3),
            torch.nn.Sigmoid())

    def forward(self,x):
        x = x.permute(0,2,1,3)                   # (batch size, measure dim,  channel num, pitch dim)
        x = x.reshape(x.size(0), x.size(1), -1)  # (batcj size, measure dim, channel num * pitch dim)
        x = self.input_proj(x)
        x, (_h, _c) = self.bilstm(x)
        x = x[:, 0, :]
        # pick the first vector in x for each element, so x will be (batch size, hidden dim)
        out = self.out(x)
        return out


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


def test_model(model, loader, device='cpu'):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        # for overall accuracy
        total = 0.0
        correct_label_wise = 0.0
        correct_for_all_three_labels = 0.0
        
        # for precision and recall of each label
        TP = np.zeros(3, dtype=np.float32)
        pred_label_count = np.zeros(3, dtype=np.float32)
        gt_label_count = np.zeros(3, dtype=np.float32)
        
        # for predicted labels analysis including confusion matrix
        predicted_class = []
        gt_class = []

        for i, (x, y_gt) in enumerate(loader):
            x = x.to(device)
            y_gt = y_gt.to(device)
            y_pred = model(x)
            y_pred = y_pred > 0.5
            total += x.shape[0]
            same = (y_gt==y_pred)
            
            # calculate overall accuracy
            correct_label_wise += same.sum()
            number_of_correct_labels_per_data = same.sum(axis=1)
            correct_for_all_three_labels += (number_of_correct_labels_per_data==3).sum()
             
            # calculate Precision and Recall for each label
            gt_label_count += y_gt.sum(axis=0).cpu().numpy()
            pred_label_count += y_pred.sum(axis=0).cpu().numpy()
            TP += (torch.logical_and(same,y_gt)).sum(axis=0).cpu().numpy()
                
            # for confusion matrix
            predicted_class.append( three_bool_to_eight_class(y_pred) )
            gt_class.append( three_bool_to_eight_class(y_gt) )
            
        # overall accuracy
        label_wise_acc = (correct_label_wise/(total*3)).cpu().item()
        data_wise_acc = (correct_for_all_three_labels/total).cpu().item()
        print('label-wise accuracy = {:.3f}'.format(label_wise_acc))
        print('data-wise accuracy (all three labels are correct) = {:.3f}'.format(data_wise_acc))
        
        # precision and recall for each roles
        precision_mel = -1 if pred_label_count[0]==0 else TP[0]/pred_label_count[0]
        recall_mel = -1 if gt_label_count[0]==0 else TP[0]/gt_label_count[0]
        precision_rhythm = -1 if pred_label_count[1]==0 else TP[1]/pred_label_count[1]
        recall_rhythm = -1 if gt_label_count[1]==0 else TP[1]/gt_label_count[1]
        precision_harm = -1 if pred_label_count[2]==0 else TP[2]/pred_label_count[2]
        recall_harm = -1 if gt_label_count[2]==0 else TP[2]/gt_label_count[2]
        print('(Precision, Recall) of [mel, rhythm, harm] : [({:.3f}, {:.3f}), ({:.3f}, {:.3f}), ({:.3f}, {:.3f})]'.format(\
                precision_mel, recall_mel, precision_rhythm, recall_rhythm, precision_harm, recall_harm))
        
        # confusion matrix
        predicted_class = torch.concatenate(predicted_class, axis=0).cpu().numpy()
        gt_class = torch.concatenate(gt_class, axis=0).cpu().numpy()
        con_mat = confusion_matrix(gt_class, predicted_class, normalize='true')
        
        # return testing information
        info_dict = {
            'label_wise_acc': label_wise_acc,
            'data_wise_acc': data_wise_acc,
            'precision_recall_of_role': (
                precision_mel, recall_mel,
                precision_rhythm, recall_rhythm,
                precision_harm, recall_harm
            ),
            'con_mat': con_mat
        }
        return info_dict


    

# To create a Transformer-encoder model, we need:
class SelfAttention(torch.nn.Module):
    def __init__(self, d_model=512, head_num=8):
        super().__init__()
        self.d_model = d_model       # embed_size
        self.head_num = head_num     # heads
        self.d_k = int(d_model/head_num)  # heads_dim
        assert (
            self.d_k * head_num == d_model
        ), "Embedding size needs to be divisible by heads"

        self.W_v = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_q = Linear(d_model, d_model)
        self.fc_out = Linear(d_model, d_model)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        batch_size = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.W_v(values)  # (N, value_len, embed_size)
        keys = self.W_k(keys)  # (N, key_len, embed_size)
        queries = self.W_q(query)  # (N, query_len, embed_size)

        # Split the embedding into self.heads different pieces
        values = values.reshape(batch_size, value_len, self.head_num, self.d_k)
        keys = keys.reshape(batch_size, key_len, self.head_num, self.d_k)
        queries = queries.reshape(batch_size, query_len, self.head_num, self.d_k)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # == similarity, attention
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.d_model ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            batch_size, query_len, self.head_num * self.d_k
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.
        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)
        return out

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, head_num, dropout, forward_expansion):
        super().__init__()
        self.attention = SelfAttention(d_model, head_num)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)

        self.feed_forward = torch.nn.Sequential(
            Linear(d_model, forward_expansion * d_model),
            ReLU(),
            Linear(forward_expansion * d_model, d_model),
        )
        self.dropout = Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        seq_len = x.size(1)  # Extract sequence length
        # Add positional encodings to the input
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)
    

class Transformer(torch.nn.Module):
    def __init__(self,
                 seq_len=512,
                 input_channels=2, 
                 hidden=15,
                 dropout=0.0, 
                 num_layers=2,
                 d_model=32,
                 head_num=4,
                 forward_expansion=4):
        super().__init__()
        self.input_dim = 128 * input_channels
        # self.reduced_dim = 8 * input_channels
        self.seq_len = seq_len
        self.hidden_size = hidden
        self.num_layers = num_layers
        self.d_model = d_model
        self.head_num = head_num
        self.forward_expansion = forward_expansion
        self.input_proj = torch.nn.Linear(self.input_dim, self.d_model)  # e.g. 256 → 64
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=self.seq_len)
        self.decoder = torch.nn.ModuleList(
            [
                TransformerBlock(
                    self.d_model, self.head_num, dropout, self.forward_expansion
                ) for _ in range(num_layers)
            ]
        )
        self.out = torch.nn.Sequential(
            Linear(self.d_model, 3),
            torch.nn.Sigmoid())

    def forward(self, x):
        x = x.permute(0,2,1,3)                   # (batch size, measure dim,  channel num, pitch dim)
        x = x.reshape(x.size(0), x.size(1), -1)  # (batcj size, measure dim, channel num * pitch dim)
        x = self.input_proj(x)

        embeddings = self.positional_encoding(x)
        for transformer_layer in self.decoder:
            embeddings = transformer_layer(embeddings,embeddings,embeddings,mask=None)
        embeddings = embeddings[:, 0, :]
        out = self.out(embeddings)
        return out
