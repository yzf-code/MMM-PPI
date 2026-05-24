from collections import OrderedDict
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dgl.nn.pytorch import GraphConv, GINConv, HeteroGraphConv
from utils import get_global_mapping_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GIN(nn.Module):
    def __init__(self, param):
        super(GIN, self).__init__()
        self.num_layers = param['ppi_num_layers']
        self.dropout = nn.Dropout(param['ppi_dropout_ratio'])
        self.layers = nn.ModuleList()
        
        if param['modality'] == 'all':
            modals = ['seq', 'struct', 'func']
        else:
            modals = param['modality'].split('_')

        if len(modals) == 1:
            input_dim = param[f'protein_emb_dim_{modals[0]}']
        else:
            input_dim = sum([param[f'protein_emb_dim_{m}'] for m in modals])
            
        self.layers.append(GINConv(nn.Sequential(nn.Linear(input_dim , param['ppi_hidden_dim']), 
                                                 nn.ReLU(), 
                                                 nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim']), 
                                                 nn.ReLU(), 
                                                 nn.BatchNorm1d(param['ppi_hidden_dim'])), 
                                                 aggregator_type='sum', 
                                                 learn_eps=True))

        for i in range(self.num_layers - 1):
            self.layers.append(GINConv(nn.Sequential(nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim']), 
                                                     nn.ReLU(), 
                                                     nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim']), 
                                                     nn.ReLU(), 
                                                     nn.BatchNorm1d(param['ppi_hidden_dim'])), 
                                                     aggregator_type='sum', 
                                                     learn_eps=True))

        self.linear = nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim'])
        self.fc = nn.Linear(param['ppi_hidden_dim'], param['output_dim'])

    def forward(self, g, x, ppi_list, idx):
        for layer in self.layers:
            x = layer(g, x)
            x = self.dropout(x)

        x = F.dropout(F.relu(self.linear(x)), p=0.5, training=self.training)
        node_id = np.array(ppi_list)[idx]
        x1 = x[node_id[:, 0]]
        x2 = x[node_id[:, 1]]
        x = self.fc(torch.mul(x1, x2))
        return x
    
class HGNN(nn.Module):
    def __init__(self, param, modality):
        super(HGNN, self).__init__()
        self.dropout = nn.Dropout(param['dropout_ratio'])
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.fc = nn.ModuleList()
        self.modality = modality
        
        if modality == 'seq':
            self.init_dim = param['prot_seq_emb_dim'] 
            self.num_layers = param['amino_gnn_layers_seq']
            self.hgnn_embed_gim = param['protein_emb_dim_seq']
        elif modality == 'struct':
            self.init_dim = param['prot_struct_emb_dim'] 
            self.num_layers = param['amino_gnn_layers_struct']
            self.hgnn_embed_gim = param['protein_emb_dim_struct']
        elif modality == 'func':
            self.init_dim = param['prot_func_emb_dim'] 
            self.num_layers = param['amino_gnn_layers_func']
            self.hgnn_embed_gim = param['protein_emb_dim_func']

        self.norms.append(nn.BatchNorm1d(self.hgnn_embed_gim))
        self.fc.append(nn.Linear(self.hgnn_embed_gim, self.hgnn_embed_gim))
        self.layers.append(HeteroGraphConv({'SEQ' : GraphConv(self.init_dim, self.hgnn_embed_gim), 
                                            'STR_KNN' : GraphConv(self.init_dim, self.hgnn_embed_gim), 
                                            'STR_DIS' : GraphConv(self.init_dim, self.hgnn_embed_gim),
                                            }, aggregate='sum'))

        for i in range(self.num_layers - 1):
            self.norms.append(nn.BatchNorm1d(self.hgnn_embed_gim))
            self.fc.append(nn.Linear(self.hgnn_embed_gim, self.hgnn_embed_gim))
            self.layers.append(HeteroGraphConv({'SEQ' : GraphConv(self.hgnn_embed_gim, self.hgnn_embed_gim), 
                                                'STR_KNN' : GraphConv(self.hgnn_embed_gim, self.hgnn_embed_gim), 
                                                'STR_DIS' : GraphConv(self.hgnn_embed_gim, self.hgnn_embed_gim),
                                                }, aggregate='sum'))

    def forward(self, batch_graph, modality):
        x = batch_graph.ndata[f'{modality}_h']
        for l, layer in enumerate(self.layers):
            x = layer(batch_graph, {'amino_acid': x})
            x = self.norms[l](F.relu(self.fc[l](x['amino_acid'])))
            if l != self.num_layers - 1:
                x = self.dropout(x)
           
        batch_graph.ndata[f'{self.modality}_repr'] = x
        return batch_graph.ndata[f'{self.modality}_repr']
    

class MLP(nn.Module):
    def __init__(self, inp_dim, hidden_dim, out_dim, num_layers=3, dropout=0.1):
        super().__init__()
        layer_list = OrderedDict()
        in_dim = inp_dim
        for l in range(num_layers):
            if l < num_layers - 1:
                layer_list['fc{}'.format(l)] = nn.Linear(in_dim, hidden_dim)
                if dropout > 0:
                    layer_list['drop{}'.format(l)] = nn.Dropout(p=dropout)
                in_dim = hidden_dim
            else:
                layer_list['fc_score'] = nn.Linear(in_dim, out_dim)
        self.network = nn.Sequential(layer_list)

    def forward(self, emb):
        out = self.network(emb)
        return out

class CoAttentionNetwork(nn.Module):
    def __init__(self, emb_dim, co_attn_dim):
        super(CoAttentionNetwork, self).__init__()
        self.W_q = nn.Parameter(torch.randn(co_attn_dim, emb_dim))
        self.W_v = nn.Parameter(torch.randn(co_attn_dim, emb_dim))
        self.W_mq = nn.Parameter(torch.randn(co_attn_dim, emb_dim))
        self.W_hq = nn.Parameter(torch.randn(1, co_attn_dim))
        self.W_mv = nn.Parameter(torch.randn(co_attn_dim, emb_dim))
        self.W_hv = nn.Parameter(torch.randn(1, co_attn_dim))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, agg_features1, agg_features2, mask1, mask2):
        graph1_embed = agg_features1[:,0,:].unsqueeze(1)
        graph2_embed = agg_features2[:,0,:].unsqueeze(1)
        node_features1 = agg_features1[:,1:,:]
        node_features2 = agg_features2[:,1:,:]

        m0 = graph1_embed.transpose(1,2) * graph2_embed.transpose(1,2)

        h_a = self.tanh(self.W_v @ node_features1.transpose(1,2)) * self.tanh(self.W_mv @ m0) 
        h_b = self.tanh(self.W_q @ node_features2.transpose(1,2)) * self.tanh(self.W_mq @ m0)

        alpha_a = self.W_hv @ h_a  
        alpha_b = self.W_hq @ h_b 

        if mask1 is not None:
            mask1 = mask1[:, 1:].unsqueeze(1) 
            alpha_a = alpha_a.masked_fill(mask1 == 0, float('-inf'))
        if mask2 is not None:
            mask2 = mask2[:, 1:].unsqueeze(1) 
            alpha_b = alpha_b.masked_fill(mask2 == 0, float('-inf'))
        
        alpha_a = self.softmax(alpha_a) 
        alpha_b = self.softmax(alpha_b)
        
        D_a = self.tanh(alpha_a @ node_features1)
        D_b = self.tanh(alpha_b @ node_features2)
        
        D_a = D_a.squeeze(1)
        D_b = D_b.squeeze(1)
        
        return D_a, D_b

class Pair_wise_Encoder(torch.nn.Module):
    def __init__(self, param):
        super().__init__()
        self.modality = param['modality']
        
        if self.modality == 'all':
            self.modals = ['seq', 'struct', 'func']
        else:
            self.modals = self.modality.split('_')
            
        self.amino_gnn_model_dict = nn.ModuleDict()
        for m in self.modals:
            self.amino_gnn_model_dict[m] = HGNN(param, m).to(device)
            
        if len(self.modals) == 1:
            self.emb_dim = param[f'protein_emb_dim_{self.modals[0]}']
        else:
            self.emb_dim = sum([param[f'protein_emb_dim_{m}'] for m in self.modals])

        self.score_dim = self.emb_dim
        self.co_attn_dim = param['co_attn_dim']
        self.proja = nn.Linear(self.score_dim, self.score_dim, bias=False)
        self.projb = nn.Linear(self.score_dim, self.score_dim, bias=False)
        self.co_attn = CoAttentionNetwork(self.score_dim, self.co_attn_dim)
        
    def amino_graph_message_passing_and_merged_to_segment(self, batched_amino_graph, global_motif_to_aa_mapping_matrix, motif_batch_list):
        for modal in self.modals:
            batched_amino_graph.ndata[f'{modal}_repr'] = self.amino_gnn_model_dict[modal](batched_amino_graph, modal)

        if len(self.modals) == 1:
            aa_features = batched_amino_graph.ndata[f'{self.modals[0]}_repr'] 
        else:
            repr_list = [batched_amino_graph.ndata[f'{m}_repr'] for m in self.modals]
            aa_features = torch.concat(repr_list, dim=-1)
        
        all_segment_features = torch.sparse.mm(global_motif_to_aa_mapping_matrix, aa_features)  
        all_segment_features = F.normalize(all_segment_features, p=2, dim=1)  
        segment_feature_list = torch.split(all_segment_features, motif_batch_list.tolist())
        
        padded_segment_features = torch.nn.utils.rnn.pad_sequence(segment_feature_list, batch_first=True, padding_value=0)
        padded_mask = (padded_segment_features == 0).all(dim=-1).to(device)

        agg_features = padded_segment_features
        agg_mask = ~padded_mask
        return agg_features, agg_mask

    def encoding(self, batched_amino_graph1, batched_amino_graph2, batched_motif_to_aa_mapping_matrix1_list, batched_motif_to_aa_mapping_matrix2_list):
        global_motif_to_aa_mapping_matrix1, motif_batch_list1 = get_global_mapping_matrix(batched_motif_to_aa_mapping_matrix1_list)
        global_motif_to_aa_mapping_matrix2, motif_batch_list2 = get_global_mapping_matrix(batched_motif_to_aa_mapping_matrix2_list)
        
        concated_segment_features1, segment_mask1 = self.amino_graph_message_passing_and_merged_to_segment(batched_amino_graph1, global_motif_to_aa_mapping_matrix1, motif_batch_list1)
        concated_segment_features2, segment_mask2 = self.amino_graph_message_passing_and_merged_to_segment(batched_amino_graph2, global_motif_to_aa_mapping_matrix2, motif_batch_list2)
        
        concated_segment_features1 = self.proja(concated_segment_features1)
        concated_segment_features2 = self.projb(concated_segment_features2)

        D_a, D_b = self.co_attn(concated_segment_features1, concated_segment_features2, segment_mask1, segment_mask2)
        return D_a, D_b

    def forward(self, all_ppi_dataloader, num_proteins):
        prot_embed_dict = {i: [] for i in range(num_proteins)}

        for iter, (ppi_idx_list, batched_amino_graph1, batched_amino_graph2, batched_motif_to_aa_mapping_matrix1_list, batched_motif_to_aa_mapping_matrix2_list, labels) in tqdm(enumerate(all_ppi_dataloader)):
            batched_amino_graph1 = batched_amino_graph1.to(device)
            batched_amino_graph2 = batched_amino_graph2.to(device)

            D_a, D_b = self.encoding(batched_amino_graph1, batched_amino_graph2, batched_motif_to_aa_mapping_matrix1_list, batched_motif_to_aa_mapping_matrix2_list)
            D_a = D_a.detach().cpu()
            D_b = D_b.detach().cpu()

            for idx, (protein_id_1, protein_id_2) in enumerate(ppi_idx_list):
                prot_embed_dict[protein_id_1].append(D_a[idx])
                prot_embed_dict[protein_id_2].append(D_b[idx])
            
        sorted_keys = sorted(prot_embed_dict.keys())
        averaged_values = []
        for key in sorted_keys:
            if len(prot_embed_dict[key]) > 0:
                averaged_values.append(torch.mean(torch.stack(prot_embed_dict[key]), dim=0))
            else:
                placeholder_shape = torch.Size([self.score_dim]) 
                averaged_values.append(torch.zeros(placeholder_shape))

        prot_embeds = torch.stack(averaged_values)
        return prot_embeds
        
class Pair_wise_Classifier_model(torch.nn.Module):
    def __init__(self, param):
        super().__init__()
        self.pair_wise_encoder = Pair_wise_Encoder(param)

        self.n_rel = param['output_dim']
        pred_dim = 2 * self.pair_wise_encoder.score_dim
        self.dropout_linear = nn.Linear(pred_dim, pred_dim)
        self.W_final = MLP(pred_dim, int(pred_dim/2), self.n_rel) 
    
    def forward(self, batched_amino_graph1, batched_amino_graph2, batched_motif_to_aa_mapping_matrix1_list, batched_motif_to_aa_mapping_matrix2_list):
        D_a, D_b = self.pair_wise_encoder.encoding(batched_amino_graph1, batched_amino_graph2, batched_motif_to_aa_mapping_matrix1_list, batched_motif_to_aa_mapping_matrix2_list)

        pred = torch.cat([D_a, D_b], dim=1)
        pred = F.dropout(F.relu(self.dropout_linear(pred)), p=0.5, training=self.training)
        scores = self.W_final(pred)
        return scores