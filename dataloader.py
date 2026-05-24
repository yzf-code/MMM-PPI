import os
import csv
import json
import pickle
import random
import numpy as np
from tqdm import tqdm
import dgl 
import torch
import lmdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(param, skip_head=True):
    dataset = param['dataset']
    split_mode = param['split_mode']
    modality = param['modality']
    is_transductive = param['is_transductive']

    name = 0
    ppi_name = 0
    
    protein_name = {}
    protein_seq_len_list = []
    ppi_dict = {}
    ppi_list = []
    ppi_label_list = []

    class_map = {'reaction':0, 'binding':1, 'ptmod':2, 'activation':3, 'inhibition':4, 'catalysis':5, 'expression':6}

    ppi_path = 'data/{}/protein.actions.{}.txt'.format(dataset,dataset)
    prot_seq_path = 'data/{}/protein.{}.sequences.dictionary.csv'.format(dataset, dataset)
    amino_prot_r_edge_path = 'data/{}/amino_protein.rball.edges.{}.npy'.format(dataset,dataset)
    amino_prot_k_edge_path = 'data/{}/amino_protein.knn.edges.{}.npy'.format(dataset,dataset)
    
    motif_to_aa_map_list_path = 'data/{}/protein.motif_to_aa_map_list.{}.npy'.format(dataset, dataset)

    with open(prot_seq_path) as f:
        reader = csv.reader(f)
        for row in reader:
            protein_name[row[0]] = name
            protein_seq_len_list.append(len(row[1]))
            name += 1

    if os.path.exists("data/{}/{}_ppi.pkl".format(dataset,dataset)):
        with open("data/{}/{}_ppi.pkl".format(dataset,dataset), "rb") as tf:
            ppi_list = pickle.load(tf)
        with open("data/{}/{}_ppi_label.pkl".format(dataset,dataset), "rb") as tf:
            ppi_label_list = pickle.load(tf)
    else:
        for line in tqdm(open(ppi_path)):
            if skip_head:
                skip_head = False
                continue
            line = line.strip().split('\t')
        
            if line[0] < line[1]:
                temp_data = line[0] + "__" + line[1]
            else:
                temp_data = line[1] + "__" + line[0]

            if temp_data not in ppi_dict.keys():
                ppi_dict[temp_data] = ppi_name
                temp_label = [0, 0, 0, 0, 0, 0, 0]
                temp_label[class_map[line[2]]] = 1
                ppi_label_list.append(temp_label)
                ppi_name += 1
            else:
                index = ppi_dict[temp_data]
                temp_label = ppi_label_list[index]
                temp_label[class_map[line[2]]] = 1
                ppi_label_list[index] = temp_label

        for ppi in tqdm(ppi_dict.keys()):
            temp = ppi.strip().split('__')
            ppi_list.append(temp)

        ppi_num = len(ppi_list)
        for i in tqdm(range(ppi_num)):
            seq1_name = ppi_list[i][0]
            seq2_name = ppi_list[i][1]
            ppi_list[i][0] = protein_name[seq1_name]
            ppi_list[i][1] = protein_name[seq2_name]

        with open("data/{}/{}_ppi.pkl".format(dataset,dataset), "wb") as tf:
            pickle.dump(ppi_list, tf)
        with open("data/{}/{}_ppi_label.pkl".format(dataset,dataset), "wb") as tf:
            pickle.dump(ppi_label_list, tf)

    print('buliding single protein graphs...')
    if os.path.exists("data/{}/{}_protein_amino_graphs_{}_lmdb".format(dataset, dataset, modality)):
        protein_amino_graph_data = "data/{}/{}_protein_amino_graphs_{}_lmdb".format(dataset, dataset, modality)
    else:
        protein_amino_graph_data = get_amino_dgl_graph(amino_prot_r_edge_path, amino_prot_k_edge_path, dataset, modality)

    motif_to_aa_mapping_matrix_list = get_motif_to_aa_mapping_matrix(protein_seq_len_list, motif_to_aa_map_list_path, dataset)

    num_proteins = len(protein_seq_len_list)
    print('number of proteins: {}'.format(num_proteins))
    print('spliting data...')
    ppi_split_dict = split_dataset(ppi_list, dataset, split_mode)
    
    print('buliding PPI graph...')
    ppi_gs = {}
    if is_transductive:
        ppi_gs['train_index'] = dgl.to_bidirected(dgl.graph(ppi_list, num_nodes=len(protein_name))).to(device)
    else:
        train_ppi_edges = np.array(ppi_list)[ppi_split_dict['train_index']].tolist()
        ppi_gs['train_index'] = dgl.to_bidirected(dgl.graph(train_ppi_edges, num_nodes=len(protein_name))).to(device)
        
    ppi_gs['val_index'] = dgl.to_bidirected(dgl.graph(ppi_list, num_nodes=len(protein_name))).to(device)
    ppi_gs['test_index'] = dgl.to_bidirected(dgl.graph(ppi_list, num_nodes=len(protein_name))).to(device)  

    return ppi_gs, protein_amino_graph_data, motif_to_aa_mapping_matrix_list, ppi_list, ppi_label_list, ppi_split_dict, num_proteins ,protein_name

class PPI_Dataset(torch.utils.data.Dataset):
    def __init__(self, ppi_list, labels, data_split_idxs, protein_amino_graph_data, motif_to_aa_mapping_matrix_list, do_neg_sample=False):
        self.data_split_ppi_list = np.array(ppi_list)[data_split_idxs]
        self.labels = np.array(labels)[data_split_idxs]
        self.protein_amino_graph_data_lmdb = protein_amino_graph_data
        self.motif_to_aa_mapping_matrix_list = motif_to_aa_mapping_matrix_list
        self.all_ppi_list = self.data_split_ppi_list
        self.all_labels = self.labels
   
    def __len__(self):
        return len(self.all_labels)
    
    def neg_sampling(self, data_split_ppi_list, labels, ppi_list):
        ppi_set = set(map(tuple, ppi_list))
        negative_samples = []
        negative_labels = []
        print('generating neg samples....')
        for (node1, node2), label in tqdm(zip(data_split_ppi_list, labels)):
            while True:
                neg_node1 = random.randint(0, self.num_protein - 1)  
                if (neg_node1, node2) not in ppi_set and (node2, neg_node1) not in ppi_set:  
                    negative_samples.append((neg_node1, node2))
                    negative_labels.append(label)
                    break
        return np.array(negative_samples), np.array(negative_labels)

    def connect_graphs(self, graph1, graph2):
        num_nodes1 = graph1.num_nodes()
        num_nodes2 = graph2.num_nodes()
        combined_graph = dgl.batch([graph1, graph2])
        graph_ids = torch.cat([
            torch.zeros(num_nodes1, dtype=torch.long),  
            torch.ones(num_nodes2, dtype=torch.long)  
        ])
        combined_graph.ndata['graph_id'] = graph_ids
        offset = num_nodes1
        src_nodes = torch.arange(num_nodes1).repeat_interleave(num_nodes2)
        dst_nodes = torch.arange(num_nodes2).repeat(num_nodes1) + offset
        combined_graph.add_edges(src_nodes, dst_nodes)
        return combined_graph
    
    def open_lmdb(self):
        self.env = lmdb.open(self.protein_amino_graph_data_lmdb, readonly=True, create=False, lock=False)
        self.txn = self.env.begin(buffers=True)

    def __getitem__(self, idx):
        ppi_idx = self.all_ppi_list[idx]
        if not hasattr(self, 'txn'):
            self.open_lmdb()
        
        key1 = f"graph_{ppi_idx[0]}"
        data1 = self.txn.get(key1.encode())
        amino_graph1 = pickle.loads(data1)

        key2 = f"graph_{ppi_idx[1]}"
        data2 = self.txn.get(key2.encode())
        amino_graph2 = pickle.loads(data2)
        
        motif_to_aa_mapping_matrix1 = self.motif_to_aa_mapping_matrix_list[ppi_idx[0]]
        motif_to_aa_mapping_matrix2 = self.motif_to_aa_mapping_matrix_list[ppi_idx[1]]
        return ppi_idx, amino_graph1, amino_graph2, motif_to_aa_mapping_matrix1, motif_to_aa_mapping_matrix2, torch.FloatTensor(self.all_labels[idx])
  
def get_amino_dgl_graph(prot_r_edge_path, prot_k_edge_path, dataset, modality):
    if modality == 'all':
        modals = ['seq', 'struct', 'func']
    else:
        modals = modality.split('_') 

    prot_node_path = []
    for m in modals:
        prot_node_path.append('data/{}/amino_protein_{}.nodes.{}.pt'.format(dataset, m, dataset))

    prot_r_edge = np.load(prot_r_edge_path, allow_pickle=True)
    prot_k_edge = np.load(prot_k_edge_path, allow_pickle=True)

    prot_node = []
    for p in prot_node_path:
        prot_node.append(torch.load(p))    

    prot_graph_list = []
    for i in tqdm(range(len(prot_r_edge))):
        prot_seq = []
        num_nodes = prot_node[0][i].shape[0]
        
        for k in range(1, len(prot_node)):
            assert prot_node[0][i].shape[0] == prot_node[k][i].shape[0]

        for j in range(num_nodes-2):
            prot_seq.append((j, j+1))
            prot_seq.append((j+1, j))
        
        for j in range(num_nodes-1):
            prot_seq.append((j, num_nodes-1))
            prot_seq.append((num_nodes-1, j))

        prot_g = dgl.heterograph({('amino_acid', 'SEQ', 'amino_acid') : prot_seq, 
                                    ('amino_acid', 'STR_KNN', 'amino_acid') : prot_k_edge[i],
                                    ('amino_acid', 'STR_DIS', 'amino_acid') : prot_r_edge[i]
                            }, num_nodes_dict={'amino_acid': num_nodes}
                            )   

        for k, m in enumerate(modals):
            prot_g.ndata[f'{m}_h'] = prot_node[k][i]

        prot_graph_list.append(prot_g)
    
    prot_graph_list_path = "data/{}/{}_protein_amino_graphs_{}_lmdb".format(dataset, dataset, modality)
    print(f"Storing graphs to {prot_graph_list_path}...")
    env = lmdb.open(prot_graph_list_path, map_size=int(1e12)) 
    with env.begin(write=True) as txn:
        for idx, graph in tqdm(enumerate(prot_graph_list)):
            data = pickle.dumps(graph)
            key = f"graph_{idx}".encode()
            txn.put(key, data)
    env.close()
    print(f"Successfully stored {len(prot_graph_list)} graphs to {prot_graph_list_path}.")
    return prot_graph_list_path

def get_motif_to_aa_mapping_matrix(protein_seq_len_list, motif_to_aa_map_list_path, dataset):
    cache_filename = "protein_motif_to_aa_map_metrix.pkl"
    cache_path = "data/{}/{}".format(dataset, cache_filename)
    
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as tf:
            motif_to_aa_mapping_matrix_list = pickle.load(tf)
    else:
        motif_to_aa_mapping_matrix_list = []
        motif_to_aa_map_list = np.load(motif_to_aa_map_list_path, allow_pickle=True)
        for i in tqdm(range(len(protein_seq_len_list))):
            protein_seq_len = protein_seq_len_list[i]
            motif_to_aa_map = motif_to_aa_map_list[i]

            aa_indices = []
            motif_indices = []
            cnt=0
            for motif_idx, values in motif_to_aa_map.items():
                aa_indices_list = values['indices']
                aa_indices.extend(aa_indices_list)
                motif_indices.extend([cnt] * len(aa_indices_list))
                cnt+=1
            
            N_aa = protein_seq_len
            N_motif = len(motif_to_aa_map.keys())

            mapping_matrix = torch.sparse_coo_tensor(
                indices=torch.tensor([aa_indices, motif_indices]),
                values=torch.ones(len(aa_indices)),
                size=(N_aa, N_motif)
            ).to_dense() 

            mapping_matrix_T = mapping_matrix.T 
            motif_to_aa_mapping_matrix_list.append(mapping_matrix_T) 
            
        with open(cache_path, "wb") as tf:
            pickle.dump(motif_to_aa_mapping_matrix_list, tf)

    return motif_to_aa_mapping_matrix_list

def split_dataset(ppi_list, dataset, split_mode):
    with open("data/{}/{}_{}.json".format(dataset,dataset, split_mode), 'r') as f:
        ppi_split_dict = json.load(f)
        f.close()

    print("Train_PPI: {} | Val_PPI: {} | Test_PPI: {}".format(len(ppi_split_dict['train_index']), len(ppi_split_dict['val_index']), len(ppi_split_dict['test_index'])))
    return ppi_split_dict