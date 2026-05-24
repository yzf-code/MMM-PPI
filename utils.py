import os
import dgl
import torch
import shutil
import random
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

def check_writable(path, overwrite=True):
    if not os.path.exists(path):
        os.makedirs(path)
    elif overwrite:
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        pass
        
def evaluat_metrics(output, label):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    pre_y = (torch.sigmoid(output) > 0.5).numpy()
    truth_y = label.numpy()
    N, C = pre_y.shape

    for i in range(N):
        for j in range(C):
            if pre_y[i][j] == truth_y[i][j]:
                if truth_y[i][j] == 1:
                    TP += 1
                else:
                    TN += 1
            elif truth_y[i][j] == 1:
                FN += 1
            elif truth_y[i][j] == 0:
                FP += 1

    Precision = TP / (TP + FP + 1e-10)
    Recall = TP / (TP + FN + 1e-10)
    F1_score = 2 * Precision * Recall / (Precision + Recall + 1e-10)
    return F1_score

def get_bfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size):
    candiate_node = []
    selected_edge_index = []
    selected_node = []

    random_node = random.randint(0, node_num - 1)
    while len(node_to_edge_index[random_node]) > 20:
        random_node = random.randint(0, node_num - 1)
    candiate_node.append(random_node)

    while len(selected_edge_index) < sub_graph_size:
        cur_node = candiate_node.pop(0)
        selected_node.append(cur_node)

        for edge_index in node_to_edge_index[cur_node]:
            if edge_index not in selected_edge_index:
                selected_edge_index.append(edge_index)

                end_node = -1
                if ppi_list[edge_index][0] == cur_node:
                    end_node = ppi_list[edge_index][1]
                else:
                    end_node = ppi_list[edge_index][0]

                if end_node not in selected_node and end_node not in candiate_node:
                    candiate_node.append(end_node)
            else:
                continue
    return selected_edge_index

def get_dfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size):
    stack = []
    selected_edge_index = []
    selected_node = []

    random_node = random.randint(0, node_num - 1)
    while len(node_to_edge_index[random_node]) > 20:
        random_node = random.randint(0, node_num - 1)
    stack.append(random_node)

    while len(selected_edge_index) < sub_graph_size:
        cur_node = stack[-1]

        if cur_node in selected_node:
            flag = True

            for edge_index in node_to_edge_index[cur_node]:
                if flag:
                    end_node = -1

                    if ppi_list[edge_index][0] == cur_node:
                        end_node = ppi_list[edge_index][1]
                    else:
                        end_node = ppi_list[edge_index][0]

                    if end_node in selected_node:
                        continue
                    else:
                        stack.append(end_node)
                        flag = False
                else:
                    break

            if flag:
                stack.pop()
            continue
        else:
            selected_node.append(cur_node)
            for edge_index in node_to_edge_index[cur_node]:
                if edge_index not in selected_edge_index:
                    selected_edge_index.append(edge_index)
    return selected_edge_index



def get_global_mapping_matrix(batched_motif_to_aa_mapping_matrix_list):
    batched_motif_to_aa_mapping_matrix_list = [t.to(device) for t in batched_motif_to_aa_mapping_matrix_list]
    aa_batch = torch.tensor([tensor.size(1) for tensor in batched_motif_to_aa_mapping_matrix_list])
    motif_batch = torch.tensor([tensor.size(0) for tensor in batched_motif_to_aa_mapping_matrix_list])
    
    aa_offsets = torch.cat([aa_batch.new_zeros(1), aa_batch.cumsum(dim=0)[:-1]], dim=0)
    motif_offsets = torch.cat([motif_batch.new_zeros(1), motif_batch.cumsum(dim=0)[:-1]], dim=0)
    
    total_aa = sum(aa_batch)
    total_motif = sum(motif_batch)
    all_indices = []
    all_values = []
    
    for idx, dense_mapping_matrix in enumerate(batched_motif_to_aa_mapping_matrix_list):
        mapping_matrix = torch.sparse_coo_tensor(dense_mapping_matrix.nonzero().T, dense_mapping_matrix[dense_mapping_matrix != 0], dense_mapping_matrix.size())

        indices = mapping_matrix._indices()  
        values = mapping_matrix._values()  
        aa_offset = aa_offsets[idx]
        motif_offset = motif_offsets[idx]
        
        adjusted_indices = indices.clone()
        adjusted_indices[0, :] += motif_offset  
        adjusted_indices[1, :] += aa_offset  
        
        all_indices.append(adjusted_indices)
        all_values.append(values)

    global_indices = torch.cat(all_indices, dim=1)  
    global_values = torch.cat(all_values)  
    global_mapping_matrix = torch.sparse_coo_tensor(
        global_indices, global_values, size=(total_motif, total_aa)
    )   
    return global_mapping_matrix, motif_batch

def collate_dgl(batch):
    ppi_idx_list, amino_graph1, amino_graph2, batched_motif_to_aa_mapping_matrix1_list, batched_motif_to_aa_mapping_matrix2_list, r_labels = map(list, zip(*batch))
    batched_amino_graph1 = dgl.batch(amino_graph1)
    batched_amino_graph2 = dgl.batch(amino_graph2)
    r_labels = torch.stack(r_labels, dim=0)
    return ppi_idx_list, batched_amino_graph1, batched_amino_graph2, batched_motif_to_aa_mapping_matrix1_list, batched_motif_to_aa_mapping_matrix2_list, r_labels