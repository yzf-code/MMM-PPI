import csv
import time
import json
import math
import copy
import argparse
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import *
from Classifier_model import *
from dataloader import *
from pretrain_Pair_wise_Encoder import pretrain_Pair_wise_Encoder

warnings.filterwarnings("ignore", category=Warning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def trainer(model, ppi_g, prot_embed, ppi_list, labels, index, batch_size, optimizer, loss_fn, epoch):
    f1_sum = 0.0
    loss_sum = 0.0

    batch_num = math.ceil(len(index) / batch_size)
    random.shuffle(index)
    model.train()

    for batch in range(batch_num):
        if batch == batch_num - 1:
            train_idx = index[batch * batch_size:]
        else:
            train_idx = index[batch * batch_size : (batch+1) * batch_size]

        output = model(ppi_g, prot_embed, ppi_list, train_idx)
        loss = loss_fn(output, labels[train_idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        f1_score = evaluat_metrics(output.detach().cpu(), labels[train_idx].detach().cpu())
        f1_sum += f1_score
    return loss_sum / batch_num, f1_sum / batch_num

def evaluator(model, ppi_g, prot_embed, ppi_list, labels, index, batch_size, mode='metric'):
    eval_output_list = []
    eval_labels_list = []
    batch_num = math.ceil(len(index) / batch_size)
    model.eval()

    with torch.no_grad():
        for batch in range(batch_num):
            if batch == batch_num - 1:
                eval_idx = index[batch * batch_size:]
            else:
                eval_idx = index[batch * batch_size : (batch+1) * batch_size]

            output = model(ppi_g, prot_embed, ppi_list, eval_idx)
            eval_output_list.append(output.detach().cpu())
            eval_labels_list.append(labels[eval_idx].detach().cpu())

        f1_score = evaluat_metrics(torch.cat(eval_output_list, dim=0), torch.cat(eval_labels_list, dim=0))

    if mode == 'metric':
        return f1_score
    elif mode == 'output':
        return torch.cat(eval_output_list, dim=0), torch.cat(eval_labels_list, dim=0)


def main(param):
    ppi_gs, protein_amino_graph_data, motif_to_aa_mapping_matrix_list, ppi_list, labels, ppi_split_dict, num_proteins, protein_name_dict = load_data(param)

    train_dataset = PPI_Dataset(ppi_list, labels, ppi_split_dict['train_index'], protein_amino_graph_data, motif_to_aa_mapping_matrix_list, do_neg_sample=False)
    val_dataset = PPI_Dataset(ppi_list, labels, ppi_split_dict['val_index'], protein_amino_graph_data, motif_to_aa_mapping_matrix_list, do_neg_sample=False)
    test_dataset = PPI_Dataset(ppi_list, labels, ppi_split_dict['test_index'], protein_amino_graph_data, motif_to_aa_mapping_matrix_list, do_neg_sample=False)
    all_dataset = PPI_Dataset(ppi_list, labels, list(range(len(ppi_list))), protein_amino_graph_data, motif_to_aa_mapping_matrix_list, do_neg_sample=False)

    train_dataloader = DataLoader(train_dataset, batch_size=param['pretrain_batch_size'], shuffle=True, collate_fn=collate_dgl, num_workers=param['num_workers'])
    val_dataloader = DataLoader(val_dataset, batch_size=param['pretrain_batch_size'], shuffle=False, collate_fn=collate_dgl, num_workers=param['num_workers'])
    test_dataloader = DataLoader(test_dataset, batch_size=param['pretrain_batch_size'], shuffle=False, collate_fn=collate_dgl, num_workers=param['num_workers'])
    all_dataloader = DataLoader(all_dataset, batch_size=param['pretrain_batch_size'], shuffle=False, collate_fn=collate_dgl, num_workers=param['num_workers'])

    if len(param['ckpt_path']) == 0:
        pretrain_Pair_wise_Encoder(param, timestamp, train_dataloader, val_dataloader, test_dataloader)
    
    Pair_wise_model = Pair_wise_Classifier_model(param).to(device)
    
    if len(param['ckpt_path']) == 0 :
        Pair_wise_model.load_state_dict(torch.load(os.path.join("results/{}/{}/Pair_wise_Encoder/".format(param['dataset'], timestamp), "Pair_wise_Encoder_Classifier_model.pth")))
    else:
        Pair_wise_model.load_state_dict(torch.load(param['ckpt_path']))

    print('getting protein embeddings....')
    prot_embed = Pair_wise_model.pair_wise_encoder.forward(all_dataloader, num_proteins).to(device)

    del Pair_wise_model
    torch.cuda.empty_cache()

    output_dir = "results/{}/{}/SEES_{}/".format(param['dataset'], timestamp, param['seed'])
    check_writable(output_dir, overwrite=False)
    log_file = open(os.path.join(output_dir, "train_log.txt"), 'a+')

    model = GIN(param).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(param['learning_rate']), weight_decay=float(param['weight_decay']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5,min_lr=float(param['learning_rate'])*0.1, verbose=True)
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    
    es = 0
    val_best = 0
    test_val = 0
    test_best = 0
    best_epoch = 0
    labels = torch.FloatTensor(np.array(labels)).to(device)
    print('GIN on PPI_graph_training....')
    for epoch in range(1, param["max_epoch"] + 1):
        train_loss, train_f1_score = trainer(model, ppi_gs['train_index'], prot_embed, ppi_list, labels, ppi_split_dict['train_index'], param['batch_size'], optimizer, loss_fn, epoch)
        scheduler.step(train_loss)

        if (epoch - 1) % param['log_num'] == 0:
            val_f1_score = evaluator(model, ppi_gs['val_index'], prot_embed, ppi_list, labels, ppi_split_dict['val_index'], param['batch_size'])
            test_f1_score = evaluator(model, ppi_gs['test_index'], prot_embed, ppi_list, labels, ppi_split_dict['test_index'], param['batch_size'])

            if test_f1_score > test_best:
                test_best = test_f1_score
                state = copy.deepcopy(model.state_dict())
                es = 0
                best_epoch = epoch

            if val_f1_score >= val_best:
                val_best = val_f1_score
                test_val = test_f1_score
            else:
                es += 1

            print("Epoch: {}, Train Loss: {:.5f} | Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f} | Best Epoch: {}".format(
                    epoch, train_loss, train_f1_score, val_f1_score, test_f1_score, val_best, test_val, test_best, best_epoch))
            log_file.write(" Epoch: {}, Train Loss: {:.5f} | Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f} | Best Epoch: {}\n".format(
                    epoch, train_loss, train_f1_score, val_f1_score, test_f1_score, val_best, test_val, test_best, best_epoch))
            log_file.flush()

    torch.save(state, os.path.join(output_dir, "model_state.pth"))
    log_file.close()

    model.load_state_dict(state)
    eval_output, eval_labels = evaluator(model, ppi_gs['test_index'], prot_embed, ppi_list, labels, ppi_split_dict['test_index'], param['batch_size'], 'output')

    np.save(os.path.join(output_dir, "eval_output.npy"), eval_output.detach().cpu().numpy())
    np.save(os.path.join(output_dir, "eval_labels.npy"), eval_labels.detach().cpu().numpy())

    jsobj = json.dumps(ppi_split_dict)
    with open(os.path.join(output_dir, "ppi_split_dict.json"), 'w') as f:
        f.write(jsobj)
        f.close()
    return test_f1_score, test_val, test_best, best_epoch

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument("--dataset", type=str, default="SHS27k")
    parser.add_argument("--split_mode", type=str, default="bfs")
    parser.add_argument("--output_dim", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_num", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pretrain_batch_size", type=int, default=256)
    parser.add_argument("--pretrain_learning_rate", type=float, default=0.001)
    parser.add_argument("--pretrain_weight_decay", type=float, default=0.0005)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--pretrain_epoch", type=int, default=50)
    parser.add_argument("--max_epoch", type=int, default=100)

    parser.add_argument("--prot_seq_emb_dim", type=int, default=2560)
    parser.add_argument("--prot_struct_emb_dim", type=int, default=256)
    parser.add_argument("--prot_func_emb_dim", type=int, default=1024)
    parser.add_argument("--modality", type=str, default='seq')
    parser.add_argument("--protein_emb_dim", type=int, default=512)
    parser.add_argument("--amino_gnn_layers", type=int, default=4)  
    parser.add_argument("--protein_emb_dim_seq", type=int, default=512)
    parser.add_argument("--amino_gnn_layers_seq", type=int, default=4)  
    parser.add_argument("--protein_emb_dim_struct", type=int, default=512)
    parser.add_argument("--amino_gnn_layers_struct", type=int, default=4) 
    parser.add_argument("--protein_emb_dim_func", type=int, default=512)
    parser.add_argument("--amino_gnn_layers_func", type=int, default=4)  
    parser.add_argument("--dropout_ratio", type=float, default=0.0)
    parser.add_argument("--co_attn_dim", type=int, default=128)

    parser.add_argument("--ckpt_path", type=str, default='')
    parser.add_argument("--is_transductive", type=int, default=1)
    parser.add_argument("--ppi_hidden_dim", type=int, default=512)
    parser.add_argument("--ppi_num_layers", type=int, default=2)
    parser.add_argument("--ppi_dropout_ratio", type=float, default=0.0)

    args = parser.parse_args()
    param = args.__dict__
    timestamp = time.strftime("%Y-%m-%d %H-%M-%S") + f"-%3d" % ((time.time() - int(time.time())) * 1000)
    

    if param['modality'] == 'all':
        modals = ['seq', 'struct', 'func']
    else:
        modals = param['modality'].split('_')

    if len(modals) == 1:
        single_modality = modals[0]
        param[f'protein_emb_dim_{single_modality}'] = param['protein_emb_dim']
        param[f'amino_gnn_layers_{single_modality}'] = param['amino_gnn_layers']

    print(param)
    set_seed(param['seed'])

    test_acc, test_val, test_best, best_epoch = main(param)


    outFile = open('PerformMetrics_Metrics.csv','a+', newline='')
    writer = csv.writer(outFile, dialect='excel')
    results = [timestamp]
    for v, k in param.items():
        results.append(k)
    
    results.append(str(test_acc))
    results.append(str(test_val))
    results.append(str(test_best))
    writer.writerow(results)