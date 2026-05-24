import copy
import warnings
import torch

from utils import *
from Classifier_model import *
from dataloader import *

warnings.filterwarnings("ignore", category=Warning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pretrain_trainer(model, train_dataloader, optimizer, loss_fn, epoch):
    f1_sum = 0.0
    loss_sum = 0.0
    model.train()
    
    for b_idx, (ppi_idx_list, batched_amino_graph1, batched_amino_graph2, batched_motif_to_aa_mapping_matrix1_list, batched_motif_to_aa_mapping_matrix2_list, labels) in tqdm(enumerate(train_dataloader)):
        batched_amino_graph1 = batched_amino_graph1.to(device)
        batched_amino_graph2 = batched_amino_graph2.to(device)

        output = model(batched_amino_graph1, batched_amino_graph2, batched_motif_to_aa_mapping_matrix1_list, batched_motif_to_aa_mapping_matrix2_list)
        labels = labels.to(device)
        loss = loss_fn(output, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        f1_score = evaluat_metrics(output.detach().cpu(), labels.detach().cpu())
        f1_sum += f1_score
    return loss_sum / (b_idx+1), f1_sum / (b_idx+1)

def pretrain_evaluator(model, eval_dataloader, mode='metric'):
    eval_output_list = []
    eval_labels_list = []
    model.eval()

    with torch.no_grad():
        for b_idx, (ppi_idx_list, batched_amino_graph1, batched_amino_graph2, batched_motif_to_aa_mapping_matrix1_list, batched_motif_to_aa_mapping_matrix2_list, labels) in tqdm(enumerate(eval_dataloader)):
            batched_amino_graph1 = batched_amino_graph1.to(device)
            batched_amino_graph2 = batched_amino_graph2.to(device)

            output = model(batched_amino_graph1, batched_amino_graph2, batched_motif_to_aa_mapping_matrix1_list, batched_motif_to_aa_mapping_matrix2_list)
            labels = labels.to(device)
            
            eval_output_list.append(output.detach().cpu())
            eval_labels_list.append(labels.detach().cpu())

        f1_score = evaluat_metrics(torch.cat(eval_output_list, dim=0), torch.cat(eval_labels_list, dim=0))

    if mode == 'metric':
        return f1_score
    elif mode == 'output':
        return torch.cat(eval_output_list, dim=0), torch.cat(eval_labels_list, dim=0)

def pretrain_Pair_wise_Encoder(param, timestamp, train_dataloader, val_dataloader, test_dataloader):
    output_dir = "results/{}/{}/Pair_wise_Encoder/".format(param['dataset'], timestamp)
    check_writable(output_dir, overwrite=False)
    log_file = open(os.path.join(output_dir, "train_log.txt"), 'a+')

    Pair_wise_Encoder_Classifier_model = Pair_wise_Classifier_model(param).to(device)
        
    print(Pair_wise_Encoder_Classifier_model)
    total_params = sum(p.numel() for p in Pair_wise_Encoder_Classifier_model.parameters())
    print(f"Pair_wise_Encoder parameter number: {total_params}")

    optimizer = torch.optim.AdamW(Pair_wise_Encoder_Classifier_model.parameters(), lr=float(param['pretrain_learning_rate']), weight_decay=float(param['pretrain_weight_decay']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5,min_lr=float(param['pretrain_learning_rate'])*0.01, verbose=True)
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    
    es = 0
    val_best = 0
    test_val = 0
    test_best = 0
    best_epoch = 0

    for epoch in range(1, param["pretrain_epoch"] + 1):
        train_loss, train_f1_score = pretrain_trainer(Pair_wise_Encoder_Classifier_model, train_dataloader, optimizer, loss_fn, epoch)
        scheduler.step(train_loss)

        if (epoch - 1) % param['log_num'] == 0:
            val_f1_score = pretrain_evaluator(Pair_wise_Encoder_Classifier_model, val_dataloader)
            test_f1_score = pretrain_evaluator(Pair_wise_Encoder_Classifier_model, test_dataloader)

            if test_f1_score > test_best:
                test_best = test_f1_score

            if val_f1_score >= val_best:
                val_best = val_f1_score
                test_val = test_f1_score
                state = copy.deepcopy(Pair_wise_Encoder_Classifier_model.state_dict())
                es = 0
                best_epoch = epoch
            else:
                es += 1

            print("Epoch: {}, Train Loss: {:.5f} | Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f} | Best Epoch: {}".format(
                    epoch, train_loss, train_f1_score, val_f1_score, test_f1_score, val_best, test_val, test_best, best_epoch))
            log_file.write(" Epoch: {}, Train Loss: {:.5f} | Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f} | Best Epoch: {}\n".format(
                    epoch, train_loss, train_f1_score, val_f1_score, test_f1_score, val_best, test_val, test_best, best_epoch))
            log_file.flush()


            if es == 4:
                print("Early stopping!")
                break

    torch.save(state, os.path.join(output_dir, "Pair_wise_Encoder_Classifier_model.pth"))
    log_file.close()

    del Pair_wise_Encoder_Classifier_model
    torch.cuda.empty_cache()