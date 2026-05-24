### SHS27k ###
CUDA_VISIBLE_DEVICES=0 python -B main.py --dataset SHS27k --split_mode random --modality all  --seed 2222\
    --pretrain_batch_size 64 --log_num 5 --pretrain_epoch 50 \
    --pretrain_learning_rate 0.0005 --pretrain_weight_decay 0.005 \
    --protein_emb_dim_seq 512 --amino_gnn_layers_seq 2 \
    --protein_emb_dim_struct 256 --amino_gnn_layers_struct 2 \
    --protein_emb_dim_func 512 --amino_gnn_layers_func 1 \
    --dropout_ratio 0.3 --co_attn_dim 256 \
    --batch_size 64 --learning_rate 0.001 --weight_decay 0.05  --max_epoch 1000 \
    --is_transductive 1 --ppi_hidden_dim 2048 --ppi_num_layers 2 --ppi_dropout_ratio 0.4 \
    --ckpt_path 'ckpt/SHS27k/random/Pair_wise_Encoder/Pair_wise_Encoder_Classifier_model.pth'


CUDA_VISIBLE_DEVICES=0 python -B main.py --dataset SHS27k --split_mode dfs --modality all  --seed 4444\
    --pretrain_batch_size 64 --log_num 5 --pretrain_epoch 50 \
    --pretrain_learning_rate 0.0005 --pretrain_weight_decay 0.005 \
    --protein_emb_dim_seq 512 --amino_gnn_layers_seq 4 \
    --protein_emb_dim_struct 512 --amino_gnn_layers_struct 4 \
    --protein_emb_dim_func 512 --amino_gnn_layers_func 4 \
    --dropout_ratio 0.3 --co_attn_dim 256 \
    --batch_size 64 --learning_rate 0.001 --weight_decay 0.05  --max_epoch 500 \
    --is_transductive 1 --ppi_hidden_dim 1024 --ppi_num_layers 2 --ppi_dropout_ratio 0.2 \
    --ckpt_path 'ckpt/SHS27k/dfs/Pair_wise_Encoder/Pair_wise_Encoder_Classifier_model.pth'


CUDA_VISIBLE_DEVICES=0 python -B main.py --dataset SHS27k --split_mode bfs --modality all --seed 3333\
    --pretrain_batch_size 64 --log_num 5 --pretrain_epoch 50 \
    --pretrain_learning_rate 0.0005 --pretrain_weight_decay 0.005 \
    --protein_emb_dim_seq 512 --amino_gnn_layers_seq 4 \
    --protein_emb_dim_struct 512 --amino_gnn_layers_struct 4 \
    --protein_emb_dim_func 512 --amino_gnn_layers_func 4 \
    --dropout_ratio 0.5 --co_attn_dim 256  \
    --batch_size 64 --learning_rate 0.001 \
    --weight_decay 0.05  --max_epoch 500 --is_transductive 1 \
    --ppi_hidden_dim 1024 --ppi_num_layers 2 --ppi_dropout_ratio 0.2 \
    --ckpt_path 'ckpt/SHS27k/bfs/Pair_wise_Encoder/Pair_wise_Encoder_Classifier_model.pth'


### SHS148k ###
CUDA_VISIBLE_DEVICES=0 python -B main.py --dataset SHS148k --split_mode random --modality all \
    --pretrain_batch_size 128 --log_num 5 --pretrain_epoch 50 \
    --pretrain_learning_rate 0.001 --pretrain_weight_decay 0.005 \
    --protein_emb_dim_seq 256 --amino_gnn_layers_seq 2 \
    --protein_emb_dim_struct 128 --amino_gnn_layers_struct 1 \
    --protein_emb_dim_func 256 --amino_gnn_layers_func 2 \
    --dropout_ratio 0.1 --co_attn_dim 128 \
    --batch_size 512 --learning_rate 0.002 --weight_decay 0.05  --max_epoch 1000 \
    --is_transductive 1 --ppi_hidden_dim 2048 --ppi_num_layers 2 --ppi_dropout_ratio 0.4 \
    --ckpt_path 'ckpt/SHS148k/random/Pair_wise_Encoder/Pair_wise_Encoder_Classifier_model.pth'


CUDA_VISIBLE_DEVICES=0 python -B main.py --dataset SHS148k --split_mode dfs --modality all --seed 4444\
    --pretrain_batch_size 128 --log_num 5 --pretrain_epoch 50 \
    --pretrain_learning_rate 0.001 --pretrain_weight_decay 0.005 \
    --protein_emb_dim_seq 128 --amino_gnn_layers_seq 2 \
    --protein_emb_dim_struct 128 --amino_gnn_layers_struct 2 \
    --protein_emb_dim_func 128 --amino_gnn_layers_func 2 \
    --co_attn_dim 256 --dropout_ratio 0.1 \
    --batch_size 512 --learning_rate 0.002 --weight_decay 0.05  --max_epoch 1000 \
    --is_transductive 1 --ppi_hidden_dim 1024 --ppi_num_layers 2 --ppi_dropout_ratio 0.4 \
    --ckpt_path 'ckpt/SHS148k/dfs/Pair_wise_Encoder/Pair_wise_Encoder_Classifier_model.pth'


CUDA_VISIBLE_DEVICES=0 python -B main.py --dataset SHS148k --split_mode bfs --modality all \
    --pretrain_batch_size 128 --log_num 5 --pretrain_epoch 50 \
    --pretrain_learning_rate 0.0005 --pretrain_weight_decay 0.0005 \
    --protein_emb_dim_seq 256 --amino_gnn_layers_seq 2 \
    --protein_emb_dim_struct 256 --amino_gnn_layers_struct 1 \
    --protein_emb_dim_func 128 --amino_gnn_layers_func 2 \
    --co_attn_dim 128 --dropout_ratio 0.5 \
    --batch_size 512 --learning_rate 0.002 --weight_decay 0.05  --max_epoch 1000 \
    --is_transductive 1 --ppi_hidden_dim 1024 --ppi_num_layers 2 --ppi_dropout_ratio 0.4 \
    --ckpt_path 'ckpt/SHS148k/bfs/Pair_wise_Encoder/Pair_wise_Encoder_Classifier_model.pth'


### STRING ###
CUDA_VISIBLE_DEVICES=0 python -B main.py --dataset STRING --split_mode random --modality all \
    --pretrain_batch_size 200 --log_num 5 --pretrain_epoch 30 \
    --pretrain_learning_rate 0.002 --pretrain_weight_decay 0.005 \
    --protein_emb_dim_seq 256 --amino_gnn_layers_seq 2 \
    --protein_emb_dim_struct 128 --amino_gnn_layers_struct 1 \
    --protein_emb_dim_func 256 --amino_gnn_layers_func 2 \
    --dropout_ratio 0.1 --co_attn_dim 128 \
    --batch_size 1024 --learning_rate 0.003 --weight_decay 0.05  --max_epoch 1500 \
    --is_transductive 1 --ppi_hidden_dim 2048 --ppi_num_layers 2 --ppi_dropout_ratio 0.2 \
    --ckpt_path 'ckpt/STRING/random/Pair_wise_Encoder/Pair_wise_Encoder_Classifier_model.pth'


CUDA_VISIBLE_DEVICES=1 python -B main.py --dataset STRING --split_mode dfs --modality all \
    --pretrain_batch_size 48 --log_num 5 --pretrain_epoch 30 \
    --pretrain_learning_rate 0.0005 --pretrain_weight_decay 0.005 \
    --protein_emb_dim_seq 128 --amino_gnn_layers_seq 2 \
    --protein_emb_dim_struct 128 --amino_gnn_layers_struct 2 \
    --protein_emb_dim_func 128 --amino_gnn_layers_func 2 \
    --co_attn_dim 256 --dropout_ratio 0.1 \
    --batch_size 1024 --learning_rate 0.003 --weight_decay 0.05  --max_epoch 500 \
    --is_transductive 1 --ppi_hidden_dim 1024 --ppi_num_layers 2 --ppi_dropout_ratio 0.2 \
    --ckpt_path 'ckpt/STRING/dfs/Pair_wise_Encoder/Pair_wise_Encoder_Classifier_model.pth'


CUDA_VISIBLE_DEVICES=0 python -B main.py --dataset STRING --split_mode bfs --modality all \
    --pretrain_batch_size 48 --log_num 5 --pretrain_epoch 30 \
    --pretrain_learning_rate 0.0005 --pretrain_weight_decay 0.005 \
    --protein_emb_dim_seq 256 --amino_gnn_layers_seq 2 \
    --protein_emb_dim_struct 256 --amino_gnn_layers_struct 1 \
    --protein_emb_dim_func 128 --amino_gnn_layers_func 2 \
    --co_attn_dim 128 --dropout_ratio 0.5 \
    --batch_size 1024 --learning_rate 0.003 --weight_decay 0.05  --max_epoch 500 \
    --is_transductive 1 --ppi_hidden_dim 2048 --ppi_num_layers 2 --ppi_dropout_ratio 0.4 \
    --ckpt_path 'ckpt/STRING/bfs/Pair_wise_Encoder/Pair_wise_Encoder_Classifier_model.pth'