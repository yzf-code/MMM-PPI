import csv
import json
import os
import pickle
from collections import defaultdict

dataset = 'SHS27k'
p_value_threshold = 1e-4
q_value_threshold = 1
keep_number = 200

print('dataset:', dataset)
print('p_value_threshold:', p_value_threshold)
print('q_value_threshold:', q_value_threshold)
print('keep_number:', keep_number)

fimo_file = "backup_files/fimo_output_{}/fimo.tsv".format(dataset)
sequence_motif_positions = defaultdict(list)
motif_sequences = defaultdict(set)
motif_id_2_index = {}
max_value = 0

with open(fimo_file, "r") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        if row.get("motif_alt_id") is None or row.get("sequence_name") is None:
            continue
        if row["motif_alt_id"] == "" or row["sequence_name"] == "":
            continue
        motif_id = row["motif_alt_id"]
        sequence_name = row["sequence_name"]
        start = int(row["start"])
        stop = int(row["stop"])
        p_value = float(row["p-value"])
        q_value = float(row["q-value"])
        if p_value > max_value:
            max_value = p_value
        if p_value < p_value_threshold and q_value < q_value_threshold:
            if motif_id not in motif_id_2_index:
                motif_id_2_index[motif_id] = len(motif_id_2_index)
            sequence_motif_positions[sequence_name].append(
                (motif_id_2_index[motif_id], start, stop, row["q-value"])
            )
            motif_sequences[motif_id].add(sequence_name)

cutout_sequence_motif_positions = {}
for protein_name in sequence_motif_positions.keys():
    raw_protein_segment_idx = sequence_motif_positions[protein_name]
    sorted_with_index = sorted(enumerate(raw_protein_segment_idx), key=lambda x: x[1][3])
    protein_segment_idx = [x[1] for x in sorted_with_index]
    cut_protein_segment_idx = protein_segment_idx[:keep_number]
    cutout_sequence_motif_positions[protein_name] = cut_protein_segment_idx

sorted_sequence_motif_positions = {}
for protein_name in cutout_sequence_motif_positions.keys():
    raw_protein_segment_idx = cutout_sequence_motif_positions[protein_name]
    sorted_with_index = sorted(enumerate(raw_protein_segment_idx), key=lambda x: x[1][1])
    protein_segment_idx = [x[1] for x in sorted_with_index]
    sorted_sequence_motif_positions[protein_name] = protein_segment_idx

motif_sequences = {motif: list(seq_names) for motif, seq_names in motif_sequences.items()}

cnt = 0
min_cnt = 100000
cnt_num = 0
max_cnt = 0
min_name = None
max_name = None
for sequence_name in sorted_sequence_motif_positions:
    l = len(sorted_sequence_motif_positions[sequence_name])
    cnt += l
    if min_cnt > l:
        min_cnt = l
        min_name = sequence_name
    if max_cnt < l:
        max_cnt = l
        max_name = sequence_name
    if l > 200:
        cnt_num += 1

print('sequence has motif')
print('total sequence:', len(sorted_sequence_motif_positions))
if len(sorted_sequence_motif_positions) > 0:
    print('mean num:', cnt / len(sorted_sequence_motif_positions))
print('min num:', min_cnt)
print('min name:', min_name)
print('max num:', max_cnt)
print('max name:', max_name)
print('cnt_200:', cnt_num)
print('==============================')

cnt = 0
min_cnt = 100000
max_cnt = 0
min_name = None
max_name = None
for motif_id in motif_sequences:
    l = len(motif_sequences[motif_id])
    cnt += l
    if min_cnt > l:
        min_cnt = l
        min_name = motif_id
    if max_cnt < l:
        max_cnt = l
        max_name = motif_id

print('motif in sequence')
if len(motif_sequences) > 0:
    print('mean num:', cnt / len(motif_sequences))
print('total motif:', len(motif_sequences))
print('min num:', min_cnt)
print('min name:', min_name)
print('max num:', max_cnt)
print('max name:', max_name)

parsed_json_path = "new_{}_rev_parsed_motifs_q_{}_k{}.json".format(dataset, q_value_threshold, keep_number)
with open(parsed_json_path, "w") as f:
    json.dump(sorted_sequence_motif_positions, f, indent=4)

with open("{}_motif_id_2_index.json".format(dataset), "w") as f:
    json.dump(motif_id_2_index, f, indent=4)

protein_segment_dict = sorted_sequence_motif_positions

motif_to_aa_map_list = []
seq_csv_path = './motif_processed_data/{}/rev_protein.{}.sequences.dictionary.csv'.format(dataset, dataset)
with open(seq_csv_path, 'r') as seq_f:
    reader = csv.reader(seq_f)
    for i, row in enumerate(reader):
        protein_name, protein_ori_seq = row
        protein_ori_seq = protein_ori_seq.strip()

        motif_to_aa_map = {}
        motif_to_aa_map[0] = {'motif_idx': -1, 'indices': list(range(0, len(protein_ori_seq)))}

        if protein_name in protein_segment_dict:
            protein_segments = protein_segment_dict[protein_name]
            assert len(protein_segments) != 0
            for idx, seg in enumerate(protein_segments):
                motif_idx, start_idx, end_idx = seg[0], seg[1], seg[2]
                indices = list(range(start_idx - 1, end_idx))
                motif_to_aa_map[idx + 1] = {'motif_idx': motif_idx, 'indices': indices}

        if len(motif_to_aa_map) == 1:
            motif_to_aa_map[1] = {'motif_idx': -1, 'indices': list(range(0, len(protein_ori_seq)))}
            print(protein_name)

        motif_to_aa_map_list.append(motif_to_aa_map)

out_dir = './motif_processed_data/{}'.format(dataset)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'protein.motif_to_aa_map_list.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(motif_to_aa_map_list, f)

print('Saved:', out_path)