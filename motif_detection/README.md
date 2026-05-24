# Protein Motif Detection with MEME / FIMO

This module detects sequence motifs in protein sequences using [FIMO](https://meme-suite.org/meme/doc/fimo.html) from the MEME Suite, and post-processes the results into a motif-to-amino-acid mapping (`protein.motif_to_aa_map_list.pkl`) that can be consumed by the downstream PPI pipeline.

---

## 1. Environment

- Python ≥ 3.8
- Standard library packages: `csv`, `json`, `os`, `pickle`, `collections`
- [MEME Suite 5.5.7](https://meme-suite.org/meme/doc/download.html) — the `fimo` binary must be on your `PATH`.

### Install MEME (Linux / macOS)

```bash
tar -zxvf meme-5.5.7.tar.gz
cd meme-5.5.7
./configure --prefix=$HOME/meme \
            --enable-build-libxml2 \
            --enable-build-libxslt
make
make test
make install

export PATH=$HOME/meme/bin:$HOME/meme/libexec/meme-5.5.7:$PATH
```

Verify:

```bash
fimo --version
```

---

## 2. Directory Layout

```
MEME/
├── backup_files/
│   └── fimo_output_SHS27k/
│       └── fimo.tsv                 # FIMO output used by postprocess.py
├── fimo_output/                     # raw FIMO output for SHS27k
├── fimo_output_SHS148k/
├── fimo_output_STRING/
├── meme-5.5.7/
├── motif_databases/
│   ├── PROTEIN/                     # used here (prosite2021_04.meme)
│   ├── ARABD/        CIS-BP_1.02/   CIS-BP_2.00/   CISBP-RNA/
│   ├── ECOLI/        EUKARYOTE/     FLY/           HOCOMOCO/
│   ├── HUMAN/        JASPAR/        MALARIA/       METHYLCYTOSINE/
│   ├── MIRBASE/      MOUSE/         PROKARYOTE/    RNA/
│   ├── TFBSshape/    WORM/          YEAST/
├── motif_processed_data/
│   └── SHS27k/
│       ├── protein.SHS27k.sequences.dictionary.csv
│       └── protein.motif_to_aa_map_list.pkl   # final output
├── protein_SHS27k_sequences.fasta
└── postprocess.py
```

---

## 3. Inputs

1. **Protein FASTA file**, e.g. `protein_SHS27k_sequences.fasta`.
2. **Motif database**: `motif_databases/PROTEIN/prosite2021_04.meme` (PROSITE motifs).
3. **Sequence dictionary CSV** at  
   `./motif_processed_data/{dataset}/protein.{dataset}.sequences.dictionary.csv`  
   Two columns, no header:
   ```
   protein_name,protein_sequence
   ```

---

## 4. Step 1 — Run FIMO

For dataset `SHS27k`:

```bash
fimo --oc fimo_output \
     motif_databases/PROTEIN/prosite2021_04.meme \
     protein_SHS27k_sequences.fasta
```

For other datasets:

```bash
# SHS148k
fimo --oc fimo_output_SHS148k \
     motif_databases/PROTEIN/prosite2021_04.meme \
     protein_SHS148k_sequences.fasta

# STRING
fimo --oc fimo_output_STRING \
     motif_databases/PROTEIN/prosite2021_04.meme \
     protein_STRING_sequences.fasta
```

FIMO produces (among other files) a `fimo.tsv`, with one motif match per row including the columns:
`motif_alt_id`, `sequence_name`, `start`, `stop`, `p-value`, `q-value`.

Move (or copy) the produced `fimo.tsv` into `backup_files/` so the post-processing script can locate it:

```bash
mkdir -p backup_files/fimo_output_SHS27k
cp fimo_output/fimo.tsv backup_files/fimo_output_SHS27k/fimo.tsv
```

---

## 5. Step 2 — Post-processing

Run:

```bash
python postprocess.py
```

### Configurable parameters (top of `postprocess.py`)

| Parameter           | Default | Description                                                                 |
|---------------------|---------|-----------------------------------------------------------------------------|
| `dataset`           | `SHS27k`| Dataset name; controls input/output paths.                                  |
| `p_value_threshold` | `1e-4`  | Matches with `p-value` ≥ threshold are discarded.                           |
| `q_value_threshold` | `1`     | Matches with `q-value` ≥ threshold are discarded.                           |
| `keep_number`       | `200`   | Keep at most this many top (lowest q-value) motif hits per protein.         |

### Outputs

- `new_{dataset}_parsed_motifs_q_{q}_k{k}.json` — parsed and filtered motif hits per sequence.
- `{dataset}_motif_id_2_index.json` — mapping from motif id (string) to integer index.
- `./motif_processed_data/{dataset}/protein.motif_to_aa_map_list.pkl` — **final output**.

The final pickle file is a `list` (one entry per protein, in the order of the sequence dictionary CSV) of `dict`s of the form:

```python
{
    0: {'motif_idx': -1, 'indices': [0, 1, ..., L-1]},      # whole-protein node
    1: {'motif_idx':  k, 'indices': [start-1, ..., end-1]}, # motif #1
    2: {'motif_idx':  m, 'indices': [...]},                 # motif #2
    ...
}
```

- `motif_idx == -1` denotes the whole-protein (global) entry.
- `indices` are 0-based amino-acid positions in the original sequence.

---

## 6. Reproduction Checklist

1. Install **MEME Suite 5.5.7** and make sure `fimo` is on your `PATH`.
2. Place `motif_databases/` at the project root (next to `meme-5.5.7/`).
3. Put your protein FASTA file (e.g. `protein_SHS27k_sequences.fasta`) at the project root.
4. Run FIMO (Section 4).
5. Copy the resulting `fimo.tsv` to `backup_files/fimo_output_{dataset}/fimo.tsv`.
6. Prepare the sequence dictionary CSV at  
   `motif_processed_data/{dataset}/protein.{dataset}.sequences.dictionary.csv`.
7. Run `python postprocess.py`.
8. Use the resulting `motif_processed_data/{dataset}/protein.motif_to_aa_map_list.pkl` in the downstream PPI pipeline.

---

## 7. Switching Datasets

To process `SHS148k` or `STRING`, simply change the `dataset` variable at the top of `postprocess.py`:

```python
dataset = 'SHS148k'   # or 'STRING'
```

and ensure that:

- `backup_files/fimo_output_{dataset}/fimo.tsv` exists, and
- `motif_processed_data/{dataset}/protein.{dataset}.sequences.dictionary.csv` exists.

Then re-run:

```bash
python postprocess.py
```