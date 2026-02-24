# BraTS 2024 MRI Analysis Agent

A comprehensive Python CLI tool for exploring and analysing the BraTS 2024 Glioma dataset (1,350 cases, 4 MRI modalities + segmentation per case).

---

## Project Structure

```
brats-data-analysis/
├── agent.py              — CLI dispatcher
├── requirements.txt      — nibabel, numpy, matplotlib, scipy, pandas, tqdm, rich, scikit-image
├── CLAUDE.md             — full technical documentation
├── tools/
│   ├── explore.py        — Script 1: Dataset inventory
│   ├── visualize.py      — Script 2: Single-case multi-planar visualization
│   ├── grid.py           — Script 3: Multi-case overview grid
│   ├── stats.py          — Script 4: Statistical analysis
│   ├── intensity.py      — Script 5: Intensity distribution analysis
│   ├── longitudinal.py   — Script 6: Longitudinal patient tracking
│   └── qc.py             — Script 7: Quality control
└── output/               — all generated files land here
```

---

## Setup

```bash
cd brats-data-analysis
python3.12 -m venv .venv          # python3.11 symlink is broken on this machine
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Script Breakdown & Thought Process

| Script | What it answers | Key outputs |
|---|---|---|
| `tools/explore.py` | "What cases do I have and what's in each?" | `dataset_summary.csv` — per-case shape, spacing, all label voxel counts & mm³ volumes |
| `tools/visualize.py` | "What does one case actually look like?" | 4×3 PNG (modality × plane) with RGBA seg overlay centred on tumour |
| `tools/grid.py` | "Quick sanity check across many cases" | N-case axial slice grid with overlay |
| `tools/stats.py` | "What are the dataset-wide statistics?" | 6-panel figure: volume histograms, class balance, scan distribution, WT boxplots, scatter, violins |
| `tools/intensity.py` | "How does MRI signal differ by sequence and region?" | 4-panel figure + CSV: histograms, bar chart, heatmap (modality × region), violins — mean, std, IQR, skewness, kurtosis |
| `tools/longitudinal.py` | "How does tumour volume change over time?" | Spaghetti plot of volume trajectories, ΔWT histogram, `longitudinal_summary.csv` |
| `tools/qc.py` | "Are there bad/anomalous cases?" | `qc_report.csv` + summary plot flagging: missing files, wrong shape, anisotropic spacing, empty masks, near-zero variance, intensity outliers |

---

## BraTS 2024 Dataset Notes

- **Case format:** `BraTS-GLI-XXXXX-YYY` — patient `XXXXX`, scan session `YYY` (100 = first, 101 = second, …)
- **Modalities:** `t1n` (T1 native), `t1c` (T1 contrast), `t2f` (T2 FLAIR), `t2w` (T2 weighted)
- **Segmentation labels (BraTS 2024):**
  - `0` = Background
  - `1` = NCR — Necrotic Core (red in overlays)
  - `2` = SNFH — Surrounding Non-enhancing FLAIR Hyperintensity / edema (green)
  - `3` = ET — Enhancing Tumour (blue)
  > **Note:** BraTS 2024 uses label **3** for ET. BraTS 2021 used label **4**.
- **Tumour regions:** ET = label 3 · TC = labels 1+3 · WT = labels 1+2+3

---

## How to Run

```bash
cd brats-data-analysis
source .venv/bin/activate

DATA=~/workspace/brats/training_data1_v2

# 1. Explore full dataset — creates output/dataset_summary.csv
python agent.py --data_dir $DATA --tool explore

# 2. Statistical overview — reads CSV, produces 6-panel figure
python agent.py --data_dir $DATA --tool stats

# 3. Visualise a single case (all 4 modalities, 3 anatomical planes)
python agent.py --data_dir $DATA --tool visualize --case BraTS-GLI-00005-100

# 4. Grid of 12 random cases (T1 contrast axial slices)
python agent.py --data_dir $DATA --tool grid --n 12 --modality t1c

# 5. Intensity analysis on 50 sampled cases
python agent.py --data_dir $DATA --tool intensity --sample 50

# 6. Longitudinal patient tracking
python agent.py --data_dir $DATA --tool longitudinal

# 7. Quality-control sweep
python agent.py --data_dir $DATA --tool qc
```

> Run `explore` first — `stats` will auto-trigger it if `dataset_summary.csv` is missing.

---

## Output Files

| File | Produced by |
|---|---|
| `output/dataset_summary.csv` | `explore` |
| `output/stats_overview.png` | `stats` |
| `output/stats_summary.txt` | `stats` |
| `output/{case_id}_visualization.png` | `visualize` |
| `output/grid_{N}cases_{modality}.png` | `grid` |
| `output/intensity_analysis.png` | `intensity` |
| `output/intensity_stats.csv` | `intensity` |
| `output/longitudinal_analysis.png` | `longitudinal` |
| `output/longitudinal_summary.csv` | `longitudinal` |
| `output/qc_report.csv` | `qc` |
| `output/qc_summary.png` | `qc` |

---

## CLI Reference

```
python agent.py --help

Arguments:
  --data_dir     PATH         Path to training_data1_v2 (required)
  --tool         TOOL         explore | visualize | grid | stats | intensity | longitudinal | qc
  --case         CASE_ID      Case ID for --tool visualize
  --n            INT          Cases for --tool grid (default: 6)
  --modality     MODALITY     t1n | t1c | t2f | t2w (default: t1c)
  --sample       INT          Cases to sample for --tool intensity (default: 50)
  --output_dir   PATH         Output directory (default: ./output)
```
