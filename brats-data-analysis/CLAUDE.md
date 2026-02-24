# BraTS 2024 MRI Analysis Agent

A comprehensive Python CLI tool for exploring and analysing the BraTS 2024 Glioma dataset.

---

## Dataset Format — BraTS 2024 GLI

### Directory Structure
```
training_data1_v2/
├── BraTS-GLI-00005-100/
│   ├── BraTS-GLI-00005-100-t1n.nii.gz    T1 native
│   ├── BraTS-GLI-00005-100-t1c.nii.gz    T1 with gadolinium contrast
│   ├── BraTS-GLI-00005-100-t2f.nii.gz    T2 FLAIR
│   ├── BraTS-GLI-00005-100-t2w.nii.gz    T2 weighted
│   └── BraTS-GLI-00005-100-seg.nii.gz    Segmentation mask
├── BraTS-GLI-00005-101/                   (second scan of same patient)
...
```

### Case ID Naming
`BraTS-GLI-XXXXX-YYY`
- `XXXXX` = 5-digit patient identifier
- `YYY`   = scan session index: `100` = first scan, `101` = second, `102` = third, …

### Segmentation Labels (BraTS 2024)
| Label | Region | Abbreviation | Colour (in visualisations) |
|-------|--------|--------------|---------------------------|
| 0     | Background | — | — |
| 1     | Necrotic Core | NCR | Red |
| 2     | Surrounding Non-enhancing FLAIR Hyperintensity | SNFH | Green |
| 3     | Enhancing Tumour | ET | Blue |

> **Important:** BraTS 2024 uses label **3** for ET.  BraTS 2021 used label **4**.

### Tumour Regions
| Region | Labels | Description |
|--------|--------|-------------|
| ET | 3 | Enhancing tumour core |
| TC | 1 + 3 | Tumour core (NCR + ET) |
| WT | 1 + 2 + 3 | Whole tumour |

---

## Project Structure

```
brats-data-analysis/
├── agent.py              CLI entry point
├── requirements.txt
├── CLAUDE.md             This file
├── tools/
│   ├── __init__.py
│   ├── explore.py        Tool 1 — dataset inventory
│   ├── visualize.py      Tool 2 — single-case multi-planar viz
│   ├── grid.py           Tool 3 — multi-case overview grid
│   ├── stats.py          Tool 4 — statistical analysis
│   ├── intensity.py      Tool 5 — intensity distribution analysis
│   ├── longitudinal.py   Tool 6 — longitudinal patient tracking
│   └── qc.py             Tool 7 — quality control
└── output/               All generated files land here
```

---

## Tool Reference

### Tool 1 — `explore`
Scans all cases, verifies file completeness, reads each segmentation to
extract shape, voxel spacing, and per-label voxel counts / volumes.

**Outputs:**
- `output/dataset_summary.csv` — per-case metadata table
- Rich terminal table: totals, shapes, spacings, % with ET, % with SNFH

**CSV columns:** `case_id, patient_id, scan_idx, complete, shape,
spacing_x/y/z, n_background, n_ncr, n_snfh, n_et,
vol_ncr_mm3, vol_snfh_mm3, vol_et_mm3, vol_tc_mm3, vol_wt_mm3, has_tumor`

---

### Tool 2 — `visualize`
Loads all 4 MRI modalities + segmentation for a single case and produces
a 4 × 3 figure (rows = modalities, columns = Axial / Coronal / Sagittal).
Each panel shows the grayscale MRI with a semi-transparent colour overlay
centred on the tumour.

**Outputs:** `output/{case_id}_visualization.png`

---

### Tool 3 — `grid`
Randomly samples N cases and renders the best axial slice (at tumour
centre) for each, with segmentation overlay.  3-column layout.

**Outputs:** `output/grid_{N}cases_{modality}.png`

---

### Tool 4 — `stats`
Loads `dataset_summary.csv` (runs `explore` first if missing) and produces
a 6-panel statistical overview:
1. Tumour volume histograms (ET, TC, WT)
2. Class-balance bar chart (mean % voxels per label)
3. Scan-index distribution (patients with 1 / 2 / 3+ scans)
4. WT volume boxplots grouped by scan index
5. WT vs ET volume scatter with regression
6. Violin plots — volume per tumour region

**Outputs:** `output/stats_overview.png`, `output/stats_summary.txt`

---

### Tool 5 — `intensity`
Samples N cases (default 50) and computes per-modality, per-region
intensity statistics (mean, std, median, IQR, percentiles, skewness,
kurtosis).  Produces 4-panel figure:
1. Whole-brain intensity histograms (all modalities overlaid)
2. Grouped bar chart — median intensity by modality × region
3. Heatmap — mean intensity, modality × region
4. Violin plot — ET-region intensity per modality

**Outputs:** `output/intensity_analysis.png`, `output/intensity_stats.csv`

---

### Tool 6 — `longitudinal`
Groups cases by patient ID, finds patients with ≥2 scans, computes WT
volume per time point, and plots:
1. Bar chart — patients by number of scans
2. Spaghetti plot — WT volume trajectory (top 30 multi-scan patients)
3. Histogram — ΔWT volume between scan 2 and scan 1

**Outputs:** `output/longitudinal_analysis.png`, `output/longitudinal_summary.csv`

---

### Tool 7 — `qc`
Systematic quality-control sweep:
- Missing files
- Non-standard image shape
- Anisotropic / extreme voxel spacing
- Empty segmentation masks
- Near-zero-variance modalities
- Intensity outliers (max >> 99th percentile)

**Outputs:** `output/qc_report.csv`, `output/qc_summary.png`

---

## Setup

```bash
# Create virtual environment (Python 3.12)
cd brats-data-analysis
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Example Commands

```bash
DATA=~/workspace/brats/training_data1_v2

# 1. Explore the full dataset (creates dataset_summary.csv)
python agent.py --data_dir $DATA --tool explore

# 2. Visualise one case (all 4 modalities, 3 planes)
python agent.py --data_dir $DATA --tool visualize --case BraTS-GLI-00005-100

# 3. Grid of 12 random cases using T1 contrast
python agent.py --data_dir $DATA --tool grid --n 12 --modality t1c

# 4. Statistical analysis (needs explore to have run first, or runs it)
python agent.py --data_dir $DATA --tool stats

# 5. Intensity analysis on 100 sampled cases
python agent.py --data_dir $DATA --tool intensity --sample 100

# 6. Longitudinal patient tracking
python agent.py --data_dir $DATA --tool longitudinal

# 7. Quality-control sweep
python agent.py --data_dir $DATA --tool qc

# Custom output directory
python agent.py --data_dir $DATA --tool grid --n 6 --output_dir ~/Desktop/brats_output
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| nibabel | Load / inspect NIfTI (.nii.gz) files |
| numpy | Array operations, statistics |
| matplotlib | All visualisations (headless via Agg backend) |
| scipy | Centre-of-mass for tumour centring; skewness/kurtosis |
| pandas | DataFrame construction, CSV I/O |
| tqdm | Progress bars |
| rich | Styled terminal output (tables, panels) |
| scikit-image | (available for morphological ops if needed) |
