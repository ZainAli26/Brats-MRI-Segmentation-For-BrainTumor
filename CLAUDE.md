Please build me a BraTS MRI Visualizer Agent from scratch. Here is everything you need to know:

Goal
Create a Python CLI agent that explores and visualizes BraTS 2024 brain tumor MRI datasets stored as .nii.gz NIfTI files on macOS (/Users/zainalizubair/workspace/brats/training_data1_v2).

Project Structure to Create
brats-data-analysis/
├── CLAUDE.md
├── agent.py
├── requirements.txt
├── tools/
│   ├── __init__.py
│   ├── explore.py
│   ├── visualize.py
│   └── grid.py
└── output/        (create this empty folder)

File Specifications
requirements.txt
Include: nibabel, numpy, matplotlib, scipy, pandas, tqdm, rich
CLAUDE.md
Document all 3 tools (explore, visualize, grid), the BraTS 2021 data format (flair/t1/t1ce/t2/seg modalities, label values 0/1/2/4, tumor regions ET/TC/WT), how to activate the venv, and example run commands.
agent.py

CLI entrypoint using argparse
Arguments: --data_dir (required), --tool (choices: explore/visualize/grid), --case, --n (int, default 6), --output_dir (default ./output), --modality (choices: flair/t1/t1ce/t2, default flair)
Use rich for a styled header panel
Dispatch to the correct tool function based on --tool
Validate that data_dir exists, exit with a clear error if not
Expand ~ in all paths using Path.expanduser()

tools/explore.py
Function: explore_dataset(data_dir, output_dir)

Scan all case subdirectories
Check that all 5 modalities exist per case (flair, t1, t1ce, t2, seg)
For complete cases: read shape, voxel spacing, count voxels with ET label (4) and ED label (2)
Build a pandas DataFrame, save to output/dataset_summary.csv
Print a rich Table with: total cases, complete cases, missing cases, most common shape, voxel spacing, number of cases with ET, number with edema

tools/visualize.py
Function: visualize_case(data_dir, case_id, output_dir, modality)

Load all 4 available MRI modalities for the case
Load segmentation mask
Find tumor center of mass for slice selection (fall back to volume center if no tumor)
Plot a grid of (num_modalities rows) × 3 columns (Axial, Coronal, Sagittal views)
Each slice: grayscale MRI with semi-transparent color overlay (NCR=red, Edema=green, ET=blue, alpha=0.5)
Use dark background (#1a1a2e), white titles, legend at bottom
Use matplotlib.use("Agg") so it works headless on macOS
Normalize each slice using 1st–99th percentile of non-zero voxels
Save to output/{case_id}_visualization.png at 150 dpi
Print the save path and an open command to view it

tools/grid.py
Function: visualize_grid(data_dir, output_dir, n, modality)

Randomly sample n cases from the dataset
For each case: load the specified modality, find the tumor center axial slice, show grayscale + segmentation overlay
Layout: 3 columns, ceil(n/3) rows
Dark background, case ID as subtitle per cell
Same color scheme as visualize.py
Save to output/grid_{n}cases_{modality}.png
Print save path and open command


Implementation Requirements

All matplotlib figures must use matplotlib.use("Agg") — no display window
All paths must support ~ expansion
Segmentation overlay must use RGBA numpy arrays, not matplotlib colormaps
BraTS label mapping: 1=NCR (red), 2=Edema (green), 4=ET (blue)
Tumor region definitions: ET=label 4, TC=labels 1+4, WT=labels 1+2+4
tqdm progress bars for any loop over cases
rich Console for all terminal output (no plain print statements)


After Creating All Files

Create a Python 3.11 virtual environment: python3.11 -m venv .venv
Activate it and install requirements: source .venv/bin/activate && pip install -r requirements.txt
Run a quick smoke test with a sample path to confirm no import errors: python agent.py --help
Show me the full directory tree of what was created