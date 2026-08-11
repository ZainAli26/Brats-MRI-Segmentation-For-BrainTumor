# Superseded exp20–exp25 configs (pre-replica)

These are the original exp20–exp25 experiments: the ResEnc-M architecture and a
nnU-Net-*flavoured* recipe run through `train_kfold.py` / `src/training/trainer.py` with
MONAI preprocessing and a full-data-pass epoch. They are kept because the runs already in
`runs/` were produced from them, and because `docker-compose` still exposes the exp20
services that reproduce those runs.

They were superseded by the `*_replica_*` configs in the parent directory, which run the
same architectures through `src/nnunet_replica/` — an actual reimplementation of nnU-Net
v2's training procedure rather than an approximation of it. See `../../NNUNET_REPLICA.md`
for what differed and why it mattered.

Do not start new work from these files.
