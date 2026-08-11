# Experiment index — exp20 onwards (the nnU-Net replica series)

Every experiment here runs the **same** loop (`src/nnunet_replica/`, see
[../NNUNET_REPLICA.md](../NNUNET_REPLICA.md)) and changes exactly **one** thing against
exp20, so the series reads as controlled ablations rather than differently-built pipelines.

| exp | file | one-line summary |
|---|---|---|
| **20** | `exp20_replica_resenc_m_11g_5fold.yaml` | The replica baseline: the 11 GB ResEnc-M plan run verbatim through our own loop — its job is to reproduce the native nnU-Net numbers. |
| **21** | `exp21_replica_5ch_subtraction_5fold.yaml` | exp20 plus a 5th input channel, the T1Gd−T1 enhancement-subtraction map. |
| **22** | `exp22_replica_dicefocal_5fold.yaml` | exp20 with Dice+**Focal** instead of Dice+CE, to push gradient onto the rare classes (NETC, ET). |
| **23** | `exp23_replica_5ch_dicefocal_5fold.yaml` | exp20 with **both** the 5th channel and the focal loss — completes the 2×2 so 21 and 22 can be read as additive or not. |
| **24** | `exp24_replica_plain_unet_11g_5fold.yaml` | exp20 with the **plain U-Net** plan instead of ResEnc — same patch and batch, 31.2 M params vs 102.4 M. |
| **25** | `exp25_replica_segresnet_ds_5fold.yaml` | exp20 with **SegResNetDS** as the network, everything else in the recipe held fixed. |
| **26** | `exp26_replica_short_budget_250ep_5fold.yaml` | exp20 trained for **250 epochs instead of 1000**, to measure what a 4× shorter schedule costs. |

All seven share: the ResEnc-M 11 GB plan's patch [128, 192, 128] / batch 2 (exp24 uses the
plain plan, which independently chose the same), 1000×250-iteration epochs (except exp26),
SGD 1e-2 / 0.99 nesterov with poly decay, patient-level 5-fold at `split_seed: 42`, and the
seed-42 10 % test patients held out of every fold.

Caches: exp21 and exp23 need a **separate 5-channel** preprocessed cache (and `*-sub.nii.gz`
precomputed); exp20, 22, 24, 25 and 26 share the 4-channel one.

Reporting: compare exp26 only against other 250-epoch runs, and compare exp25 on Dice
rather than `val_loss` (its eval-mode network returns one head, so its loss is not on the
same scale as the others).

`superseded/` holds the pre-replica exp20–exp25 configs. They describe runs already in
`runs/`; do not start new work from them. exp01–exp19 are unchanged and documented in the
main [../README.md](../README.md).
