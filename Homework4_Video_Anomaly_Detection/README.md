# HW4 — Video Anomaly Detection on CUHK Avenue

This coursework evaluates three temporal prediction formulations derived from Jigsaw-VAD on the CUHK Avenue dataset. Training uses normal video only; testing includes both normal and abnormal events.

## Experimental setup

- Dataset: CUHK Avenue
- Training subset: videos 01–08
- Test subset: videos 01–05
- Spatio-temporal cube: 5 consecutive frames
- Detection-filter ratio: 0.9
- Training epochs: 30
- Metric: frame-level micro-AUROC

## Compared formulations

### V1 — Original Jigsaw-VAD

The baseline jointly predicts spatial jigsaw positions and temporal permutation labels.

### V2 — Binary temporal classification

Temporal prediction is reduced to a balanced binary task: an unpermuted sequence is normal, while any permutation is abnormal. The anomaly score is the predicted abnormal probability.

### V3 — Full permutation classification

The temporal branch predicts one of all `5! = 120` permutations. The anomaly score is one minus the probability assigned to the correct permutation.

## Reported results

| Version | Temporal objective | micro-AUROC |
| --- | --- | ---: |
| V1 | Original decoupled jigsaw objective | **0.7944** |
| V2 | Binary normal/permuted classification | 0.5162 |
| V3 | 120-class permutation classification | 0.6490 |

Within these runs, the original objective performed best. The binary formulation discarded too much temporal structure, while the 120-class formulation retained finer ordering information but introduced a substantially harder classification task.

## Reproduction status

This repository contains the experiment report only. The modified model code, preprocessing scripts, dataset, detections, and checkpoints are not tracked, so the reported values cannot be independently reproduced here. Refer to the [Jigsaw-VAD paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6451_ECCV_2022_paper.php) and [official implementation](https://github.com/gdwang08/Jigsaw-VAD) for the original method.

## Attribution

The baseline method is from Guodong Wang et al., “Video Anomaly Detection by Solving Decoupled Spatio-Temporal Jigsaw Puzzles,” ECCV 2022. This coursework's binary and 120-class temporal objectives are experiment-specific modifications; they are not presented as features of the upstream implementation.
