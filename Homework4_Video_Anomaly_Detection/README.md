# HW4 - Video Anomaly Detection on CUHK Avenue

This coursework report compares three temporal prediction formulations derived from Jigsaw-VAD on the CUHK Avenue dataset. Training uses normal video, while evaluation covers normal and abnormal events.

## Experimental setup

- Dataset: CUHK Avenue
- Training subset: videos 01-08
- Test subset: videos 01-05
- Spatio-temporal cube: 5 consecutive frames
- Detection-filter ratio: 0.9
- Training epochs: 30
- Metric: frame-level micro-AUROC

## Compared formulations

### V1 - Original Jigsaw-VAD

The baseline jointly predicts spatial jigsaw positions and temporal permutation labels.

### V2 - Binary temporal classification

Temporal prediction becomes a balanced binary task: an unpermuted sequence is normal, while a permutation is anomalous. The anomaly score is the predicted anomalous probability.

### V3 - Full permutation classification

The temporal branch predicts one of all `5! = 120` permutations. The anomaly score is one minus the probability assigned to the correct permutation.

## Recorded results

| Version | Temporal objective | micro-AUROC |
| --- | --- | ---: |
| V1 | Original decoupled jigsaw objective | **0.7944** |
| V2 | Binary normal/permuted classification | 0.5162 |
| V3 | 120-class permutation classification | 0.6490 |

The original objective performed best in these runs. The binary formulation compressed temporal structure into one decision, while the 120-class formulation retained order information at the cost of a more demanding classification problem.

## Report scope

This directory presents the experiment design, variants, and recorded measurements. A full rerun pairs this report with the CUHK Avenue data and the upstream Jigsaw-VAD training pipeline.

## Attribution

The baseline method is from Guodong Wang et al., ["Video Anomaly Detection by Solving Decoupled Spatio-Temporal Jigsaw Puzzles"](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6451_ECCV_2022_paper.php), ECCV 2022. The [official Jigsaw-VAD repository](https://github.com/gdwang08/Jigsaw-VAD) provides the upstream implementation. The binary and 120-class temporal objectives are coursework variants.
