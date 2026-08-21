# Reviewer Experiment Completion V2

## Original H20 Campaign

- rows: 99
- tau crossings: 25
- task-nonrecovered: 43
- delayed rows: 9

## H100 Extension

- crossings after step 20: 0
- H100 task-nonrecovered rows: 26

## Multi-Future

- completed case summaries: 9

## Mechanism

- see delayed_mechanism_steps.csv and delayed_mechanism_summary.md

## Controlled Scale

- not completed unless controlled_scale/summary.json is present

## Longer Training

- not completed unless long_training/summary.json is present

## Alternative Metrics

- original_global_parameter_divergence: H100 AUROC=0.952, AUPRC=0.863
- clean_update_normalized_divergence: H100 AUROC=0.916, AUPRC=0.830
- max_layer_relative_displacement: H100 AUROC=0.952, AUPRC=0.860
- function_space_kl: H100 AUROC=0.982, AUPRC=0.948
- function_space_rms_logit: H100 AUROC=0.985, AUPRC=0.957
