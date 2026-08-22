# Leave-One-Training-Seed-Out Robustness

Thresholds are selected on the other two training seeds using maximum balanced accuracy / Youden J, then applied to the held-out training seed. This is not an independent external test set.

| metric | held-out seed | threshold | balanced accuracy | precision | recall | F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| global_normalized_parameter_divergence | 1101 | 0.0445191 | 0.893 | 0.455 | 1.000 | 0.625 | 0.957 | 0.792 |
| global_normalized_parameter_divergence | 1102 | 0.0537935 | 0.882 | 0.727 | 0.889 | 0.800 | 0.963 | 0.881 |
| global_normalized_parameter_divergence | 1103 | 0.0445191 | 0.885 | 0.538 | 1.000 | 0.700 | 0.934 | 0.814 |
| clean_update_normalized_divergence | 1101 | 5.44907 | 0.829 | 0.500 | 0.800 | 0.615 | 0.921 | 0.744 |
| clean_update_normalized_divergence | 1102 | 5.44907 | 0.847 | 0.778 | 0.778 | 0.778 | 0.917 | 0.841 |
| clean_update_normalized_divergence | 1103 | 1.57553 | 0.723 | 0.417 | 0.714 | 0.526 | 0.835 | 0.706 |
| max_layer_relative_displacement | 1101 | 0.504259 | 0.893 | 0.455 | 1.000 | 0.625 | 0.957 | 0.792 |
| max_layer_relative_displacement | 1102 | 0.504259 | 0.938 | 0.750 | 1.000 | 0.857 | 0.954 | 0.860 |
| max_layer_relative_displacement | 1103 | 0.561446 | 0.852 | 0.600 | 0.857 | 0.706 | 0.951 | 0.834 |
| function_space_kl | 1101 | 2.54909 | 0.929 | 0.556 | 1.000 | 0.714 | 0.979 | 0.872 |
| function_space_kl | 1102 | 4.81553 | 0.903 | 0.800 | 0.889 | 0.842 | 0.972 | 0.899 |
| function_space_kl | 1103 | 2.54909 | 0.904 | 0.583 | 1.000 | 0.737 | 0.967 | 0.877 |
| function_space_rms_logit | 1101 | 3.47691 | 0.911 | 0.500 | 1.000 | 0.667 | 0.979 | 0.872 |
| function_space_rms_logit | 1102 | 6.7312 | 0.833 | 1.000 | 0.667 | 0.800 | 0.972 | 0.899 |
| function_space_rms_logit | 1103 | 3.47691 | 0.904 | 0.583 | 1.000 | 0.737 | 0.989 | 0.926 |
