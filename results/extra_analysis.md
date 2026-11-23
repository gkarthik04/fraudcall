# Extra analyses

## H2: F (fusion) minus T-asr (text only), set 1 test, seeds [0, 1, 2, 3, 4]

| condition | metric | n_seeds | F | T-asr | diff | ci95_lo | ci95_hi | p_paired_t |
|---|---|---|---|---|---|---|---|---|
| clean | F1 | 5 | 0.888 | 0.881 | 0.007 | -0.031 | 0.045 | 0.627 |
| clean | AUROC | 5 | 0.957 | 0.961 | -0.004 | -0.013 | 0.004 | 0.237 |
| snr10 | F1 | 5 | 0.855 | 0.852 | 0.002 | -0.018 | 0.023 | 0.773 |
| snr10 | AUROC | 5 | 0.931 | 0.932 | -0.002 | -0.022 | 0.019 | 0.837 |
| snr5 | F1 | 5 | 0.813 | 0.812 | 0.001 | -0.016 | 0.018 | 0.873 |
| snr5 | AUROC | 5 | 0.910 | 0.908 | 0.002 | -0.028 | 0.031 | 0.892 |
| loss10 | F1 | 5 | 0.888 | 0.877 | 0.011 | -0.036 | 0.058 | 0.553 |
| loss10 | AUROC | 5 | 0.958 | 0.962 | -0.003 | -0.022 | 0.015 | 0.651 |
| opus8 | F1 | 5 | 0.880 | 0.895 | -0.016 | -0.037 | 0.005 | 0.108 |
| opus8 | AUROC | 5 | 0.959 | 0.964 | -0.005 | -0.018 | 0.009 | 0.399 |
| hard | F1 | 5 | 0.659 | 0.642 | 0.017 | -0.048 | 0.083 | 0.505 |
| hard | AUROC | 5 | 0.779 | 0.778 | 0.001 | -0.021 | 0.024 | 0.882 |
| mean of 13 degraded | F1 | 5 | 0.856 | 0.859 | -0.003 | -0.022 | 0.016 | 0.697 |
| mean of 13 degraded | AUROC | 5 | 0.939 | 0.942 | -0.002 | -0.017 | 0.012 | 0.671 |

Paired-t CI is over seeds (n = shared seeds, so wide). Session-cluster bootstrap of the seed-averaged AUROC difference:

| condition | metric | diff | boot_ci95_lo | boot_ci95_hi |
|---|---|---|---|---|
| clean | AUROC (seed-averaged scores) | -0.0023 | -0.0158 | 0.0092 |
| snr5 | AUROC (seed-averaged scores) | 0.0000 | -0.0118 | 0.0121 |
| hard | AUROC (seed-averaged scores) | -0.0081 | -0.0358 | 0.0164 |

## H3: T-asr-aug minus T-asr (augmentation, text only), set 1 test, seeds [0, 1, 2, 3, 4]

| condition | metric | n_seeds | T-asr-aug | T-asr | diff | ci95_lo | ci95_hi | p_paired_t |
|---|---|---|---|---|---|---|---|---|
| clean | F1 | 5 | 0.882 | 0.881 | 0.002 | -0.016 | 0.020 | 0.778 |
| clean | AUROC | 5 | 0.956 | 0.961 | -0.005 | -0.021 | 0.011 | 0.462 |
| snr10 | F1 | 5 | 0.859 | 0.852 | 0.007 | -0.035 | 0.048 | 0.682 |
| snr10 | AUROC | 5 | 0.937 | 0.932 | 0.005 | -0.020 | 0.029 | 0.624 |
| snr5 | F1 | 5 | 0.828 | 0.812 | 0.016 | -0.061 | 0.094 | 0.585 |
| snr5 | AUROC | 5 | 0.914 | 0.908 | 0.006 | -0.032 | 0.044 | 0.692 |
| loss10 | F1 | 5 | 0.870 | 0.877 | -0.007 | -0.048 | 0.034 | 0.655 |
| loss10 | AUROC | 5 | 0.953 | 0.962 | -0.008 | -0.021 | 0.005 | 0.148 |
| opus8 | F1 | 5 | 0.883 | 0.895 | -0.012 | -0.046 | 0.022 | 0.379 |
| opus8 | AUROC | 5 | 0.957 | 0.964 | -0.006 | -0.022 | 0.009 | 0.309 |
| hard | F1 | 5 | 0.665 | 0.642 | 0.023 | -0.020 | 0.066 | 0.211 |
| hard | AUROC | 5 | 0.809 | 0.778 | 0.031 | -0.015 | 0.076 | 0.137 |
| mean of 13 degraded | F1 | 5 | 0.858 | 0.859 | -0.001 | -0.024 | 0.023 | 0.950 |
| mean of 13 degraded | AUROC | 5 | 0.939 | 0.942 | -0.002 | -0.016 | 0.012 | 0.697 |

Paired-t CI is over seeds (n = shared seeds, so wide). Session-cluster bootstrap of the seed-averaged AUROC difference:

| condition | metric | diff | boot_ci95_lo | boot_ci95_hi |
|---|---|---|---|---|
| clean | AUROC (seed-averaged scores) | -0.0021 | -0.0198 | 0.0127 |
| snr5 | AUROC (seed-averaged scores) | -0.0057 | -0.0322 | 0.0135 |
| hard | AUROC (seed-averaged scores) | 0.0130 | -0.0175 | 0.0470 |

## H3: F-aug minus F (augmentation, fusion), set 1 test, seeds [0, 1, 2, 3, 4]

| condition | metric | n_seeds | F-aug | F | diff | ci95_lo | ci95_hi | p_paired_t |
|---|---|---|---|---|---|---|---|---|
| clean | F1 | 5 | 0.876 | 0.888 | -0.011 | -0.049 | 0.026 | 0.447 |
| clean | AUROC | 5 | 0.961 | 0.957 | 0.004 | -0.015 | 0.023 | 0.575 |
| snr10 | F1 | 5 | 0.853 | 0.855 | -0.001 | -0.033 | 0.031 | 0.921 |
| snr10 | AUROC | 5 | 0.941 | 0.931 | 0.010 | 0.001 | 0.020 | 0.040 |
| snr5 | F1 | 5 | 0.819 | 0.813 | 0.007 | -0.074 | 0.087 | 0.830 |
| snr5 | AUROC | 5 | 0.923 | 0.910 | 0.013 | -0.010 | 0.036 | 0.196 |
| loss10 | F1 | 5 | 0.874 | 0.888 | -0.014 | -0.046 | 0.019 | 0.311 |
| loss10 | AUROC | 5 | 0.961 | 0.958 | 0.003 | -0.008 | 0.014 | 0.509 |
| opus8 | F1 | 5 | 0.885 | 0.880 | 0.005 | -0.025 | 0.035 | 0.643 |
| opus8 | AUROC | 5 | 0.967 | 0.959 | 0.008 | -0.007 | 0.023 | 0.207 |
| hard | F1 | 5 | 0.658 | 0.659 | -0.002 | -0.148 | 0.145 | 0.978 |
| hard | AUROC | 5 | 0.805 | 0.779 | 0.025 | -0.003 | 0.053 | 0.067 |
| mean of 13 degraded | F1 | 5 | 0.857 | 0.856 | 0.001 | -0.038 | 0.040 | 0.947 |
| mean of 13 degraded | AUROC | 5 | 0.946 | 0.939 | 0.007 | -0.006 | 0.020 | 0.203 |

Paired-t CI is over seeds (n = shared seeds, so wide). Session-cluster bootstrap of the seed-averaged AUROC difference:

| condition | metric | diff | boot_ci95_lo | boot_ci95_hi |
|---|---|---|---|---|
| clean | AUROC (seed-averaged scores) | 0.0000 | -0.0106 | 0.0112 |
| snr5 | AUROC (seed-averaged scores) | 0.0010 | -0.0182 | 0.0200 |
| hard | AUROC (seed-averaged scores) | 0.0075 | -0.0268 | 0.0432 |

## Alert rule: k consecutive steps above a validation threshold (<= 5 % of normal validation calls alert), mean over all seeds

flagged@t = share of fraud calls alerted by t seconds; far = share of normal test calls alerted; mean_ttd counts never-alerted fraud calls as 60 s.

| model | k | cond | far | flagged_final | f30 | f40 | f60 | mean_ttd | thr |
|---|---|---|---|---|---|---|---|---|---|
| F | 1 | clean | 0.016 | 0.438 | 0.368 | 0.400 | 0.438 | 43.979 | 0.998 |
| F | 1 | hard | 0.039 | 0.114 | 0.089 | 0.098 | 0.114 | 56.254 | 0.998 |
| F | 1 | snr5 | 0.023 | 0.276 | 0.241 | 0.260 | 0.276 | 49.968 | 0.998 |
| F | 2 | clean | 0.020 | 0.473 | 0.349 | 0.419 | 0.473 | 46.221 | 0.996 |
| F | 2 | hard | 0.059 | 0.124 | 0.063 | 0.095 | 0.124 | 56.913 | 0.996 |
| F | 2 | snr5 | 0.013 | 0.337 | 0.222 | 0.289 | 0.337 | 50.751 | 0.996 |
| F | 3 | clean | 0.016 | 0.584 | 0.267 | 0.505 | 0.584 | 46.421 | 0.987 |
| F | 3 | hard | 0.105 | 0.222 | 0.063 | 0.149 | 0.222 | 55.793 | 0.987 |
| F | 3 | snr5 | 0.016 | 0.457 | 0.168 | 0.368 | 0.457 | 50.106 | 0.987 |
| T-asr | 1 | clean | 0.016 | 0.463 | 0.410 | 0.435 | 0.463 | 41.796 | 0.996 |
| T-asr | 1 | hard | 0.079 | 0.146 | 0.124 | 0.130 | 0.146 | 54.730 | 0.996 |
| T-asr | 1 | snr5 | 0.033 | 0.356 | 0.305 | 0.333 | 0.356 | 46.413 | 0.996 |
| T-asr | 2 | clean | 0.023 | 0.543 | 0.435 | 0.495 | 0.543 | 43.155 | 0.990 |
| T-asr | 2 | hard | 0.075 | 0.181 | 0.098 | 0.137 | 0.181 | 55.657 | 0.990 |
| T-asr | 2 | snr5 | 0.020 | 0.435 | 0.308 | 0.378 | 0.435 | 47.457 | 0.990 |
| T-asr | 3 | clean | 0.020 | 0.590 | 0.298 | 0.521 | 0.590 | 45.915 | 0.971 |
| T-asr | 3 | hard | 0.118 | 0.213 | 0.073 | 0.165 | 0.213 | 55.531 | 0.971 |
| T-asr | 3 | snr5 | 0.020 | 0.492 | 0.200 | 0.397 | 0.492 | 49.361 | 0.971 |