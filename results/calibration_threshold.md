# Calibration and threshold analysis (set 1 test; fitted on validation only; mean over seeds)

## Temperature scaling (ECE, lower is better)

| model | cond | T | ece_before | ece_after | n_seeds |
|---|---|---|---|---|---|
| F | clean | 2.324 | 0.106 | 0.086 | 5 |
| F | hard | 2.324 | 0.278 | 0.189 | 5 |
| F | snr5 | 2.324 | 0.161 | 0.099 | 5 |
| T-asr | clean | 2.019 | 0.099 | 0.101 | 5 |
| T-asr | hard | 2.019 | 0.226 | 0.145 | 5 |
| T-asr | snr5 | 2.019 | 0.127 | 0.091 | 5 |

## Threshold chosen on clean validation versus clean + degraded validation

| model | threshold from | cond | thr | f1 | recall | normal_flagged | n_seeds |
|---|---|---|---|---|---|---|---|
| F | clean+degraded validation | clean | 0.193 | 0.884 | 0.870 | 0.102 | 5 |
| F | clean+degraded validation | hard | 0.193 | 0.675 | 0.619 | 0.203 | 5 |
| F | clean+degraded validation | snr5 | 0.193 | 0.823 | 0.784 | 0.121 | 5 |
| F | clean-validation | clean | 0.308 | 0.888 | 0.851 | 0.069 | 5 |
| F | clean-validation | hard | 0.308 | 0.659 | 0.590 | 0.190 | 5 |
| F | clean-validation | snr5 | 0.308 | 0.813 | 0.759 | 0.108 | 5 |
| T-asr | clean+degraded validation | clean | 0.594 | 0.876 | 0.844 | 0.085 | 5 |
| T-asr | clean+degraded validation | hard | 0.594 | 0.619 | 0.552 | 0.197 | 5 |
| T-asr | clean+degraded validation | snr5 | 0.594 | 0.800 | 0.743 | 0.105 | 5 |
| T-asr | clean-validation | clean | 0.582 | 0.881 | 0.854 | 0.089 | 5 |
| T-asr | clean-validation | hard | 0.582 | 0.642 | 0.578 | 0.213 | 5 |
| T-asr | clean-validation | snr5 | 0.582 | 0.812 | 0.756 | 0.102 | 5 |