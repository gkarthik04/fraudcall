# Data statistics (full)

* conversations (TTS set 1): 800; set 2 (test only): 124; sessions: 402; scenarios: 30
* split/label counts: {'train/0': 281, 'train/1': 276, 'val/0': 58, 'val/1': 61, 'test/0': 61, 'test/1': 63}
* domains: {'bank_payment_fraud': 75, 'ecommerce_marketplace': 88, 'utility_bill_service': 79, 'job_employment': 77, 'loan_shark_harassment': 76, 'tech_support_account': 87, 'personal_emergency_extortion': 83, 'insurance_healthcare': 92, 'travel_lottery': 77, 'government_impersonation': 66}
* pace buckets: {'normal': 518, 'slow': 16, 'fast': 266}
* call duration (s): mean 54.1, median 60.0, histogram over 0-10..50-60 s bins [0, 1, 20, 59, 118, 602]
* rows per condition: {'set1/alaw': 124, 'set1/amrnb': 124, 'set1/bw8k': 124, 'set1/clean': 800, 'set1/gsm': 124, 'set1/hard': 124, 'set1/loss10': 124, 'set1/loss5': 124, 'set1/mulaw': 124, 'set1/opus12': 124, 'set1/opus8': 124, 'set1/rand0': 676, 'set1/rand1': 676, 'set1/snr10': 124, 'set1/snr20': 124, 'set1/snr5': 124, 'set2/clean': 124, 'set2/gsm': 124, 'set2/hard': 124, 'set2/snr10': 124}

## Word error rate of recognised vs clean transcript (whole call), by condition

| condition | mean WER | median WER | calls |
|---|---|---|---|
| set1/alaw | 0.140 | 0.123 | 124 |
| set1/amrnb | 0.149 | 0.128 | 124 |
| set1/bw8k | 0.141 | 0.119 | 124 |
| set1/clean | 0.135 | 0.117 | 800 |
| set1/gsm | 0.151 | 0.132 | 124 |
| set1/hard | 0.663 | 0.668 | 124 |
| set1/loss10 | 0.156 | 0.137 | 124 |
| set1/loss5 | 0.148 | 0.132 | 124 |
| set1/mulaw | 0.141 | 0.124 | 124 |
| set1/opus12 | 0.143 | 0.125 | 124 |
| set1/opus8 | 0.149 | 0.127 | 124 |
| set1/rand0 | 0.182 | 0.151 | 676 |
| set1/rand1 | 0.181 | 0.152 | 676 |
| set1/snr10 | 0.225 | 0.209 | 124 |
| set1/snr20 | 0.155 | 0.138 | 124 |
| set1/snr5 | 0.337 | 0.313 | 124 |
| set2/clean | 0.192 | 0.178 | 124 |
| set2/gsm | 0.200 | 0.185 | 124 |
| set2/hard | 0.442 | 0.426 | 124 |
| set2/snr10 | 0.233 | 0.219 | 124 |

## Gate
```
{
 "clips_checked": 800,
 "silent_or_short": [],
 "clean_wer_mean": 0.1354994109973698,
 "overlap_free": true,
 "passed": true
}
```
## Cost
```
{
 "audio_gb_set1": 3.950803282,
 "audio_mb_per_call_clean": 1.7317262525000001,
 "asr_sec_per_call_mean": 16.269000000000002,
 "build_synth_set1_pilot": {
  "n": 150,
  "elapsed_s": 75.38680958747864,
  "workers": 5
 },
 "build_degrade_set1_pilot": {
  "n_tasks": 703,
  "elapsed_s": 2142.2201116085052,
  "workers": 3
 },
 "build_synth_set1_all": {
  "n": 800,
  "elapsed_s": 1302.8750476837158,
  "workers": 5
 },
 "build_degrade_set1_all": {
  "n_tasks": 3764,
  "elapsed_s": 8613.022258758545,
  "workers": 5
 }
}
```