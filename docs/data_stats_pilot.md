# Data statistics (pilot)

* conversations (TTS set 1): 150; set 2 (test only): 0; sessions: 136; scenarios: 30
* split/label counts: {'train/0': 53, 'train/1': 52, 'val/0': 11, 'val/1': 11, 'test/0': 11, 'test/1': 12}
* domains: {'bank_payment_fraud': 12, 'ecommerce_marketplace': 9, 'utility_bill_service': 22, 'job_employment': 15, 'tech_support_account': 14, 'personal_emergency_extortion': 18, 'insurance_healthcare': 18, 'travel_lottery': 18, 'government_impersonation': 10, 'loan_shark_harassment': 14}
* pace buckets: {'normal': 93, 'fast': 56, 'slow': 1}
* call duration (s): mean 53.4, median 60.0, histogram over 0-10..50-60 s bins [0, 0, 6, 13, 19, 112]
* rows per condition: {'set1/alaw': 23, 'set1/amrnb': 23, 'set1/bw8k': 23, 'set1/clean': 150, 'set1/gsm': 23, 'set1/hard': 23, 'set1/loss10': 23, 'set1/loss5': 23, 'set1/mulaw': 23, 'set1/opus12': 23, 'set1/opus8': 23, 'set1/rand0': 127, 'set1/rand1': 127, 'set1/snr10': 23, 'set1/snr20': 23, 'set1/snr5': 23}

## Word error rate of recognised vs clean transcript (whole call), by condition

| condition | mean WER | median WER | calls |
|---|---|---|---|
| set1/alaw | 0.147 | 0.126 | 23 |
| set1/amrnb | 0.148 | 0.135 | 23 |
| set1/bw8k | 0.146 | 0.118 | 23 |
| set1/clean | 0.137 | 0.118 | 150 |
| set1/gsm | 0.149 | 0.127 | 23 |
| set1/hard | 0.672 | 0.688 | 23 |
| set1/loss10 | 0.160 | 0.146 | 23 |
| set1/loss5 | 0.155 | 0.146 | 23 |
| set1/mulaw | 0.150 | 0.129 | 23 |
| set1/opus12 | 0.149 | 0.136 | 23 |
| set1/opus8 | 0.155 | 0.129 | 23 |
| set1/rand0 | 0.185 | 0.154 | 127 |
| set1/rand1 | 0.185 | 0.156 | 127 |
| set1/snr10 | 0.219 | 0.213 | 23 |
| set1/snr20 | 0.154 | 0.140 | 23 |
| set1/snr5 | 0.329 | 0.312 | 23 |

## Gate
```
{
 "clips_checked": 150,
 "silent_or_short": [],
 "clean_wer_mean": 0.13717709931110839,
 "overlap_free": true,
 "passed": true
}
```
## Cost
```
{
 "audio_gb_set1": 0.719012707,
 "audio_mb_per_call_clean": 1.7094058933333334,
 "asr_sec_per_call_mean": 7.8892,
 "build_synth_set1_pilot": {
  "n": 150,
  "elapsed_s": 75.38680958747864,
  "workers": 5
 },
 "build_degrade_set1_pilot": {
  "n_tasks": 703,
  "elapsed_s": 2142.2201116085052,
  "workers": 3
 }
}
```