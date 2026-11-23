# Fusion and training
**Fusion (model F).** The text feature (256 numbers) and audio feature (128 numbers) are joined and passed to a small network that outputs one fraud score per 10-second step. While training, one branch is occasionally blanked ("modality dropout") so the model cannot rely on just one. Late fusion (just averaging the two branch probabilities) is the ablation.
**Augmentation (F-aug).** Trained on a random mix of channels instead of clean audio only.
**Training rules.** Same splits for every model, 3 seeds, AdamW, early stopping on validation F1, every threshold chosen on validation only; the test split is predicted once after training.
