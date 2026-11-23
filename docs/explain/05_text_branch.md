# Text branch: BiLSTM
**What it does.** The transcript so far is cut into subword pieces (XLM-RoBERTa tokenizer), each piece becomes a learned vector, and a 2-layer bidirectional LSTM reads the sequence. An attention layer then picks the important moments and a final layer outputs a fraud probability.
**Why a BiLSTM and not a transformer.** The brief asks for a recurrent/convolutional backbone; it is small enough to train on a CPU in minutes and is easy to explain. A distilled multilingual transformer is the stronger-but-heavier alternative (planned as optional baseline T-rob).
