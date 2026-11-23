"""The two branches and the fusion model.

TextBiLSTM : subword embedding -> 2-layer bidirectional LSTM -> attention pooling -> feature (256-d).
AudioCNNBiGRU : log-mel (64 x frames) -> 3 conv blocks (conv, batch-norm, ReLU, pool) that shrink
                frequency 8x and time 4x -> bidirectional GRU over time -> attention pooling -> feature (128-d).
Fusion      : concatenate both features -> small MLP -> one fraud logit (early fusion).
Every model returns (logit, feature) so they can be trained alone or combined.
"""
import torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AttnPool(nn.Module):
    """Learned weighted average over time (weights from a tiny linear scorer), mask-aware."""
    def __init__(self, d):
        super().__init__(); self.score = nn.Linear(d, 1)

    def forward(self, h, mask=None):           # h: (B,T,D)
        s = self.score(h).squeeze(-1)
        if mask is not None:
            s = s.masked_fill(~mask, -1e9)
        a = torch.softmax(s, dim=1).unsqueeze(-1)
        return (a * h).sum(1)


class TextBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb=128, hid=128, layers=2, drop=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, num_layers=layers, bidirectional=True, batch_first=True, dropout=drop)
        self.pool = AttnPool(2 * hid); self.drop = nn.Dropout(drop)
        self.out_dim = 2 * hid; self.cls = nn.Linear(2 * hid, 1)

    def features(self, ids, lengths):
        x = self.drop(self.emb(ids))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=ids.shape[1])
        mask = torch.arange(ids.shape[1])[None, :] < lengths[:, None]
        return self.pool(h, mask)

    def forward(self, ids, lengths):
        f = self.features(ids, lengths)
        return self.cls(self.drop(f)).squeeze(-1), f


class AudioCNNBiGRU(nn.Module):
    def __init__(self, n_mels=64, ch=(8, 16, 32), gru=64, drop=0.3):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(1)
        blocks, c_in = [], 1
        for i, c in enumerate(ch):
            blocks += [nn.Conv2d(c_in, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
                       nn.MaxPool2d((2, 2) if i < 2 else (2, 1))]
            c_in = c
        self.cnn = nn.Sequential(*blocks)
        f_after = n_mels // 8
        self.proj = nn.Linear(ch[-1] * f_after, 2 * gru)
        self.gru = nn.GRU(2 * gru, gru, bidirectional=True, batch_first=True)
        self.pool = AttnPool(2 * gru); self.drop = nn.Dropout(drop)
        self.out_dim = 2 * gru; self.cls = nn.Linear(2 * gru, 1)

    def features(self, mel):                      # mel: (B, 64, T)
        x = self.cnn(self.bn0(mel.unsqueeze(1)))  # (B, C, F', T')
        B, C, F, T = x.shape
        x = self.proj(x.permute(0, 3, 1, 2).reshape(B, T, C * F))
        h, _ = self.gru(x)
        return self.pool(h)

    def forward(self, mel):
        f = self.features(mel)
        return self.cls(self.drop(f)).squeeze(-1), f


class Fusion(nn.Module):
    """Early fusion: concat(text feature, audio feature) -> MLP -> logit.
    Modality dropout (during training only) randomly blanks one branch so neither is ignored."""
    def __init__(self, text, audio, hid=128, drop=0.3, mod_drop=0.15):
        super().__init__()
        self.text, self.audio, self.mod_drop = text, audio, mod_drop
        self.mlp = nn.Sequential(nn.Linear(text.out_dim + audio.out_dim, hid), nn.ReLU(), nn.Dropout(drop), nn.Linear(hid, 1))

    def forward(self, ids, lengths, mel):
        ft, fa = self.text.features(ids, lengths), self.audio.features(mel)
        if self.training and self.mod_drop > 0:
            r = torch.rand(ft.shape[0], device=ft.device)
            ft = ft * (r >= self.mod_drop / 2).float().unsqueeze(1)           # drop text
            fa = fa * ((r < self.mod_drop / 2) | (r >= self.mod_drop)).float().unsqueeze(1)  # drop audio
        return self.mlp(torch.cat([ft, fa], 1)).squeeze(-1)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
