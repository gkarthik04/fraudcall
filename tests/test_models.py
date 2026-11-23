"""Shape and behaviour tests for the branches and the fusion model."""
import torch
from src.models.branches import TextBiLSTM, AudioCNNBiGRU, Fusion


def test_text_branch_shapes():
    m = TextBiLSTM(100); ids = torch.randint(1, 100, (3, 20)); ln = torch.tensor([20, 10, 5])
    lg, f = m(ids, ln); assert lg.shape == (3,) and f.shape == (3, 256)


def test_audio_branch_shapes():
    m = AudioCNNBiGRU(); lg, f = m(torch.randn(2, 64, 1500)); assert lg.shape == (2,) and f.shape == (2, 128)


def test_fusion_modality_dropout_only_in_training():
    f = Fusion(TextBiLSTM(50), AudioCNNBiGRU(), mod_drop=0.5)
    ids = torch.randint(1, 50, (4, 8)); ln = torch.full((4,), 8); mel = torch.randn(4, 64, 400)
    f.eval(); a = f(ids, ln, mel); b = f(ids, ln, mel)
    assert torch.allclose(a, b) and a.shape == (4,)
