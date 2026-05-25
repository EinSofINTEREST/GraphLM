"""Tests for graphlm.neuron.backbone — NeuronGrowingDecoder."""

from __future__ import annotations

import pytest
import torch

from graphlm.neuron.backbone import NeuronConfig, NeuronGrowingDecoder


@pytest.fixture
def small_cfg():
    return NeuronConfig(
        vocab_size=32,
        hidden_dim=64,
        n_heads=2,
        ffn_dim=128,
        max_seq_len=16,
        n_layers=2,
        n_init_attn=1,
    )


def test_forward_shape(small_cfg):
    model = NeuronGrowingDecoder(small_cfg)
    x = torch.randint(0, small_cfg.vocab_size, (2, 8))
    logits = model(x)
    assert logits.shape == (2, 8, small_cfg.vocab_size)


def test_n_attn_per_block_initial(small_cfg):
    model = NeuronGrowingDecoder(small_cfg)
    assert model.n_attn_per_block == [1, 1]


def test_add_attn_increments_count(small_cfg):
    model = NeuronGrowingDecoder(small_cfg)
    model.add_attn(0)
    model.add_attn(0)
    model.add_attn(1)
    assert model.n_attn_per_block == [3, 2]


def test_add_attn_invalid_block_idx_raises(small_cfg):
    model = NeuronGrowingDecoder(small_cfg)
    with pytest.raises(IndexError, match="block_idx"):
        model.add_attn(99)


def test_add_attn_alpha_zero_default(small_cfg):
    model = NeuronGrowingDecoder(small_cfg)
    model.add_attn(0)
    # 새 attention 은 alpha=0.0 으로 init (function preservation)
    new_alpha = model.blocks[0].attn_alphas[-1].item()
    assert new_alpha == 0.0


def test_n_params_increases_after_add_attn(small_cfg):
    model = NeuronGrowingDecoder(small_cfg)
    before = model.n_params
    model.add_attn(0)
    after = model.n_params
    # 새 LN (2*hidden) + new qkv (hidden * 3*hidden) + new out (hidden * hidden) + alpha (1)
    expected_delta = (
        2 * small_cfg.hidden_dim  # LN weight + bias
        + small_cfg.hidden_dim * 3 * small_cfg.hidden_dim  # qkv
        + small_cfg.hidden_dim * small_cfg.hidden_dim  # out
        + 1  # alpha
    )
    assert after - before == expected_delta


def test_forward_after_add_attn(small_cfg):
    model = NeuronGrowingDecoder(small_cfg)
    model.add_attn(0)
    x = torch.randint(0, small_cfg.vocab_size, (2, 8))
    logits = model(x)
    assert logits.shape == (2, 8, small_cfg.vocab_size)


def test_max_seq_len_exceeded_raises(small_cfg):
    model = NeuronGrowingDecoder(small_cfg)
    x = torch.randint(0, small_cfg.vocab_size, (1, small_cfg.max_seq_len + 1))
    with pytest.raises(ValueError, match="seq_len"):
        model(x)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("n_layers", 0, "n_layers"),
        ("n_init_attn", 0, "n_init_attn"),
        ("hidden_dim", 0, "hidden_dim"),
        ("n_heads", 0, "n_heads"),
        ("vocab_size", 0, "vocab_size"),
        ("max_seq_len", 0, "max_seq_len"),
    ],
)
def test_config_validation_rejects_zero(field, value, match):
    """NeuronConfig.__post_init__ 가 잘못된 값을 명시적 ValueError 로 거부."""
    kwargs = {
        "vocab_size": 32,
        "hidden_dim": 64,
        "n_heads": 2,
        "ffn_dim": 128,
        "max_seq_len": 16,
        "n_layers": 2,
        "n_init_attn": 1,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=match):
        NeuronConfig(**kwargs)


def test_config_validation_hidden_dim_not_divisible():
    with pytest.raises(ValueError, match="divisible"):
        NeuronConfig(vocab_size=32, hidden_dim=65, n_heads=2)


# ---------- Phase 4: per-channel α ---------- #


@pytest.fixture
def per_channel_cfg():
    return NeuronConfig(
        vocab_size=32,
        hidden_dim=64,
        n_heads=2,
        ffn_dim=128,
        max_seq_len=16,
        n_layers=2,
        n_init_attn=1,
        alpha_per_channel=True,
    )


def test_per_channel_alpha_shape_at_init(per_channel_cfg):
    model = NeuronGrowingDecoder(per_channel_cfg)
    for block in model.blocks:
        assert block.attn_alphas[0].shape == (per_channel_cfg.hidden_dim,)
        assert block.ffn_alpha.shape == (per_channel_cfg.hidden_dim,)


def test_per_channel_alpha_shape_after_add(per_channel_cfg):
    model = NeuronGrowingDecoder(per_channel_cfg)
    model.add_attn(0, residual_scale=0.10)
    new_alpha = model.blocks[0].attn_alphas[-1]
    assert new_alpha.shape == (per_channel_cfg.hidden_dim,)
    assert torch.allclose(new_alpha, torch.full((per_channel_cfg.hidden_dim,), 0.10))


def test_per_channel_alpha_zero_preserves_function(per_channel_cfg):
    """α=0 vector init → forward 불변 (function preservation 채널 단위로도 보장)."""
    torch.manual_seed(0)
    model = NeuronGrowingDecoder(per_channel_cfg)
    model.eval()
    x = torch.randint(0, per_channel_cfg.vocab_size, (2, 8))
    with torch.no_grad():
        out_before = model(x)
        model.add_attn(0, residual_scale=0.0)
        out_after = model(x)
    assert torch.allclose(out_before, out_after, atol=1e-6)


def test_per_channel_alpha_equivalent_to_scalar_at_uniform_init(per_channel_cfg):
    """모든 채널이 같은 값이면 scalar α 와 forward 결과 동일 — broadcasting 수학적 등가 sanity."""
    torch.manual_seed(0)
    scalar_cfg = NeuronConfig(
        vocab_size=per_channel_cfg.vocab_size,
        hidden_dim=per_channel_cfg.hidden_dim,
        n_heads=per_channel_cfg.n_heads,
        ffn_dim=per_channel_cfg.ffn_dim,
        max_seq_len=per_channel_cfg.max_seq_len,
        n_layers=per_channel_cfg.n_layers,
        n_init_attn=per_channel_cfg.n_init_attn,
        alpha_per_channel=False,
    )
    torch.manual_seed(0)
    m_scalar = NeuronGrowingDecoder(scalar_cfg)
    torch.manual_seed(0)
    m_per = NeuronGrowingDecoder(per_channel_cfg)
    # 같은 시드로 같은 weight init — α 만 shape 다름 (둘 다 1.0 으로 init 되어 broadcasting 결과 동일)
    m_scalar.eval()
    m_per.eval()
    x = torch.randint(0, scalar_cfg.vocab_size, (2, 8))
    with torch.no_grad():
        out_scalar = m_scalar(x)
        out_per = m_per(x)
    assert torch.allclose(out_scalar, out_per, atol=1e-6)


def test_per_channel_n_params_delta(per_channel_cfg):
    model = NeuronGrowingDecoder(per_channel_cfg)
    before = model.n_params
    model.add_attn(0)
    after = model.n_params
    expected_delta = (
        2 * per_channel_cfg.hidden_dim  # LN weight + bias
        + per_channel_cfg.hidden_dim * 3 * per_channel_cfg.hidden_dim  # qkv
        + per_channel_cfg.hidden_dim * per_channel_cfg.hidden_dim  # out
        + per_channel_cfg.hidden_dim  # per-channel alpha (= hidden_dim instead of +1)
    )
    assert after - before == expected_delta


# ---------- Phase 6: positional (SinusoidalAlpha) α ---------- #


@pytest.fixture
def positional_cfg():
    return NeuronConfig(
        vocab_size=32,
        hidden_dim=64,
        n_heads=2,
        ffn_dim=128,
        max_seq_len=16,
        n_layers=2,
        n_init_attn=1,
        alpha_positional=True,
    )


def test_positional_alpha_module_shape(positional_cfg):
    """attn_alphas, ffn_alpha 가 SinusoidalAlpha 모듈, forward 출력 (T, hidden_dim)."""
    from graphlm.neuron.backbone import SinusoidalAlpha

    model = NeuronGrowingDecoder(positional_cfg)
    for block in model.blocks:
        assert isinstance(block.attn_alphas[0], SinusoidalAlpha)
        assert isinstance(block.ffn_alpha, SinusoidalAlpha)
        a = block.attn_alphas[0](8)
        assert a.shape == (8, positional_cfg.hidden_dim)


def test_positional_alpha_sweet_spot_init_equivalent_to_per_channel(positional_cfg):
    """init_amp=0, init_bias=value → forward 결과가 per-channel α=value 와 동일.

    sweet spot 등가 init 보장 — Phase 6 가 Phase 4/5 의 위치 무관 baseline 으로부터 출발.
    """
    init_v = 0.10
    # positional: bias=0.10, amplitude=0 → α(t)=0.10 ∀t
    torch.manual_seed(0)
    m_pos = NeuronGrowingDecoder(positional_cfg)
    # 모든 SinusoidalAlpha 의 bias 를 init_v 로 (기본은 1.0, sweet spot 비교용으로 변경)
    for block in m_pos.blocks:
        for alpha_mod in block.attn_alphas:
            with torch.no_grad():
                alpha_mod.bias.fill_(init_v)
                alpha_mod.amplitude.fill_(0.0)
        with torch.no_grad():
            block.ffn_alpha.bias.fill_(init_v)
            block.ffn_alpha.amplitude.fill_(0.0)

    # per-channel: alpha=0.10 모든 채널
    per_ch_cfg = NeuronConfig(
        vocab_size=positional_cfg.vocab_size,
        hidden_dim=positional_cfg.hidden_dim,
        n_heads=positional_cfg.n_heads,
        ffn_dim=positional_cfg.ffn_dim,
        max_seq_len=positional_cfg.max_seq_len,
        n_layers=positional_cfg.n_layers,
        n_init_attn=positional_cfg.n_init_attn,
        alpha_per_channel=True,
    )
    torch.manual_seed(0)
    m_pc = NeuronGrowingDecoder(per_ch_cfg)
    for block in m_pc.blocks:
        for alpha in block.attn_alphas:
            with torch.no_grad():
                alpha.fill_(init_v)
        with torch.no_grad():
            block.ffn_alpha.fill_(init_v)

    m_pos.eval()
    m_pc.eval()
    x = torch.randint(0, positional_cfg.vocab_size, (2, 8))
    with torch.no_grad():
        out_pos = m_pos(x)
        out_pc = m_pc(x)
    assert torch.allclose(out_pos, out_pc, atol=1e-5), (
        f"sweet spot 등가 깨짐: max diff {(out_pos - out_pc).abs().max().item()}"
    )


def test_positional_alpha_zero_preserves_function(positional_cfg):
    """추가 attn 에 residual_scale=0.0 → SinusoidalAlpha 의 bias=0, amplitude=0 → forward 불변."""
    torch.manual_seed(0)
    model = NeuronGrowingDecoder(positional_cfg)
    model.eval()
    x = torch.randint(0, positional_cfg.vocab_size, (2, 8))
    with torch.no_grad():
        out_before = model(x)
        model.add_attn(0, residual_scale=0.0)
        out_after = model(x)
    assert torch.allclose(out_before, out_after, atol=1e-6)


def test_positional_alpha_frequency_learnable(positional_cfg):
    """log_freq 에 gradient 가 흐름 — sinusoidal 가치 입증의 전제."""
    model = NeuronGrowingDecoder(positional_cfg)
    x = torch.randint(0, positional_cfg.vocab_size, (2, 8))
    out = model(x)
    loss = out.sum()
    loss.backward()
    for block in model.blocks:
        for alpha_mod in block.attn_alphas:
            assert alpha_mod.log_freq.grad is not None
            assert alpha_mod.amplitude.grad is not None
            assert alpha_mod.bias.grad is not None


def test_config_validation_mutually_exclusive_alpha_modes():
    """alpha_per_channel 과 alpha_positional 동시 True 는 거부."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        NeuronConfig(
            vocab_size=32,
            alpha_per_channel=True,
            alpha_positional=True,
        )


# ---------- Phase 7: amplitude smooth-start ---------- #


def test_positional_init_amp_default_zero():
    """기본 alpha_positional_init_amp=0.0 — Phase 6 호환 (amplitude=0 시작)."""
    cfg = NeuronConfig(
        vocab_size=32,
        hidden_dim=64,
        n_heads=2,
        ffn_dim=128,
        max_seq_len=16,
        n_layers=2,
        n_init_attn=1,
        alpha_positional=True,
    )
    model = NeuronGrowingDecoder(cfg)
    for block in model.blocks:
        for alpha_mod in block.attn_alphas:
            assert torch.allclose(alpha_mod.amplitude, torch.zeros_like(alpha_mod.amplitude))


def test_positional_init_amp_nonzero_applied_to_init_and_add():
    """alpha_positional_init_amp=v 시 init 시점 + add_attn 후 모두 amplitude=v 로 채워짐."""
    init_amp = 0.05
    cfg = NeuronConfig(
        vocab_size=32,
        hidden_dim=64,
        n_heads=2,
        ffn_dim=128,
        max_seq_len=16,
        n_layers=2,
        n_init_attn=1,
        alpha_positional=True,
        alpha_positional_init_amp=init_amp,
    )
    model = NeuronGrowingDecoder(cfg)
    # init 시점 — 모든 attn / ffn alpha 의 amplitude 가 init_amp
    for block in model.blocks:
        for alpha_mod in block.attn_alphas:
            assert torch.allclose(
                alpha_mod.amplitude, torch.full_like(alpha_mod.amplitude, init_amp)
            )
        assert torch.allclose(
            block.ffn_alpha.amplitude, torch.full_like(block.ffn_alpha.amplitude, init_amp)
        )
    # add_attn 후 — 신규 alpha 도 amplitude=init_amp
    model.add_attn(0, residual_scale=0.10)
    new_alpha = model.blocks[0].attn_alphas[-1]
    assert torch.allclose(new_alpha.amplitude, torch.full_like(new_alpha.amplitude, init_amp))
    # bias 는 residual_scale=0.10 으로 별도 설정
    assert torch.allclose(new_alpha.bias, torch.full_like(new_alpha.bias, 0.10))
