import os
import math
from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb

attn_outputs_teacher = []
attn_outputs = []
offset_step = 0

# Optional compilation is cached by static tensor shape; the step-dependent shift stays runtime data.
_BASA_USE_TORCH_COMPILE = os.environ.get("BASA_USE_TORCH_COMPILE", "0").lower() in {"1", "true", "yes", "on"}
_BASA_COMPILED_FN_CACHE = {}


def _maybe_compile(fn, key):
    if not _BASA_USE_TORCH_COMPILE or not hasattr(torch, "compile"):
        return fn
    cached = _BASA_COMPILED_FN_CACHE.get(key)
    if cached is None:
        cached = torch.compile(fn, dynamic=False, fullgraph=False)
        _BASA_COMPILED_FN_CACHE[key] = cached
    return cached


def _sdpa(q, k, v, *, scale: float, attn_mask=None):
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )


def _make_padded_index_grid(height: int, width: int, target_h: int, target_w: int, pad_index: int, device) -> torch.Tensor:
    """Build a padded spatial-token index grid."""
    device = torch.device(device)
    grid = torch.full((target_h, target_w), pad_index, dtype=torch.long, device=device)
    real = torch.arange(height * width, dtype=torch.long, device=device).view(height, width)
    grid[:height, :width] = real
    return grid


def _partition_windows(index_grid: torch.Tensor, window_size: int) -> torch.Tensor:
    padded_h, padded_w = index_grid.shape
    win_h = padded_h // window_size
    win_w = padded_w // window_size
    return (
        index_grid.view(win_h, window_size, win_w, window_size)
        .permute(0, 2, 1, 3)
        .reshape(win_h * win_w, window_size * window_size)
        .contiguous()
    )


def _index_map_flat(index_map: torch.Tensor, B: int, H: int) -> torch.Tensor:
    # Shape: (B*H, num_windows*window_area).  expand avoids materializing the
    # batch/head copies before gather.
    return index_map.reshape(1, -1).expand(B * H, -1)


def _key_padding_bias_from_window_valid(
    win_valid: torch.Tensor,
    *,
    B: int,
    H: int,
    q_len: int,
    text_len: int,
    extra_len: int,
    dtype: torch.dtype,
    device,
):
    """Mask padded K/V slots when the spatial grid is not window-aligned."""
    num_windows, window_area = win_valid.shape
    if bool(win_valid.all().item()):
        return None

    text_valid = torch.ones((num_windows, text_len), dtype=torch.bool, device=device)
    if extra_len > 0:
        extra_valid = torch.ones((num_windows, extra_len), dtype=torch.bool, device=device)
        key_valid = torch.cat([text_valid, win_valid.to(device), extra_valid], dim=-1)
    else:
        key_valid = torch.cat([text_valid, win_valid.to(device)], dim=-1)

    # Broadcast over query positions: (B*H*num_windows, 1, K).
    key_valid = (
        key_valid.view(1, 1, num_windows, 1, text_len + window_area + extra_len)
        .expand(B, H, -1, 1, -1)
        .reshape(B * H * num_windows, 1, text_len + window_area + extra_len)
    )
    bias = torch.zeros(key_valid.shape, dtype=dtype, device=device)
    return bias.masked_fill(~key_valid, -10000.0)


class FluxAttnProcessor2_0:
    """Dense FLUX attention processor kept for compatibility/teacher paths."""

    def __init__(self, distill=False):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.distill = distill

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        proportional_attention=False,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        train_seq_len = 64 ** 2 + 512
        if proportional_attention:
            attention_scale = math.sqrt(math.log(key.size(2), train_seq_len) / head_dim)
        else:
            attention_scale = math.sqrt(1 / head_dim)

        hidden_states = _sdpa(query, key, value, scale=attention_scale)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states

        if self.distill:
            attn_outputs_teacher.append(hidden_states)
        return hidden_states


@lru_cache(maxsize=4096)
def init_reshape_basa_flex(
    height,
    width,
    text_length,
    window_size,
    device,
    total_layers=1,
    layer_index=0,
    denoising_step=None,
    seed=None,
    use_pooling_token=True,
    pool_size=4,
    pool_text_attend=False,
):
    """Build BASA metadata for interleaved shifted windows and mean-pooled global K/V memory."""
    global offset_step
    device = torch.device(device)
    height = int(height)
    width = int(width)
    text_length = int(text_length)
    window_size = int(window_size)
    total_layers = int(total_layers)
    layer_index = int(layer_index)

    if denoising_step is None:
        if seed is not None:
            # Seeded fallback when no explicit denoising-step offset is provided.
            g = torch.Generator(device=device if device.type == "cuda" else "cpu")
            g.manual_seed(seed)
            offset = torch.randint(0, window_size, (1,), generator=g, device=device).item()
        else:
            offset = offset_step % window_size
            offset_step += 1
    else:
        offset = int(denoising_step)

    pool_size = int(pool_size)
    if pool_size <= 0:
        raise ValueError(f"pool_size must be positive, got {pool_size}")

    delta = max(1, window_size // total_layers)
    seq = [(i * delta) % window_size for i in range(total_layers)]
    half = total_layers // 2
    interleaved = []
    for a, b in zip(seq[:half], seq[half:]):
        interleaved.append(a)
        interleaved.append(b)
    if total_layers % 2 == 1:
        interleaved.append(seq[half])
    shift_val = (interleaved[layer_index % total_layers] + offset) % window_size

    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    padded_h = height + pad_h
    padded_w = width + pad_w
    win_h = padded_h // window_size
    win_w = padded_w // window_size
    num_windows = win_h * win_w
    window_area = window_size * window_size
    image_len = height * width
    pad_index = image_len

    padded_coords = _make_padded_index_grid(height, width, padded_h, padded_w, pad_index, device)
    if shift_val > 0:
        padded_coords = torch.roll(padded_coords, shifts=(-shift_val, -shift_val), dims=(0, 1))
    index_map = _partition_windows(padded_coords, window_size)
    win_valid = (index_map != pad_index).contiguous()

    if use_pooling_token:
        pool_pad_h = (pool_size - height % pool_size) % pool_size
        pool_pad_w = (pool_size - width % pool_size) % pool_size
        pool_padded_h = height + pool_pad_h
        pool_padded_w = width + pool_pad_w
        pool_h = pool_padded_h // pool_size
        pool_w = pool_padded_w // pool_size
        num_pool_tokens = pool_h * pool_w
        pool_area = pool_size * pool_size

        pool_grid = _make_padded_index_grid(height, width, pool_padded_h, pool_padded_w, pad_index, device)
        pool_index_map = _partition_windows(pool_grid, pool_size)
        pool_valid = (pool_index_map != pad_index).contiguous()
    else:
        pool_index_map = torch.empty((0, 1), dtype=torch.long, device=device).contiguous()
        pool_valid = torch.empty((0, 1), dtype=torch.bool, device=device).contiguous()
        pool_h = pool_w = num_pool_tokens = 0
        pool_area = 1
        pool_pad_h = pool_pad_w = 0
        pool_padded_h = height
        pool_padded_w = width

    global meta
    meta = {
        "height": height,
        "width": width,
        "text_length": text_length,
        "window_size": window_size,
        "shift": shift_val,
        "pad_h": pad_h,
        "pad_w": pad_w,
        "padded_h": padded_h,
        "padded_w": padded_w,
        "win_h": win_h,
        "win_w": win_w,
        "num_windows": num_windows,
        "window_area": window_area,
        "image_len": image_len,
        "pad_index": pad_index,
        "index_map": index_map,
        "win_valid": win_valid,
        "has_window_padding": bool(pad_h or pad_w),
        "device": device,
        "use_pooling_token": bool(use_pooling_token),
        "pool_size": pool_size,
        "pool_text_attend": bool(pool_text_attend),
        "pool_h": pool_h,
        "pool_w": pool_w,
        "num_pool_tokens": num_pool_tokens,
        "pool_area": pool_area,
        "pool_pad_h": pool_pad_h,
        "pool_pad_w": pool_pad_w,
        "pool_padded_h": pool_padded_h,
        "pool_padded_w": pool_padded_w,
        "pool_index_map": pool_index_map,
        "pool_valid": pool_valid,
    }
    return meta


def _basa_pooling_attention_sdpa(
    query,
    key,
    value,
    index_map,
    win_valid,
    pool_index_map,
    pool_valid,
    scale: float,
    text_len: int,
    image_len: int,
    pad_index: int,
    num_windows: int,
    window_area: int,
    use_pooling_token: bool,
    pool_text_attend: bool,
    num_pool_tokens: int,
    pool_area: int,
    has_window_padding: bool,
):
    B, H, _S, D = query.shape
    device = query.device
    BH = B * H

    index_map = index_map.to(device, non_blocking=True)
    win_valid = win_valid.to(device, non_blocking=True)
    pool_index_map = pool_index_map.to(device, non_blocking=True)
    pool_valid = pool_valid.to(device, non_blocking=True)

    q_text = query[:, :, :text_len, :]
    k_text = key[:, :, :text_len, :]
    v_text = value[:, :, :text_len, :]

    q_img = query[:, :, text_len:text_len + image_len, :]
    k_img = key[:, :, text_len:text_len + image_len, :]
    v_img = value[:, :, text_len:text_len + image_len, :]

    zero = q_img.new_zeros((B, H, 1, D))
    q_img_padded = torch.cat([q_img, zero], dim=2)
    k_img_padded = torch.cat([k_img, zero.to(dtype=k_img.dtype)], dim=2)
    v_img_padded = torch.cat([v_img, zero.to(dtype=v_img.dtype)], dim=2)

    q_flat = q_img_padded.reshape(BH, image_len + 1, D)
    k_flat = k_img_padded.reshape(BH, image_len + 1, D)
    v_flat = v_img_padded.reshape(BH, image_len + 1, D)

    if use_pooling_token and num_pool_tokens > 0:
        pool_idx_flat = pool_index_map.reshape(1, -1).expand(BH, -1)
        gathered_k_pool = torch.gather(k_flat, 1, pool_idx_flat.unsqueeze(-1).expand(-1, -1, D))
        gathered_v_pool = torch.gather(v_flat, 1, pool_idx_flat.unsqueeze(-1).expand(-1, -1, D))
        gathered_k_pool = gathered_k_pool.view(B, H, num_pool_tokens, pool_area, D)
        gathered_v_pool = gathered_v_pool.view(B, H, num_pool_tokens, pool_area, D)

        pool_valid_f = pool_valid.to(dtype=k_img.dtype).view(1, 1, num_pool_tokens, pool_area, 1)
        pool_denom = pool_valid_f.sum(dim=3).clamp_min(1.0)
        k_pool = (gathered_k_pool * pool_valid_f).sum(dim=3) / pool_denom
        v_pool = (gathered_v_pool * pool_valid_f).sum(dim=3) / pool_denom
    else:
        k_pool = v_pool = None
        num_pool_tokens = 0

    # Text queries: mathematically identical to the old explicit softmax path.
    if pool_text_attend and k_pool is not None:
        k_text_context = torch.cat([k_text, k_img, k_pool], dim=2)
        v_text_context = torch.cat([v_text, v_img, v_pool], dim=2)
    else:
        k_text_context = torch.cat([k_text, k_img], dim=2)
        v_text_context = torch.cat([v_text, v_img], dim=2)
    out_text = _sdpa(q_text, k_text_context, v_text_context, scale=scale)

    im_idx_flat = _index_map_flat(index_map, B, H)
    gathered_q = torch.gather(q_flat, 1, im_idx_flat.unsqueeze(-1).expand(-1, -1, D))
    gathered_k_win = torch.gather(k_flat, 1, im_idx_flat.unsqueeze(-1).expand(-1, -1, D))
    gathered_v_win = torch.gather(v_flat, 1, im_idx_flat.unsqueeze(-1).expand(-1, -1, D))

    q_gathered = gathered_q.view(B, H, num_windows, window_area, D)
    k_gathered = gathered_k_win.view(B, H, num_windows, window_area, D)
    v_gathered = gathered_v_win.view(B, H, num_windows, window_area, D)

    k_parts = [k_text.unsqueeze(2).expand(-1, -1, num_windows, -1, -1), k_gathered]
    v_parts = [v_text.unsqueeze(2).expand(-1, -1, num_windows, -1, -1), v_gathered]
    if k_pool is not None:
        k_parts.append(k_pool.unsqueeze(2).expand(-1, -1, num_windows, -1, -1))
        v_parts.append(v_pool.unsqueeze(2).expand(-1, -1, num_windows, -1, -1))

    k_windows = torch.cat(k_parts, dim=3).reshape(B * H * num_windows, text_len + window_area + num_pool_tokens, D)
    v_windows = torch.cat(v_parts, dim=3).reshape(B * H * num_windows, text_len + window_area + num_pool_tokens, D)
    q_windows = q_gathered.reshape(B * H * num_windows, window_area, D)

    attn_bias = None
    if has_window_padding:
        attn_bias = _key_padding_bias_from_window_valid(
            win_valid,
            B=B,
            H=H,
            q_len=window_area,
            text_len=text_len,
            extra_len=num_pool_tokens,
            dtype=q_windows.dtype,
            device=device,
        )

    out_windows = _sdpa(q_windows, k_windows, v_windows, scale=scale, attn_mask=attn_bias)
    out_windows = out_windows.view(B, H, num_windows, window_area, D)

    out_flat = out_windows.reshape(B, H, num_windows * window_area, D)
    index_map_flat = index_map.reshape(-1)
    result_img = out_flat.new_zeros(B, H, image_len + 1, D)
    idx_expand = index_map_flat.view(1, 1, -1).expand(B, H, -1)
    result_img = result_img.scatter_add(2, idx_expand.unsqueeze(-1).expand(-1, -1, -1, D), out_flat)
    out_img = result_img[:, :, :image_len, :]

    return torch.cat([out_text, out_img], dim=2)


class ReshapeBasaFlexAttnProcessor:
    def __init__(self, distill=False, basa_meta=None):
        super().__init__()
        self.distill = distill
        self._compiled_attn = None
        self._compiled_signature = None
        self.basa_meta = basa_meta if basa_meta is not None else globals().get("meta", None)

    def set_basa_meta(self, basa_meta):
        self.basa_meta = basa_meta
        return self

    @staticmethod
    def _meta_signature(meta):
        # Do NOT include shift.  shift changes index_map values but not tensor
        # shapes or the SDPA computation graph; including it caused needless
        # recompilation every denoising step.
        return (
            meta["height"], meta["width"], meta["text_length"],
            meta["window_size"], meta["num_windows"], meta["window_area"],
            meta["image_len"],
            bool(meta.get("use_pooling_token", False)),
            meta.get("pool_size", None),
            meta.get("num_pool_tokens", 0),
            meta.get("pool_area", 1),
            bool(meta.get("pool_text_attend", False)),
            bool(meta.get("has_window_padding", False)),
        )

    def _ensure_compiled(self, meta):
        signature = self._meta_signature(meta)
        if self._compiled_attn is not None and self._compiled_signature == signature:
            return
        self._compiled_attn = _maybe_compile(_basa_pooling_attention_sdpa, ("basa_pooling_sdpa", signature))
        self._compiled_signature = signature

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor = None,
        image_rotary_emb: torch.Tensor = None,
        proportional_attention=False,
    ):
        batch_size = hidden_states.shape[0]

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        meta = self.basa_meta if self.basa_meta is not None else globals().get("meta", None)
        if meta is None:
            raise RuntimeError("ReshapeBasaFlexAttnProcessor.basa_meta is None; call init_reshape_basa_flex(...) or set_basa_meta(...).")

        if "pool_index_map" not in meta or meta.get("pool_index_map") is None:
            meta = dict(meta)
            meta["use_pooling_token"] = False
            meta["pool_text_attend"] = False
            meta["num_pool_tokens"] = 0
            meta["pool_area"] = 1
            meta["pool_size"] = 1
            meta["pool_index_map"] = torch.empty((0, 1), dtype=torch.long, device=query.device).contiguous()
            meta["pool_valid"] = torch.empty((0, 1), dtype=torch.bool, device=query.device).contiguous()
            self.basa_meta = meta
        if "win_valid" not in meta:
            meta = dict(meta)
            meta["win_valid"] = (meta["index_map"] != meta["pad_index"]).contiguous()
            meta["has_window_padding"] = bool(meta.get("pad_h", 0) or meta.get("pad_w", 0))
            self.basa_meta = meta
        if "pool_valid" not in meta:
            meta = dict(meta)
            meta["pool_valid"] = (meta["pool_index_map"] != meta["pad_index"]).contiguous()
            self.basa_meta = meta

        self._ensure_compiled(meta)

        train_seq_len = 64 ** 2 + 512
        if proportional_attention:
            attention_scale = math.sqrt(math.log(key.size(2), train_seq_len) / head_dim)
        else:
            attention_scale = math.sqrt(1 / head_dim)

        hidden_states_out = self._compiled_attn(
            query,
            key,
            value,
            meta["index_map"],
            meta["win_valid"],
            meta["pool_index_map"],
            meta["pool_valid"],
            attention_scale,
            int(meta["text_length"]),
            int(meta["image_len"]),
            int(meta["pad_index"]),
            int(meta["num_windows"]),
            int(meta["window_area"]),
            bool(meta.get("use_pooling_token", False)) and int(meta.get("num_pool_tokens", 0)) > 0,
            bool(meta.get("pool_text_attend", False)) and int(meta.get("num_pool_tokens", 0)) > 0,
            int(meta.get("num_pool_tokens", 0)),
            int(meta.get("pool_area", 1)),
            bool(meta.get("has_window_padding", False)),
        )
        hidden_states_out = hidden_states_out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_out = hidden_states_out.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_len = encoder_hidden_states.shape[1]
            encoder_hidden_states, hidden_states_out = (
                hidden_states_out[:, : encoder_len],
                hidden_states_out[:, encoder_len:],
            )

            hidden_states_out = attn.to_out[0](hidden_states_out)
            hidden_states_out = attn.to_out[1](hidden_states_out)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            if self.distill:
                attn_outputs.append(hidden_states_out)
            return hidden_states_out, encoder_hidden_states

        if self.distill:
            attn_outputs.append(hidden_states_out)
        return hidden_states_out


@lru_cache(maxsize=4096)
def init_reshape_basa_downsample_flex(
    height,
    width,
    text_length,
    window_size,
    down_factor,
    device,
    total_layers=1,
    layer_index=0,
    denoising_step=None,
    seed=None,
    downsample_text_attend=False,
):
    """Build metadata for the optional learnable downsampled-memory variant.

    The paper configuration uses ``init_reshape_basa_flex`` with plain mean-pooled K/V memory.
    """
    down_factor = int(down_factor)
    if down_factor <= 0:
        raise ValueError(f"down_factor must be positive, got {down_factor}")

    base_meta = init_reshape_basa_flex(
        height=height,
        width=width,
        text_length=text_length,
        window_size=window_size,
        device=device,
        total_layers=total_layers,
        layer_index=layer_index,
        denoising_step=denoising_step,
        seed=seed,
        use_pooling_token=False,
        pool_size=down_factor,
        pool_text_attend=False,
    )
    down_meta = dict(base_meta)

    device = torch.device(device)
    height = int(height)
    width = int(width)
    image_len = height * width
    pad_index = image_len

    down_pad_h = (down_factor - height % down_factor) % down_factor
    down_pad_w = (down_factor - width % down_factor) % down_factor
    down_padded_h = height + down_pad_h
    down_padded_w = width + down_pad_w
    down_h = down_padded_h // down_factor
    down_w = down_padded_w // down_factor
    num_downsample_tokens = down_h * down_w
    downsample_area = down_factor * down_factor

    down_grid = _make_padded_index_grid(height, width, down_padded_h, down_padded_w, pad_index, device)
    downsample_index_map = _partition_windows(down_grid, down_factor)
    downsample_valid = (downsample_index_map != pad_index).contiguous()

    down_meta.update(
        {
            "use_downsample_token": True,
            "down_factor": down_factor,
            "downsample_text_attend": bool(downsample_text_attend),
            "down_h": down_h,
            "down_w": down_w,
            "num_downsample_tokens": num_downsample_tokens,
            "downsample_area": downsample_area,
            "down_pad_h": down_pad_h,
            "down_pad_w": down_pad_w,
            "down_padded_h": down_padded_h,
            "down_padded_w": down_padded_w,
            "downsample_index_map": downsample_index_map,
            "downsample_valid": downsample_valid,
        }
    )

    global meta
    meta = down_meta
    return down_meta


def _basa_downsample_attention_sdpa(
    query,
    key,
    value,
    index_map,
    win_valid,
    downsample_index_map,
    downsample_valid,
    spatial_weight,
    scale: float,
    text_len: int,
    image_len: int,
    pad_index: int,
    num_windows: int,
    window_area: int,
    use_downsample_token: bool,
    downsample_text_attend: bool,
    num_downsample_tokens: int,
    downsample_area: int,
    has_window_padding: bool,
):
    B, H, _S, D = query.shape
    device = query.device
    BH = B * H

    index_map = index_map.to(device, non_blocking=True)
    win_valid = win_valid.to(device, non_blocking=True)
    downsample_index_map = downsample_index_map.to(device, non_blocking=True)
    downsample_valid = downsample_valid.to(device, non_blocking=True)
    spatial_weight = spatial_weight.to(device=device, dtype=key.dtype)

    q_text = query[:, :, :text_len, :]
    k_text = key[:, :, :text_len, :]
    v_text = value[:, :, :text_len, :]

    q_img = query[:, :, text_len:text_len + image_len, :]
    k_img = key[:, :, text_len:text_len + image_len, :]
    v_img = value[:, :, text_len:text_len + image_len, :]

    zero = q_img.new_zeros((B, H, 1, D))
    q_img_padded = torch.cat([q_img, zero], dim=2)
    k_img_padded = torch.cat([k_img, zero.to(dtype=k_img.dtype)], dim=2)
    v_img_padded = torch.cat([v_img, zero.to(dtype=v_img.dtype)], dim=2)

    q_flat = q_img_padded.reshape(BH, image_len + 1, D)
    k_flat = k_img_padded.reshape(BH, image_len + 1, D)
    v_flat = v_img_padded.reshape(BH, image_len + 1, D)

    if use_downsample_token and num_downsample_tokens > 0:
        down_idx_flat = downsample_index_map.reshape(1, -1).expand(BH, -1)
        gathered_k_down = torch.gather(k_flat, 1, down_idx_flat.unsqueeze(-1).expand(-1, -1, D))
        gathered_v_down = torch.gather(v_flat, 1, down_idx_flat.unsqueeze(-1).expand(-1, -1, D))
        gathered_k_down = gathered_k_down.view(B, H, num_downsample_tokens, downsample_area, D)
        gathered_v_down = gathered_v_down.view(B, H, num_downsample_tokens, downsample_area, D)

        down_valid_f = downsample_valid.to(dtype=key.dtype).view(1, 1, num_downsample_tokens, downsample_area, 1)
        down_weight = spatial_weight * down_valid_f
        valid_rescale = float(downsample_area) / down_valid_f.sum(dim=3).clamp_min(1.0)
        k_down = (gathered_k_down * down_weight).sum(dim=3) * valid_rescale
        v_down = (gathered_v_down * down_weight).sum(dim=3) * valid_rescale
    else:
        k_down = v_down = None
        num_downsample_tokens = 0

    # Text queries keep dense access to text and full visual K/V.
    if downsample_text_attend and k_down is not None:
        k_text_context = torch.cat([k_text, k_img, k_down], dim=2)
        v_text_context = torch.cat([v_text, v_img, v_down], dim=2)
    else:
        k_text_context = torch.cat([k_text, k_img], dim=2)
        v_text_context = torch.cat([v_text, v_img], dim=2)
    out_text = _sdpa(q_text, k_text_context, v_text_context, scale=scale)

    im_idx_flat = _index_map_flat(index_map, B, H)
    gathered_q = torch.gather(q_flat, 1, im_idx_flat.unsqueeze(-1).expand(-1, -1, D))
    gathered_k_win = torch.gather(k_flat, 1, im_idx_flat.unsqueeze(-1).expand(-1, -1, D))
    gathered_v_win = torch.gather(v_flat, 1, im_idx_flat.unsqueeze(-1).expand(-1, -1, D))

    q_gathered = gathered_q.view(B, H, num_windows, window_area, D)
    k_gathered = gathered_k_win.view(B, H, num_windows, window_area, D)
    v_gathered = gathered_v_win.view(B, H, num_windows, window_area, D)

    k_parts = [k_text.unsqueeze(2).expand(-1, -1, num_windows, -1, -1), k_gathered]
    v_parts = [v_text.unsqueeze(2).expand(-1, -1, num_windows, -1, -1), v_gathered]
    if k_down is not None:
        k_parts.append(k_down.unsqueeze(2).expand(-1, -1, num_windows, -1, -1))
        v_parts.append(v_down.unsqueeze(2).expand(-1, -1, num_windows, -1, -1))

    k_windows = torch.cat(k_parts, dim=3).reshape(B * H * num_windows, text_len + window_area + num_downsample_tokens, D)
    v_windows = torch.cat(v_parts, dim=3).reshape(B * H * num_windows, text_len + window_area + num_downsample_tokens, D)
    q_windows = q_gathered.reshape(B * H * num_windows, window_area, D)

    attn_bias = None
    if has_window_padding:
        attn_bias = _key_padding_bias_from_window_valid(
            win_valid,
            B=B,
            H=H,
            q_len=window_area,
            text_len=text_len,
            extra_len=num_downsample_tokens,
            dtype=q_windows.dtype,
            device=device,
        )

    out_windows = _sdpa(q_windows, k_windows, v_windows, scale=scale, attn_mask=attn_bias)
    out_windows = out_windows.view(B, H, num_windows, window_area, D)

    out_flat = out_windows.reshape(B, H, num_windows * window_area, D)
    index_map_flat = index_map.reshape(-1)
    result_img = out_flat.new_zeros(B, H, image_len + 1, D)
    idx_expand = index_map_flat.view(1, 1, -1).expand(B, H, -1)
    result_img = result_img.scatter_add(2, idx_expand.unsqueeze(-1).expand(-1, -1, -1, D), out_flat)
    out_img = result_img[:, :, :image_len, :]

    return torch.cat([out_text, out_img], dim=2)


class ReshapeBasaDownsampleFlexAttnProcessor(nn.Module):
    """Optional learnable downsampled-memory processor executed with SDPA."""

    def __init__(self, down_factor=4, distill=False, basa_meta=None):
        super().__init__()
        down_factor = int(down_factor)
        if down_factor <= 0:
            raise ValueError(f"down_factor must be positive, got {down_factor}")

        self.down_factor = down_factor
        self.distill = distill
        self._compiled_attn = None
        self._compiled_signature = None
        self.basa_meta = basa_meta if basa_meta is not None else globals().get("meta", None)
        self.spatial_weight = nn.Parameter(
            torch.ones(1, 1, 1, down_factor * down_factor, 1) / (down_factor * down_factor)
        )

    def set_basa_meta(self, basa_meta):
        self.basa_meta = basa_meta
        return self

    @staticmethod
    def _meta_signature(meta):
        # shift intentionally excluded; index_map is a runtime tensor argument.
        return (
            meta["height"], meta["width"], meta["text_length"],
            meta["window_size"], meta["num_windows"], meta["window_area"],
            meta["image_len"],
            meta.get("down_factor", None),
            meta.get("num_downsample_tokens", 0),
            meta.get("downsample_area", 1),
            bool(meta.get("downsample_text_attend", False)),
            bool(meta.get("has_window_padding", False)),
        )

    def _ensure_compiled(self, meta):
        signature = self._meta_signature(meta)
        if self._compiled_attn is not None and self._compiled_signature == signature:
            return
        self._compiled_attn = _maybe_compile(_basa_downsample_attention_sdpa, ("basa_downsample_sdpa", signature))
        self._compiled_signature = signature

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        proportional_attention=False,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        meta = self.basa_meta if self.basa_meta is not None else globals().get("meta", None)
        if meta is None or "downsample_index_map" not in meta:
            raise RuntimeError(
                "ReshapeBasaDownsampleFlexAttnProcessor.basa_meta is None or incomplete; "
                "call init_reshape_basa_downsample_flex(...) before using this processor, "
                "or pass/set basa_meta explicitly."
            )
        if int(meta.get("down_factor", self.down_factor)) != self.down_factor:
            raise ValueError(
                f"Processor down_factor={self.down_factor} does not match meta down_factor={meta.get('down_factor')}."
            )
        if "win_valid" not in meta:
            meta = dict(meta)
            meta["win_valid"] = (meta["index_map"] != meta["pad_index"]).contiguous()
            meta["has_window_padding"] = bool(meta.get("pad_h", 0) or meta.get("pad_w", 0))
            self.basa_meta = meta
        if "downsample_valid" not in meta:
            meta = dict(meta)
            meta["downsample_valid"] = (meta["downsample_index_map"] != meta["pad_index"]).contiguous()
            self.basa_meta = meta

        self._ensure_compiled(meta)

        train_seq_len = 64 ** 2 + 512
        if proportional_attention:
            effective_kv_len = key.size(2) + int(meta.get("num_downsample_tokens", 0))
            attention_scale = math.sqrt(math.log(effective_kv_len, train_seq_len) / head_dim)
        else:
            attention_scale = math.sqrt(1 / head_dim)

        hidden_states_out = self._compiled_attn(
            query,
            key,
            value,
            meta["index_map"],
            meta["win_valid"],
            meta["downsample_index_map"],
            meta["downsample_valid"],
            self.spatial_weight,
            attention_scale,
            int(meta["text_length"]),
            int(meta["image_len"]),
            int(meta["pad_index"]),
            int(meta["num_windows"]),
            int(meta["window_area"]),
            bool(meta.get("use_downsample_token", False)) and int(meta.get("num_downsample_tokens", 0)) > 0,
            bool(meta.get("downsample_text_attend", False)) and int(meta.get("num_downsample_tokens", 0)) > 0,
            int(meta.get("num_downsample_tokens", 0)),
            int(meta.get("downsample_area", 1)),
            bool(meta.get("has_window_padding", False)),
        )
        hidden_states_out = hidden_states_out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_out = hidden_states_out.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_len = encoder_hidden_states.shape[1]
            encoder_hidden_states, hidden_states_out = (
                hidden_states_out[:, :encoder_len],
                hidden_states_out[:, encoder_len:],
            )

            hidden_states_out = attn.to_out[0](hidden_states_out)
            hidden_states_out = attn.to_out[1](hidden_states_out)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            if self.distill:
                attn_outputs.append(hidden_states_out)
            return hidden_states_out, encoder_hidden_states

        if self.distill:
            attn_outputs.append(hidden_states_out)
        return hidden_states_out


