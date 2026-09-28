import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from diffusers.models.attention_processor import Attention
from typing import Optional
from functools import lru_cache
from diffusers.models.embeddings import apply_rotary_emb


attn_outputs_teacher = []
attn_outputs = []
offset_step=0

class FluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

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
        proportional_attention=False
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
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

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
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

            # attention
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

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, scale=attention_scale)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            if self.distill:
                attn_outputs_teacher.append(hidden_states)
            return hidden_states



@lru_cache
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
    """Build BASA metadata for interleaved shifted windows and mean-pooled global K/V memory.

    Pooled tokens are K/V-only memory, so the model-state sequence length is unchanged.
    """
    global offset_step
    if denoising_step is None:
        if seed is not None:
            g = torch.Generator(device=device if str(device).startswith("cuda") else "cpu")
            g.manual_seed(seed)
            offset = torch.randint(0, window_size, (1,), generator=g, device=device).item()
        else:
            global offset_step
            offset = offset_step % window_size
            offset_step+=1
    else:
        offset = int(denoising_step)

    pool_size = int(pool_size)
    if pool_size <= 0:
        raise ValueError(f"pool_size must be positive, got {pool_size}")

    # Interleave layer offsets as in the BASA shift schedule.
    delta = max(1, int(window_size) // int(total_layers))
    seq = [(i * delta) % window_size for i in range(total_layers)]

    half = total_layers // 2
    interleaved = []
    for a, b in zip(seq[:half], seq[half:]):
        interleaved.append(a)
        interleaved.append(b)
    if total_layers % 2 == 1:
        interleaved.append(seq[half])

    shift_val = (interleaved[layer_index % total_layers] + offset) % window_size

    # Pad only when the spatial grid is not divisible by the window size.
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    padded_h = height + pad_h
    padded_w = width + pad_w

    win_h = padded_h // window_size
    win_w = padded_w // window_size
    num_windows = win_h * win_w
    window_area = window_size * window_size

    image_len = height * width
    pad_index = image_len  # append one dummy Q/K/V slot for padded spatial positions

    def create_padded_coords(target_h, target_w):
        padded_coords = torch.full((target_h, target_w), pad_index, dtype=torch.long, device=device)
        for y in range(target_h):
            for x in range(target_w):
                if y < height and x < width:
                    padded_coords[y, x] = y * width + x
        return padded_coords

    def create_shifted_coords(shift):
        padded_coords = create_padded_coords(padded_h, padded_w)
        if shift > 0:
            # Cyclic shift first, then partition shifted coordinates into windows.
            return torch.roll(padded_coords, shifts=(-shift, -shift), dims=(0, 1))
        return padded_coords

    def create_window_index_map(shifted_coords):
        coords = []
        for wy in range(win_h):
            for wx in range(win_w):
                top = wy * window_size
                left = wx * window_size
                for dy in range(window_size):
                    for dx in range(window_size):
                        coords.append(shifted_coords[top + dy, left + dx].item())
        return torch.tensor(coords, dtype=torch.long, device=device).view(num_windows, window_area).contiguous()

    def create_pool_index_map():
        """Plain mean-pooling regions on the original, unshifted image-token grid."""
        pool_pad_h = (pool_size - height % pool_size) % pool_size
        pool_pad_w = (pool_size - width % pool_size) % pool_size
        pool_padded_h = height + pool_pad_h
        pool_padded_w = width + pool_pad_w
        pool_h = pool_padded_h // pool_size
        pool_w = pool_padded_w // pool_size
        num_pool_tokens = pool_h * pool_w
        pool_area = pool_size * pool_size

        padded_coords = create_padded_coords(pool_padded_h, pool_padded_w)
        coords = []
        for py in range(pool_h):
            for px in range(pool_w):
                top = py * pool_size
                left = px * pool_size
                for dy in range(pool_size):
                    for dx in range(pool_size):
                        coords.append(padded_coords[top + dy, left + dx].item())

        pool_index_map = torch.tensor(coords, dtype=torch.long, device=device).view(num_pool_tokens, pool_area).contiguous()
        return (
            pool_index_map,
            pool_h,
            pool_w,
            num_pool_tokens,
            pool_area,
            pool_pad_h,
            pool_pad_w,
            pool_padded_h,
            pool_padded_w,
        )

    shifted_coords = create_shifted_coords(shift_val)
    index_map = create_window_index_map(shifted_coords)

    if use_pooling_token:
        (
            pool_index_map,
            pool_h,
            pool_w,
            num_pool_tokens,
            pool_area,
            pool_pad_h,
            pool_pad_w,
            pool_padded_h,
            pool_padded_w,
        ) = create_pool_index_map()
    else:
        pool_index_map = torch.empty((0, 1), dtype=torch.long, device=device).contiguous()
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
        "index_map": index_map,  # Q/K/V local window map, shape (num_windows, window_area)
        "device": device,
        # Plain pooling-token metadata.
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
        "pool_index_map": pool_index_map,  # shape (num_pool_tokens, pool_area)
    }
    return meta


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
        return (
            meta["height"], meta["width"], meta["text_length"],
            meta["window_size"], meta["num_windows"], meta["window_area"],
            meta["image_len"], meta.get("shift", None),
            bool(meta.get("use_pooling_token", False)),
            meta.get("pool_size", None),
            meta.get("num_pool_tokens", 0),
            meta.get("pool_area", 1),
            bool(meta.get("pool_text_attend", False)),
        )

    def _ensure_compiled(self, meta):
        signature = self._meta_signature(meta)
        if self._compiled_attn is not None and self._compiled_signature == signature:
            return

        def basa_pooling_attention(
            query,
            key,
            value,
            index_map,
            pool_index_map,
            scale: float,
        ):
            # query/key/value: (B, heads, seq_len, head_dim)
            B, H, S_q, D = query.shape
            text_len = meta["text_length"]
            image_len = meta["image_len"]
            pad_index = meta["pad_index"]
            num_windows = meta["num_windows"]
            window_area = meta["window_area"]
            use_pooling_token = bool(meta.get("use_pooling_token", False)) and meta.get("num_pool_tokens", 0) > 0
            pool_text_attend = bool(meta.get("pool_text_attend", False)) and use_pooling_token
            num_pool_tokens = meta.get("num_pool_tokens", 0)
            pool_area = meta.get("pool_area", 1)

            index_map = index_map.to(query.device)            # (num_windows, window_area)
            pool_index_map = pool_index_map.to(query.device)  # (num_pool_tokens, pool_area), may be empty

            # Split text/image tokens.
            q_text = query[:, :, :text_len, :]
            k_text = key[:, :, :text_len, :]
            v_text = value[:, :, :text_len, :]

            q_img = query[:, :, text_len:text_len + image_len, :]
            k_img = key[:, :, text_len:text_len + image_len, :]
            v_img = value[:, :, text_len:text_len + image_len, :]

            # Append one zero slot so padded spatial positions can be gathered safely.
            zero_q = torch.zeros((B, H, 1, D), dtype=q_img.dtype, device=q_img.device)
            zero_k = torch.zeros((B, H, 1, D), dtype=k_img.dtype, device=k_img.device)
            zero_v = torch.zeros((B, H, 1, D), dtype=v_img.dtype, device=v_img.device)
            q_img_padded = torch.cat([q_img, zero_q], dim=2)
            k_img_padded = torch.cat([k_img, zero_k], dim=2)
            v_img_padded = torch.cat([v_img, zero_v], dim=2)

            q_flat = q_img_padded.reshape(B * H, image_len + 1, D)
            k_flat = k_img_padded.reshape(B * H, image_len + 1, D)
            v_flat = v_img_padded.reshape(B * H, image_len + 1, D)

            # -------- Plain mean-pooling tokens: extra K/V memory only --------
            if use_pooling_token:
                pool_idx = pool_index_map.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
                pool_idx_flat = pool_idx.reshape(B * H, num_pool_tokens * pool_area)

                gathered_k_pool = torch.gather(k_flat, 1, pool_idx_flat.unsqueeze(-1).expand(-1, -1, D))
                gathered_v_pool = torch.gather(v_flat, 1, pool_idx_flat.unsqueeze(-1).expand(-1, -1, D))
                gathered_k_pool = gathered_k_pool.view(B, H, num_pool_tokens, pool_area, D)
                gathered_v_pool = gathered_v_pool.view(B, H, num_pool_tokens, pool_area, D)

                pool_valid = (pool_index_map != pad_index).to(dtype=k_img.dtype, device=query.device)
                pool_valid = pool_valid.view(1, 1, num_pool_tokens, pool_area, 1)
                pool_denom = pool_valid.sum(dim=3).clamp_min(1.0)

                k_pool = (gathered_k_pool * pool_valid).sum(dim=3) / pool_denom  # (B,H,Np,D)
                v_pool = (gathered_v_pool * pool_valid).sum(dim=3) / pool_denom  # (B,H,Np,D)

            # -------- Text query path --------
            # Default keeps original behavior: Text -> Text + Full Image.
            # pool_text_attend=True optionally gives text queries access to pooling K/V.
            scores_tt = torch.einsum("bhqd,bhkd->bhqk", q_text, k_text) * scale
            scores_ti = torch.einsum("bhqd,bhkd->bhqk", q_text, k_img) * scale

            if pool_text_attend:
                scores_tp = torch.einsum("bhqd,bhkd->bhqk", q_text, k_pool) * scale
                scores_text_concat = torch.cat([scores_tt, scores_ti, scores_tp], dim=-1)
                v_concat_text = torch.cat([v_text, v_img, v_pool], dim=2)
            else:
                scores_text_concat = torch.cat([scores_tt, scores_ti], dim=-1)
                v_concat_text = torch.cat([v_text, v_img], dim=2)

            probs_text = torch.softmax(scores_text_concat, dim=-1)
            out_text = torch.einsum("bhqk,bhkd->bhqd", probs_text, v_concat_text)

            # Image queries attend to text, their shifted local window, and pooled global K/V.
            im_idx = index_map.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
            im_idx_flat = im_idx.reshape(B * H, num_windows * window_area)

            gathered_q = torch.gather(q_flat, 1, im_idx_flat.unsqueeze(-1).expand(-1, -1, D))
            gathered_k_win = torch.gather(k_flat, 1, im_idx_flat.unsqueeze(-1).expand(-1, -1, D))
            gathered_v_win = torch.gather(v_flat, 1, im_idx_flat.unsqueeze(-1).expand(-1, -1, D))

            q_gathered = gathered_q.view(B, H, num_windows, window_area, D)
            k_gathered = gathered_k_win.view(B, H, num_windows, window_area, D)
            v_gathered = gathered_v_win.view(B, H, num_windows, window_area, D)

            k_text_expand = k_text.unsqueeze(2).expand(-1, -1, num_windows, -1, -1)
            v_text_expand = v_text.unsqueeze(2).expand(-1, -1, num_windows, -1, -1)

            scores_q_text = torch.einsum("bhnqd,bhnkd->bhnqk", q_gathered, k_text_expand) * scale
            scores_q_win = torch.einsum("bhnqd,bhnkd->bhnqk", q_gathered, k_gathered) * scale

            # Do not let zero-padded window slots absorb probability mass.
            win_valid = (index_map != pad_index).view(1, 1, num_windows, 1, window_area).to(query.device)
            scores_q_win = scores_q_win.masked_fill(~win_valid, -10000.0)

            if use_pooling_token:
                k_pool_expand = k_pool.unsqueeze(2).expand(-1, -1, num_windows, -1, -1)
                v_pool_expand = v_pool.unsqueeze(2).expand(-1, -1, num_windows, -1, -1)
                scores_q_pool = torch.einsum("bhnqd,bhnkd->bhnqk", q_gathered, k_pool_expand) * scale

                scores_img_concat = torch.cat([scores_q_text, scores_q_win, scores_q_pool], dim=-1)
                v_concat_windows = torch.cat([v_text_expand, v_gathered, v_pool_expand], dim=3)
            else:
                scores_img_concat = torch.cat([scores_q_text, scores_q_win], dim=-1)
                v_concat_windows = torch.cat([v_text_expand, v_gathered], dim=3)

            probs_img = torch.softmax(scores_img_concat, dim=-1)
            out_windows = torch.einsum("bhnqk,bhnkd->bhnqd", probs_img, v_concat_windows)

            # Scatter shifted W x W query outputs back to original image-token layout.
            out_flat = out_windows.view(B, H, num_windows * window_area, D)
            index_map_flat = index_map.view(-1).to(query.device)
            result_img = torch.zeros(B, H, image_len + 1, D, dtype=out_flat.dtype, device=out_flat.device)
            idx_expand = index_map_flat.unsqueeze(0).unsqueeze(0).expand(B, H, -1)
            result_img = result_img.scatter_add(2, idx_expand.unsqueeze(-1).expand(-1, -1, -1, D), out_flat)
            out_img = result_img[:, :, :image_len, :]

            out_all = torch.cat([out_text, out_img], dim=2)
            return out_all

        self._compiled_attn = torch.compile(basa_pooling_attention, dynamic=False)
        self._compiled_signature = signature

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor = None,
        image_rotary_emb: torch.Tensor = None,
        proportional_attention=False
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

        # Backward compatibility: tolerate old metadata that does not yet contain pooling fields.
        if "pool_index_map" not in meta or meta.get("pool_index_map") is None:
            meta = dict(meta)
            meta["use_pooling_token"] = False
            meta["pool_text_attend"] = False
            meta["num_pool_tokens"] = 0
            meta["pool_area"] = 1
            meta["pool_size"] = 1
            meta["pool_index_map"] = torch.empty((0, 1), dtype=torch.long, device=query.device).contiguous()
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
            meta["pool_index_map"],
            scale=attention_scale,
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
        else:
            if self.distill:
                attn_outputs.append(hidden_states_out)
            return hidden_states_out

@lru_cache
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

    # Reuse the same shifted-window partition and replace mean pooling with learnable downsampling.
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

    image_len = int(height) * int(width)
    pad_index = image_len

    down_pad_h = (down_factor - height % down_factor) % down_factor
    down_pad_w = (down_factor - width % down_factor) % down_factor
    down_padded_h = height + down_pad_h
    down_padded_w = width + down_pad_w
    down_h = down_padded_h // down_factor
    down_w = down_padded_w // down_factor
    num_downsample_tokens = down_h * down_w
    downsample_area = down_factor * down_factor

    padded_coords = torch.full((down_padded_h, down_padded_w), pad_index, dtype=torch.long, device=device)
    for y in range(down_padded_h):
        for x in range(down_padded_w):
            if y < height and x < width:
                padded_coords[y, x] = y * width + x

    coords = []
    for py in range(down_h):
        for px in range(down_w):
            top = py * down_factor
            left = px * down_factor
            for dy in range(down_factor):
                for dx in range(down_factor):
                    coords.append(padded_coords[top + dy, left + dx].item())

    downsample_index_map = torch.tensor(coords, dtype=torch.long, device=device).view(
        num_downsample_tokens, downsample_area
    ).contiguous()

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
        }
    )

    global meta
    meta = down_meta
    return down_meta


class ReshapeBasaDownsampleFlexAttnProcessor(nn.Module):
    """Optional learnable downsampled-memory processor used only for ablation/compatibility."""

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

        # Learn one weight per position inside each downsampling cell.
        self.spatial_weight = nn.Parameter(
            torch.ones(1, 1, 1, down_factor * down_factor, 1) / (down_factor * down_factor)
        )

    def set_basa_meta(self, basa_meta):
        self.basa_meta = basa_meta
        return self

    @staticmethod
    def _meta_signature(meta):
        return (
            meta["height"], meta["width"], meta["text_length"],
            meta["window_size"], meta["num_windows"], meta["window_area"],
            meta["image_len"], meta.get("shift", None),
            meta.get("down_factor", None),
            meta.get("num_downsample_tokens", 0),
            meta.get("downsample_area", 1),
            bool(meta.get("downsample_text_attend", False)),
        )

    def _ensure_compiled(self, meta):
        signature = self._meta_signature(meta)
        if self._compiled_attn is not None and self._compiled_signature == signature:
            return

        def basa_downsample_attention(
            query,
            key,
            value,
            index_map,
            downsample_index_map,
            spatial_weight,
            scale: float,
        ):
            # query/key/value: (B, heads, seq_len, head_dim)
            B, H, S_q, D = query.shape
            text_len = meta["text_length"]
            image_len = meta["image_len"]
            pad_index = meta["pad_index"]
            num_windows = meta["num_windows"]
            window_area = meta["window_area"]
            num_downsample_tokens = meta.get("num_downsample_tokens", 0)
            downsample_area = meta.get("downsample_area", 1)
            use_downsample_token = bool(meta.get("use_downsample_token", False)) and num_downsample_tokens > 0
            downsample_text_attend = bool(meta.get("downsample_text_attend", False)) and use_downsample_token

            index_map = index_map.to(query.device)
            downsample_index_map = downsample_index_map.to(query.device)
            spatial_weight = spatial_weight.to(device=query.device, dtype=key.dtype)

            # Split text/image tokens.
            q_text = query[:, :, :text_len, :]
            k_text = key[:, :, :text_len, :]
            v_text = value[:, :, :text_len, :]

            q_img = query[:, :, text_len:text_len + image_len, :]
            k_img = key[:, :, text_len:text_len + image_len, :]
            v_img = value[:, :, text_len:text_len + image_len, :]

            # Append one zero slot for padded spatial positions.
            zero_q = torch.zeros((B, H, 1, D), dtype=q_img.dtype, device=q_img.device)
            zero_k = torch.zeros((B, H, 1, D), dtype=k_img.dtype, device=k_img.device)
            zero_v = torch.zeros((B, H, 1, D), dtype=v_img.dtype, device=v_img.device)
            q_img_padded = torch.cat([q_img, zero_q], dim=2)
            k_img_padded = torch.cat([k_img, zero_k], dim=2)
            v_img_padded = torch.cat([v_img, zero_v], dim=2)

            q_flat = q_img_padded.reshape(B * H, image_len + 1, D)
            k_flat = k_img_padded.reshape(B * H, image_len + 1, D)
            v_flat = v_img_padded.reshape(B * H, image_len + 1, D)

            # -------- Learnable downsampled K/V memory tokens --------
            if use_downsample_token:
                down_idx = downsample_index_map.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
                down_idx_flat = down_idx.reshape(B * H, num_downsample_tokens * downsample_area)

                gathered_k_down = torch.gather(k_flat, 1, down_idx_flat.unsqueeze(-1).expand(-1, -1, D))
                gathered_v_down = torch.gather(v_flat, 1, down_idx_flat.unsqueeze(-1).expand(-1, -1, D))
                gathered_k_down = gathered_k_down.view(B, H, num_downsample_tokens, downsample_area, D)
                gathered_v_down = gathered_v_down.view(B, H, num_downsample_tokens, downsample_area, D)

                down_valid = (downsample_index_map != pad_index).to(dtype=key.dtype, device=query.device)
                down_valid = down_valid.view(1, 1, num_downsample_tokens, downsample_area, 1)

                # Keep CLEAR-style learnable weighted pooling. For padded boundary cells,
                # rescale by valid-token count so the initialized weights still behave like
                # a true mean without constraining the learnable weights during training.
                down_weight = spatial_weight * down_valid
                valid_rescale = float(downsample_area) / down_valid.sum(dim=3).clamp_min(1.0)
                k_down = (gathered_k_down * down_weight).sum(dim=3) * valid_rescale
                v_down = (gathered_v_down * down_weight).sum(dim=3) * valid_rescale

            # -------- Text query path --------
            # Text queries retain dense text-to-visual context.
            scores_tt = torch.einsum("bhqd,bhkd->bhqk", q_text, k_text) * scale
            scores_ti = torch.einsum("bhqd,bhkd->bhqk", q_text, k_img) * scale

            if downsample_text_attend:
                scores_td = torch.einsum("bhqd,bhkd->bhqk", q_text, k_down) * scale
                scores_text_concat = torch.cat([scores_tt, scores_ti, scores_td], dim=-1)
                v_concat_text = torch.cat([v_text, v_img, v_down], dim=2)
            else:
                scores_text_concat = torch.cat([scores_tt, scores_ti], dim=-1)
                v_concat_text = torch.cat([v_text, v_img], dim=2)

            probs_text = torch.softmax(scores_text_concat, dim=-1)
            out_text = torch.einsum("bhqk,bhkd->bhqd", probs_text, v_concat_text)

            # -------- Image query path: text + shifted window + downsampled K/V --------
            im_idx = index_map.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
            im_idx_flat = im_idx.reshape(B * H, num_windows * window_area)

            gathered_q = torch.gather(q_flat, 1, im_idx_flat.unsqueeze(-1).expand(-1, -1, D))
            gathered_k_win = torch.gather(k_flat, 1, im_idx_flat.unsqueeze(-1).expand(-1, -1, D))
            gathered_v_win = torch.gather(v_flat, 1, im_idx_flat.unsqueeze(-1).expand(-1, -1, D))

            q_gathered = gathered_q.view(B, H, num_windows, window_area, D)
            k_gathered = gathered_k_win.view(B, H, num_windows, window_area, D)
            v_gathered = gathered_v_win.view(B, H, num_windows, window_area, D)

            k_text_expand = k_text.unsqueeze(2).expand(-1, -1, num_windows, -1, -1)
            v_text_expand = v_text.unsqueeze(2).expand(-1, -1, num_windows, -1, -1)

            scores_q_text = torch.einsum("bhnqd,bhnkd->bhnqk", q_gathered, k_text_expand) * scale
            scores_q_win = torch.einsum("bhnqd,bhnkd->bhnqk", q_gathered, k_gathered) * scale

            # Do not let zero-padded window slots absorb probability mass.
            win_valid = (index_map != pad_index).view(1, 1, num_windows, 1, window_area).to(query.device)
            scores_q_win = scores_q_win.masked_fill(~win_valid, -10000.0)

            if use_downsample_token:
                k_down_expand = k_down.unsqueeze(2).expand(-1, -1, num_windows, -1, -1)
                v_down_expand = v_down.unsqueeze(2).expand(-1, -1, num_windows, -1, -1)
                scores_q_down = torch.einsum("bhnqd,bhnkd->bhnqk", q_gathered, k_down_expand) * scale

                scores_img_concat = torch.cat([scores_q_text, scores_q_win, scores_q_down], dim=-1)
                v_concat_windows = torch.cat([v_text_expand, v_gathered, v_down_expand], dim=3)
            else:
                scores_img_concat = torch.cat([scores_q_text, scores_q_win], dim=-1)
                v_concat_windows = torch.cat([v_text_expand, v_gathered], dim=3)

            probs_img = torch.softmax(scores_img_concat, dim=-1)
            out_windows = torch.einsum("bhnqk,bhnkd->bhnqd", probs_img, v_concat_windows)

            # Scatter shifted-window query outputs back to the original image-token layout.
            out_flat = out_windows.view(B, H, num_windows * window_area, D)
            index_map_flat = index_map.view(-1).to(query.device)
            result_img = torch.zeros(B, H, image_len + 1, D, dtype=out_flat.dtype, device=out_flat.device)
            idx_expand = index_map_flat.unsqueeze(0).unsqueeze(0).expand(B, H, -1)
            result_img = result_img.scatter_add(2, idx_expand.unsqueeze(-1).expand(-1, -1, -1, D), out_flat)
            out_img = result_img[:, :, :image_len, :]

            return torch.cat([out_text, out_img], dim=2)

        self._compiled_attn = torch.compile(basa_downsample_attention, dynamic=False)
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
            meta["downsample_index_map"],
            self.spatial_weight,
            scale=attention_scale,
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
        else:
            if self.distill:
                attn_outputs.append(hidden_states_out)
            return hidden_states_out


