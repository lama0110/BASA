import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from .wan_video_camera_controller import SimpleAdapter
from ..core.gradient import gradient_checkpoint_forward
from .wantodance import WanToDanceRotaryEmbedding, WanToDanceMusicEncoderLayer

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    freqs = freqs.to(torch.complex64) if freqs.device.type == "npu" else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def set_to_torch_norm(models):
    for model in models:
        for module in model.modules():
            if isinstance(module, RMSNorm):
                module.use_torch_norm = True


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.use_torch_norm = False
        self.normalized_shape = (dim,)

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        if self.use_torch_norm:
            return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        else:        
            return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class BasaSparseSelfAttention(nn.Module):
    """
    Inference-only BASA sparse self-attention with the same math as the distilled StudentSelfAttention.

    Kept from the original version:
      1) video latent grid shape (F, H, W), updated by the pipeline before block execution
      2) shifted spatial window attention
      3) temporal full attention inside each spatial window
      4) K/V-only spatial pooling memory
      5) unchanged forward(self, x, freqs) interface

    Added from Ultra Flash, with minimal adaptation:
      1) block/window-level Q/K mean pooling
      2) content-adaptive top-k true-window selection
      3) spatial local candidate mask

    Removed from Ultra Flash:
      1) temporal causal mask, because this use case is not long-video streaming

    Not added:
      1) no competition mechanism
      2) no spatial distance penalty
      3) no layer schedule
      4) no pooling gate
      5) no extra trainable parameters
    """
# The denoising-step term is an optional schedule component. In practice,
# layer-wise interleaving alone is already effective for some configurations
# (for example w=10, p=4). Adding the step term gives the full BASA schedule
# without changing the underlying sparse-attention branches.
    def __init__(
        self,
        dim: int,
        num_heads: int,
        block_index: int = 0,
        eps: float = 1e-6,
        spatial_shape: Tuple[int, int, int] = (21, 30, 52),
        window_size: Tuple[int, int] = (15, 26),
        pool_size: Tuple[int, int] = (4, 4),
        dynamic_topk: int = 1,
        local_window_radius: int = 1,
        use_pooling_token: bool = True,
        use_dynamic_true_window: bool = True,
        num_layers: int = 30,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.block_index = int(block_index)

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        # Safe fallback/debug path for unexpected sequence lengths.
        self.attn = AttentionModule(self.num_heads)

        # These are inference/runtime attributes, not trainable parameters.
        self.spatial_shape = tuple(int(v) for v in spatial_shape)
        self.window_size = tuple(int(v) for v in window_size)
        self.num_layers = int(num_layers)

        # 1-based runtime denoising-step offset, matching image BASA.
        self.denoising_step = 1

        # K/V-only spatial pooling memory branch.
        self.use_pooling_token = bool(use_pooling_token)
        self.pool_size = tuple(int(v) for v in pool_size)

        # Content-adaptive true-window routing.
        self.use_dynamic_true_window = bool(use_dynamic_true_window)
        self.dynamic_topk = int(dynamic_topk)

        # Non-causal spatial local candidate mask radius on the window grid.
        self.local_window_radius = int(local_window_radius)

        # Normal path below uses the project-level flash_attention backend.
        # This avoids requiring/compiling FlexAttention while keeping the same
        # per-window dense attention formula.

    @staticmethod
    def _build_interleaved_offsets(size: int, total_layers: int):
        """
        Build layer-wise shifts that cover the window range in an interleaved way.

        Example for size=15:
            0, 8, 1, 9, 2, 10, ...
        then repeats/truncates to total_layers.
        """
        size = int(size)
        total_layers = max(1, int(total_layers))

        if size <= 1:
            return [0 for _ in range(total_layers)]

        half = (size + 1) // 2
        offsets = []

        for i in range(half):
            offsets.append(i)
            if i + half < size:
                offsets.append(i + half)

        if len(offsets) < total_layers:
            repeat = (total_layers + len(offsets) - 1) // len(offsets)
            offsets = (offsets * repeat)[:total_layers]
        else:
            offsets = offsets[:total_layers]

        return offsets

    def _get_layer_shift(self):
        """Compute the spatial shift from DiT depth and denoising step."""
        Wh, Ww = self.window_size
        layer_id = self.block_index % self.num_layers
        step = int(self.denoising_step)

        offsets_h = self._build_interleaved_offsets(Wh, self.num_layers)
        offsets_w = self._build_interleaved_offsets(Ww, self.num_layers)

        shift_h = (offsets_h[layer_id] + step) % Wh
        shift_w = (offsets_w[layer_id] + step) % Ww
        return shift_h, shift_w

    @staticmethod
    def _partition_spatial_windows(x: torch.Tensor, window_size):
        """
        Partition only spatial dimensions into windows, while keeping full temporal attention.

        Input:
            x: [B, F, H, W, heads, head_dim]

        Output:
            windows: [B * num_spatial_windows, F * Wh * Ww, heads, head_dim]

        Each spatial window contains all frames.
        """
        Wh, Ww = window_size
        B, T, H, W, heads, head_dim = x.shape

        pad_h = (Wh - H % Wh) % Wh
        pad_w = (Ww - W % Ww) % Ww

        if pad_h != 0 or pad_w != 0:
            x = x.permute(0, 1, 4, 5, 2, 3).contiguous()
            x = F.pad(x, (0, pad_w, 0, pad_h))
            x = x.permute(0, 1, 4, 5, 2, 3).contiguous()

        Hp = H + pad_h
        Wp = W + pad_w
        nH = Hp // Wh
        nW = Wp // Ww

        x = x.view(B, T, nH, Wh, nW, Ww, heads, head_dim)
        x = x.permute(0, 2, 4, 1, 3, 5, 6, 7).contiguous()

        windows = x.view(B * nH * nW, T * Wh * Ww, heads, head_dim)

        meta = {
            "B": B,
            "T": T,
            "H": H,
            "W": W,
            "Hp": Hp,
            "Wp": Wp,
            "pad_h": pad_h,
            "pad_w": pad_w,
            "nH": nH,
            "nW": nW,
            "Wh": Wh,
            "Ww": Ww,
            "heads": heads,
            "head_dim": head_dim,
        }
        return windows, meta

    @staticmethod
    def _reverse_spatial_windows(windows: torch.Tensor, meta):
        """
        Reverse _partition_spatial_windows.

        Input:
            windows: [B * nH * nW, T * Wh * Ww, heads, head_dim]

        Output:
            x: [B, T, H, W, heads, head_dim]
        """
        B = meta["B"]
        T = meta["T"]
        H = meta["H"]
        W = meta["W"]
        Hp = meta["Hp"]
        Wp = meta["Wp"]
        nH = meta["nH"]
        nW = meta["nW"]
        Wh = meta["Wh"]
        Ww = meta["Ww"]
        heads = meta["heads"]
        head_dim = meta["head_dim"]

        x = windows.view(B, nH, nW, T, Wh, Ww, heads, head_dim)
        x = x.permute(0, 3, 1, 4, 2, 5, 6, 7).contiguous()
        x = x.view(B, T, Hp, Wp, heads, head_dim)
        x = x[:, :, :H, :W, :, :].contiguous()
        return x

    @staticmethod
    def _make_spatial_pool_tokens(x: torch.Tensor, pool_size):
        """
        Make K/V-only pooling tokens from unshifted K/V maps.

        Input:
            x: [B, F, H, W, heads, head_dim]

        Output:
            pool_tokens: [B, F * num_pool_spatial, heads, head_dim]

        Important:
          - Pooling is only over spatial dimensions.
          - Temporal dimension F is preserved.
          - This is the original pooling-token branch, kept for boundary compensation.
        """
        Ph, Pw = pool_size
        B, T, H, W, heads, head_dim = x.shape

        pad_h = (Ph - H % Ph) % Ph
        pad_w = (Pw - W % Pw) % Pw

        if pad_h != 0 or pad_w != 0:
            x_pad = x.permute(0, 1, 4, 5, 2, 3).contiguous()
            x_pad = F.pad(x_pad, (0, pad_w, 0, pad_h))
            x_pad = x_pad.permute(0, 1, 4, 5, 2, 3).contiguous()

            valid = torch.ones(
                (B, T, H, W, 1, 1),
                dtype=x.dtype,
                device=x.device,
            )
            valid = valid.permute(0, 1, 4, 5, 2, 3).contiguous()
            valid = F.pad(valid, (0, pad_w, 0, pad_h))
            valid = valid.permute(0, 1, 4, 5, 2, 3).contiguous()
        else:
            x_pad = x
            valid = torch.ones(
                (B, T, H, W, 1, 1),
                dtype=x.dtype,
                device=x.device,
            )

        Hp = H + pad_h
        Wp = W + pad_w
        pH = Hp // Ph
        pW = Wp // Pw

        x_pool = x_pad.view(B, T, pH, Ph, pW, Pw, heads, head_dim)
        valid_pool = valid.view(B, T, pH, Ph, pW, Pw, 1, 1)

        numer = (x_pool * valid_pool).sum(dim=(3, 5))
        denom = valid_pool.sum(dim=(3, 5)).clamp_min(1.0)

        pooled = numer / denom
        pooled = pooled.contiguous().view(B, T * pH * pW, heads, head_dim)
        return pooled

    def _make_local_candidate_mask(self, meta, device):
        """
        Non-causal spatial local mask M_local on the spatial-window grid.

        Output:
            mask: [num_windows, num_windows], bool

        mask[i, j] = True means window j is a valid candidate for query window i.
        No temporal causal constraint is used.
        """
        nH = meta["nH"]
        nW = meta["nW"]
        radius = int(self.local_window_radius)

        rows = torch.arange(nH, device=device)
        cols = torch.arange(nW, device=device)
        rr, cc = torch.meshgrid(rows, cols, indexing="ij")
        coords = torch.stack([rr.reshape(-1), cc.reshape(-1)], dim=-1)
        # coords: [num_windows, 2]

        diff = coords[:, None, :] - coords[None, :, :]
        dist_inf = diff.abs().amax(dim=-1)

        mask = dist_inf <= radius
        return mask

    def _select_dynamic_true_windows(self, q_win: torch.Tensor, k_win: torch.Tensor, v_win: torch.Tensor, meta):
        """
        Ultra-Flash-style content-adaptive top-k selection at true-window level.

        Inputs:
            q_win/k_win/v_win:
                [B * num_windows, local_len, heads, head_dim]

        Output:
            k_dyn/v_dyn:
                [B * num_windows, dynamic_topk * local_len, heads, head_dim]

        Mathematical form:
            q_bar_i = mean_{u in window i} q_u
            k_bar_j = mean_{v in window j} k_v

            s_ij = mean_h <q_bar_i^h, k_bar_j^h> / sqrt(d_h)

            A(i) = TopK_{j in M_local(i), j != i} s_ij

        This is the non-causal version of the paper's dynamic block-sparse selection.
        It uses true window tokens, not pooled summary tokens.
        """
        if (not self.use_dynamic_true_window) or self.dynamic_topk <= 0:
            return None, None

        B = meta["B"]
        nH = meta["nH"]
        nW = meta["nW"]
        num_windows = nH * nW

        Bwin, local_len, heads, head_dim = q_win.shape
        if num_windows <= 1:
            return None, None

        assert Bwin == B * num_windows

        # [B*num_windows, S, heads, D] -> [B, num_windows, S, heads, D]
        q_blocks = q_win.view(B, num_windows, local_len, heads, head_dim)
        k_blocks = k_win.view(B, num_windows, local_len, heads, head_dim)
        v_blocks = v_win.view(B, num_windows, local_len, heads, head_dim)

        # Block/window mean pooled Q/K descriptors.
        # This matches the paper's block-level Q/K pooling idea.
        q_desc = q_blocks.mean(dim=2).float()  # [B, N, heads, D]
        k_desc = k_blocks.mean(dim=2).float()  # [B, N, heads, D]

        # Block-level attention scores, averaged over heads.
        # [B, heads, N_query, N_key] -> [B, N_query, N_key]
        scores = torch.einsum("bihd,bjhd->bhij", q_desc, k_desc)
        scores = scores / math.sqrt(float(head_dim))
        scores = scores.mean(dim=1)

        # Spatial local candidate mask M_local.
        local_mask = self._make_local_candidate_mask(meta, device=q_win.device)
        local_mask = local_mask[None, :, :].expand(B, num_windows, num_windows)

        # Exclude self because the original local shifted window is already kept as K_i/V_i.
        eye = torch.eye(num_windows, device=q_win.device, dtype=torch.bool)
        not_self = (~eye)[None, :, :].expand(B, num_windows, num_windows)

        candidate_mask = local_mask & not_self

        # If a row has no valid non-self candidate, return nothing.
        valid_count = candidate_mask.sum(dim=-1).amax().item()
        if valid_count <= 0:
            return None, None

        topk = min(int(self.dynamic_topk), int(valid_count))
        if topk <= 0:
            return None, None

        scores = scores.masked_fill(~candidate_mask, float("-inf"))

        topk_idx = torch.topk(scores, k=topk, dim=-1).indices
        # topk_idx: [B, num_windows, topk]

        gather_idx = topk_idx[:, :, :, None, None, None].expand(
            B, num_windows, topk, local_len, heads, head_dim
        )

        k_expand = k_blocks[:, None, :, :, :, :].expand(
            B, num_windows, num_windows, local_len, heads, head_dim
        )
        v_expand = v_blocks[:, None, :, :, :, :].expand(
            B, num_windows, num_windows, local_len, heads, head_dim
        )

        k_dyn = torch.gather(k_expand, dim=2, index=gather_idx)
        v_dyn = torch.gather(v_expand, dim=2, index=gather_idx)

        # [B, num_windows, topk, local_len, heads, D]
        # -> [B*num_windows, topk*local_len, heads, D]
        k_dyn = k_dyn.contiguous().view(B * num_windows, topk * local_len, heads, head_dim)
        v_dyn = v_dyn.contiguous().view(B * num_windows, topk * local_len, heads, head_dim)

        return k_dyn.to(dtype=k_win.dtype), v_dyn.to(dtype=v_win.dtype)

    def _window_flex_attention(self, q, k, v):
        """
        q/k/v:
            [B, L, dim]

        Return:
            [B, L, dim]

        Attention contents for each shifted spatial window:
            1) original local shifted-window K/V
            2) dynamic top-k true-window K/V selected by block-level Q/K score
            3) original K/V-only spatial pooling tokens

        No temporal causal mask.
        No competition.
        No distance penalty.
        No extra layer schedule.
        """
        B, L, C = q.shape
        T, H, W = self.spatial_shape
        Wh, Ww = self.window_size

        expected_len = T * H * W
        if L != expected_len:
            import warnings
            warnings.warn(
                f"StudentSelfAttention fallback to full attention because L={L} "
                f"!= expected_len={expected_len} "
                f"for spatial_shape={self.spatial_shape}.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self.attn(q, k, v)

        heads = self.num_heads
        head_dim = self.head_dim

        # [B,L,C] -> [B,T,H,W,heads,D]
        q = q.view(B, T, H, W, heads, head_dim)
        k = k.view(B, T, H, W, heads, head_dim)
        v = v.view(B, T, H, W, heads, head_dim)

        # Original pooling tokens are created from unshifted K/V maps.
        if self.use_pooling_token:
            k_pool = self._make_spatial_pool_tokens(k, self.pool_size)
            v_pool = self._make_spatial_pool_tokens(v, self.pool_size)
        else:
            k_pool = None
            v_pool = None

        # Layer-wise shifted spatial windows. Shift only H/W, never T.
        shift_h, shift_w = self._get_layer_shift()
        if shift_h != 0 or shift_w != 0:
            q = torch.roll(q, shifts=(-shift_h, -shift_w), dims=(2, 3))
            k = torch.roll(k, shifts=(-shift_h, -shift_w), dims=(2, 3))
            v = torch.roll(v, shifts=(-shift_h, -shift_w), dims=(2, 3))

        # Partition into spatial windows.
        # q_win/k_win/v_win:
        #   [B * num_spatial_windows, T * Wh * Ww, heads, D]
        q_win, meta = self._partition_spatial_windows(q, (Wh, Ww))
        k_win, _ = self._partition_spatial_windows(k, (Wh, Ww))
        v_win, _ = self._partition_spatial_windows(v, (Wh, Ww))

        # Ultra-Flash-style dynamic true-window selection.
        # Selected true K/V windows are appended to K/V side.
        k_dyn, v_dyn = self._select_dynamic_true_windows(q_win, k_win, v_win, meta)
        if k_dyn is not None:
            k_win = torch.cat([k_win, k_dyn], dim=1)
            v_win = torch.cat([v_win, v_dyn], dim=1)

        # Original pooled K/V memory is appended to every spatial window.
        # Q branch is unchanged, so no extra output tokens are created.
        if self.use_pooling_token and k_pool is not None and k_pool.shape[1] > 0:
            nH = meta["nH"]
            nW = meta["nW"]
            num_spatial_windows = nH * nW
            pool_len = k_pool.shape[1]

            k_pool_win = (
                k_pool[:, None, :, :, :]
                .expand(B, num_spatial_windows, pool_len, heads, head_dim)
                .reshape(B * num_spatial_windows, pool_len, heads, head_dim)
                .contiguous()
            )
            v_pool_win = (
                v_pool[:, None, :, :, :]
                .expand(B, num_spatial_windows, pool_len, heads, head_dim)
                .reshape(B * num_spatial_windows, pool_len, heads, head_dim)
                .contiguous()
            )

            k_win = torch.cat([k_win, k_pool_win], dim=1)
            v_win = torch.cat([v_win, v_pool_win], dim=1)

        # Use the project-level FlashAttention/SageAttention/SDPA backend for
        # the same dense attention inside each constructed K/V window.
        # q_win/k_win/v_win: [Bwin, S, heads, head_dim]
        Bwin = q_win.shape[0]
        Sq = q_win.shape[1]
        Sk = k_win.shape[1]

        q_win = q_win.reshape(Bwin, Sq, heads * head_dim).contiguous()
        k_win = k_win.reshape(Bwin, Sk, heads * head_dim).contiguous()
        v_win = v_win.reshape(Bwin, Sk, heads * head_dim).contiguous()

        out_win = flash_attention(
            q=q_win,
            k=k_win,
            v=v_win,
            num_heads=heads,
        )

        out_win = out_win.view(Bwin, Sq, heads, head_dim).contiguous()

        # Reverse spatial windows.
        out = self._reverse_spatial_windows(out_win, meta)

        # Reverse spatial shift.
        if shift_h != 0 or shift_w != 0:
            out = torch.roll(out, shifts=(shift_h, shift_w), dims=(2, 3))

        # [B,T,H,W,heads,D] -> [B,L,C]
        out = out.reshape(B, L, C).contiguous()
        return out

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)

        # Keep original Wan RoPE behavior.
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)

        # Original shifted-window + pooling,
        # plus non-causal Ultra-Flash-style dynamic true-window top-k routing.
        x = self._window_flex_attention(q, k, v)

        return self.o(x)


def convert_wan_self_attention_to_basa(
    model: nn.Module,
    *,
    window_size: Tuple[int, int] = (15, 26),
    pool_size: Tuple[int, int] = (4, 4),
    dynamic_topk: int = 1,
    local_window_radius: int = 1,
    use_pooling_token: bool = True,
    use_dynamic_true_window: bool = True,
):
    """Replace Wan DiT self-attention modules by inference-only BASA sparse attention.

    The projection/norm parameter names are intentionally identical to SelfAttention,
    so a distilled checkpoint can be loaded as usual and then converted without a
    teacher model or deepcopy.
    """
    if not hasattr(model, "blocks"):
        return model

    num_layers = len(model.blocks)
    for block_index, block in enumerate(model.blocks):
        old_attn = getattr(block, "self_attn", None)
        if old_attn is None or isinstance(old_attn, BasaSparseSelfAttention):
            continue

        eps = getattr(getattr(old_attn, "norm_q", None), "eps", 1e-6)
        new_attn = BasaSparseSelfAttention(
            dim=old_attn.dim,
            num_heads=old_attn.num_heads,
            block_index=block_index,
            eps=eps,
            window_size=window_size,
            pool_size=pool_size,
            dynamic_topk=dynamic_topk,
            local_window_radius=local_window_radius,
            use_pooling_token=use_pooling_token,
            use_dynamic_true_window=use_dynamic_true_window,
            num_layers=num_layers,
        )
        new_attn.load_state_dict(old_attn.state_dict(), strict=True)
        ref_param = next(old_attn.parameters(), None)
        if ref_param is not None:
            new_attn.to(device=ref_param.device, dtype=ref_param.dtype)
        block.self_attn = new_attn

    model.use_basa_sparse_attention = True
    return model


def set_basa_sparse_attention_shape(model: nn.Module, f: int, h: int, w: int):
    """Update the runtime latent grid shape used by BASA sparse attention."""
    if not getattr(model, "use_basa_sparse_attention", False):
        return
    spatial_shape = (int(f), int(h), int(w))
    for block in getattr(model, "blocks", []):
        attn = getattr(block, "self_attn", None)
        if isinstance(attn, BasaSparseSelfAttention):
            attn.spatial_shape = spatial_shape


def set_basa_denoising_step(model: nn.Module, denoising_step: int):
    """Set the 1-based denoising step for all BASA self-attention blocks."""
    step = int(denoising_step)
    if step < 1:
        raise ValueError(f"denoising_step must be >= 1, got {step}")

    for block in getattr(model, "blocks", []):
        attn = getattr(block, "self_attn", None)
        if isinstance(attn, BasaSparseSelfAttention):
            attn.denoising_step = step


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


def wantodance_torch_dfs(model: nn.Module, parent_name='root'):
    module_names, modules = [], []
    current_name = parent_name if parent_name else 'root'
    module_names.append(current_name)
    modules.append(model)
    for name, child in model.named_children():
        if parent_name:
            child_name = f'{parent_name}.{name}'
        else:
            child_name = name
        child_modules, child_names = wantodance_torch_dfs(child, child_name)
        module_names += child_names
        modules += child_modules
    return modules, module_names


class WanToDanceInjector(nn.Module):
    def __init__(self, all_modules, all_modules_names, dim=2048, num_heads=32, inject_layer=[0, 27]):
        super().__init__()
        self.injected_block_id = {}
        injector_id = 0
        for mod_name, mod in zip(all_modules_names, all_modules):
            if isinstance(mod, DiTBlock):
                for inject_id in inject_layer:
                    if f'root.transformer_blocks.{inject_id}' == mod_name:
                        self.injected_block_id[inject_id] = injector_id
                        injector_id += 1

        self.injector = nn.ModuleList(
            [
                CrossAttention(
                    dim=dim,
                    num_heads=num_heads,
                )
                for _ in range(injector_id)
            ]
        )
        self.injector_pre_norm_feat = nn.ModuleList(
            [
                nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6,)
                for _ in range(injector_id)
            ]
        )
        self.injector_pre_norm_vec = nn.ModuleList(
            [
                nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6,)
                for _ in range(injector_id)
            ]
        )


class WanModel(torch.nn.Module):

    _repeated_blocks = ["DiTBlock"]

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        wantodance_enable_music_inject: bool = False,
        wantodance_music_inject_layers = [0, 4, 8, 12, 16, 20, 24, 27],
        wantodance_enable_refimage: bool = False,
        wantodance_enable_refface: bool = False,
        wantodance_enable_global: bool = False,
        wantodance_enable_dynamicfps: bool = False,
        wantodance_enable_unimodel: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads

        if wantodance_enable_dynamicfps or wantodance_enable_unimodel:
            end = int(22350 / 8 + 0.5) # 149f * 30fps * 5s = 22350
            self.freqs = precompute_freqs_cis_3d(head_dim, end=end)
        else:
            self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

        self.prepare_wantodance(in_dim, dim, num_heads, has_image_pos_emb, out_dim, patch_size, eps,
                                wantodance_enable_music_inject, wantodance_music_inject_layers, wantodance_enable_refimage, wantodance_enable_refface,
                                wantodance_enable_global, wantodance_enable_dynamicfps, wantodance_enable_unimodel)

    def prepare_wantodance(
        self,
        in_dim, dim, num_heads, has_image_pos_emb, out_dim, patch_size, eps,
        wantodance_enable_music_inject: bool = False,
        wantodance_music_inject_layers = [0, 4, 8, 12, 16, 20, 24, 27],
        wantodance_enable_refimage: bool = False,
        wantodance_enable_refface: bool = False,
        wantodance_enable_global: bool = False,
        wantodance_enable_dynamicfps: bool = False,
        wantodance_enable_unimodel: bool = False,
    ):
        if wantodance_enable_music_inject:
            all_modules, all_modules_names = wantodance_torch_dfs(self.blocks, parent_name="root.transformer_blocks")
            self.music_injector = WanToDanceInjector(all_modules, all_modules_names, dim=dim, num_heads=num_heads, inject_layer=wantodance_music_inject_layers)
        if wantodance_enable_refimage:
            self.img_emb_refimage = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if wantodance_enable_refface:
            self.img_emb_refface = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if wantodance_enable_global or wantodance_enable_dynamicfps or wantodance_enable_unimodel:
            music_feature_dim = 35
            ff_size = 1024
            dropout = 0.1
            latent_dim = 256
            nhead = 4
            activation = F.gelu
            rotary = WanToDanceRotaryEmbedding(dim=latent_dim)
            self.music_projection = nn.Linear(music_feature_dim, latent_dim)
            self.music_encoder = nn.Sequential()
            for _ in range(2):
                self.music_encoder.append(
                    WanToDanceMusicEncoderLayer(
                        d_model=latent_dim,
                        nhead=nhead,
                        dim_feedforward=ff_size,
                        dropout=dropout,
                        activation=activation,
                        batch_first=True,
                        rotary=rotary,
                        device='cuda',
                    )
                )
        if wantodance_enable_unimodel:
            self.patch_embedding_global = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        if wantodance_enable_unimodel:
            self.head_global = Head(dim, out_dim, patch_size, eps)
        self.wantodance_enable_music_inject = wantodance_enable_music_inject
        self.wantodance_enable_refimage = wantodance_enable_refimage
        self.wantodance_enable_refface = wantodance_enable_refface
        self.wantodance_enable_global = wantodance_enable_global
        self.wantodance_enable_dynamicfps = wantodance_enable_dynamicfps
        self.wantodance_enable_unimodel = wantodance_enable_unimodel

    def wantodance_after_transformer_block(self, block_idx, hidden_states):
        if self.wantodance_enable_music_inject:
            if block_idx in self.music_injector.injected_block_id.keys():
                audio_attn_id = self.music_injector.injected_block_id[block_idx]
                audio_emb = self.merged_audio_emb  # b f n c
                num_frames = audio_emb.shape[1]
                input_hidden_states = hidden_states.clone()  # b (f h w) c
                input_hidden_states = rearrange(input_hidden_states, "b (t n) c -> (b t) n c", t=num_frames)
                attn_hidden_states = self.music_injector.injector_pre_norm_feat[audio_attn_id](input_hidden_states)
                audio_emb = rearrange(audio_emb, "b t c -> (b t) 1 c", t=num_frames)
                attn_audio_emb = audio_emb
                residual_out = self.music_injector.injector[audio_attn_id](attn_hidden_states, attn_audio_emb)
                residual_out = rearrange(residual_out, "(b t) n c -> b (t n) c", t=num_frames)
                hidden_states = hidden_states + residual_out
        return hidden_states

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None, enable_wantodance_global=False):
        if enable_wantodance_global:
            x = self.patch_embedding_global(x)
        else:
            x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        x, (f, h, w) = self.patchify(x)
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        set_basa_sparse_attention_shape(self, f, h, w)

        for block in self.blocks:
            if self.training:
                x = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing,
                    use_gradient_checkpointing_offload,
                    x, context, t_mod, freqs
                )
            else:
                x = block(x, context, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x
