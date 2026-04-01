"""
Universal Transformer with Random Matrix Adapters
==================================================

Novel architecture for OpenAI Parameter Golf challenge.

Core idea: Instead of storing N unique transformer layer weights, we:
1. Generate deterministic random matrices from a seed (free storage)
2. Store small learned low-rank adapters (A @ B) that correct each projection
3. Apply the same adapted block R times (Universal Transformer)
4. Use K distinct adapter sets cycled across R steps for expressiveness

The effective weight for each projection is: W_eff = W_random + A @ B
where W_random is regenerated from seed at eval time (0 bytes stored).

Parameter savings enable wider models and/or more recurrence depth
within the 16MB artifact budget.
"""

import hashlib
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Flash Attention: prefer FA3 (Hopper), fall back to FA2
try:
    from flash_attn_interface import flash_attn_func as _flash_attn_fn
except ImportError:
    from flash_attn import flash_attn_func as _flash_attn_fn

# Components reused from the SOTA train_gpt.py
from train_gpt import BigramHashEmbedding, SmearGate, ValueEmbedding, Rotary


# =============================================================================
# Fake Quantization for QAT (Straight-Through Estimator)
# =============================================================================

def fake_quantize_int8_per_row(t: Tensor) -> Tensor:
    """Simulate int8 per-row quantization with straight-through estimator.
    
    During training, this rounds weights to the nearest int8-representable value
    (given per-row scales) but lets gradients flow through as if no rounding happened.
    This teaches the optimizer to find weight distributions that survive int8 truncation.
    """
    with torch.no_grad():
        t32 = t.float()
        row_max = t32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = row_max / 127.0
        t_q = (torch.clamp(torch.round(t32 / scale), -127, 127) * scale).to(t.dtype)
    # STE: forward uses quantized, backward uses identity
    return t + (t_q - t).detach()


# =============================================================================
# Random Matrix Generation (deterministic, seed-reproducible)
# =============================================================================

class RandomMatrixGenerator:
    """Generates deterministic random matrices from a seed.
    
    Uses a separate RNG stream so random matrices are identical across
    training and evaluation, regardless of other randomness in the program.
    
    The matrices are orthogonally initialized and scaled, following
    insights from random matrix theory that orthogonal random projections
    preserve geometric structure better than Gaussian random matrices.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self._cache: dict[str, Tensor] = {}
    
    def get_matrix(self, name: str, rows: int, cols: int, 
                   device: torch.device, dtype: torch.dtype = torch.bfloat16) -> Tensor:
        """Get a deterministic random matrix, cached after first generation.
        
        Uses orthogonal initialization scaled by sqrt(max(rows,cols)/min(rows,cols))
        to preserve norms under projection, following Saxe et al. (2013).
        """
        cache_key = f"{name}_{rows}_{cols}"
        if cache_key not in self._cache or self._cache[cache_key].device != device:
            # Derive a unique seed per matrix from stable hash + base seed
            # CRITICAL: Python's hash() is salted per-process in Python 3.3+,
            # which would give different matrices on different DDP ranks.
            name_hash = int(hashlib.md5(name.encode('utf-8')).hexdigest()[:8], 16)
            combined_seed = self.seed ^ name_hash
            
            # Save and restore global RNG state so random matrix generation
            # is fully deterministic regardless of when this is called.
            # nn.init.orthogonal_ uses the global RNG, not an explicit generator,
            # so we must control the global state directly.
            global_rng_state = torch.random.get_rng_state()
            torch.manual_seed(combined_seed)
            
            mat = torch.empty(rows, cols)
            nn.init.orthogonal_(mat)
            
            # Restore global RNG state so we don't affect other randomness
            torch.random.set_rng_state(global_rng_state)
            
            # Scale so that ||Wx|| ≈ ||x|| in expectation
            # This is the "gain" that preserves signal magnitude
            scale = math.sqrt(max(rows, cols) / min(rows, cols))
            mat.mul_(scale)
            
            self._cache[cache_key] = mat.to(device=device, dtype=dtype)
        
        return self._cache[cache_key]
    
    def clear_cache(self):
        """Free cached matrices (e.g., after moving to different device)."""
        self._cache.clear()


# =============================================================================
# Low-Rank Adapter
# =============================================================================

class LowRankAdapter(nn.Module):
    """Learned low-rank correction: output = W_random @ x + A @ B @ x
    
    The adapter matrices A and B are initialized so that A @ B ≈ 0 at init,
    meaning the model starts as a pure random projection and learns corrections.
    
    Supports Late QAT: when qat_enabled is True, the fused ΔW = scale*(A@B) is
    fake-quantized to int8 before being added to W_random, teaching AdamW to
    find int8-friendly weight distributions.
    
    Args:
        in_features: input dimension
        out_features: output dimension  
        rank: rank of the low-rank correction
        alpha: scaling factor for the adapter (like LoRA alpha)
        init_scale: std of initialization for B (A is zero-init)
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int,
                 alpha: float = 1.0, init_scale: float = 0.01):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank  # LoRA-style scaling
        self.qat_enabled = False  # Toggled externally during late QAT phase
        
        # A: (out_features, rank) - zero initialized
        # B: (rank, in_features) - small random init
        # So A @ B starts at zero -> model starts as pure random projection
        self.A = nn.Parameter(torch.zeros(out_features, rank))
        self.B = nn.Parameter(torch.empty(rank, in_features))
        nn.init.normal_(self.B, std=init_scale)
        
        # Fused delta_w buffer: set by load_fused_state_dict() for eval.
        # When not None, forward() uses this instead of computing A @ B.
        self._fused_delta_w: Tensor | None = None
    
    def compute_delta_w(self, dtype: torch.dtype = None) -> Tensor:
        """Compute the fused ΔW = scale * (A @ B).
        
        Used both in forward() and for post-training export.
        """
        d = dtype or self.A.dtype
        return self.scale * (self.A.to(d) @ self.B.to(d))
    
    def forward(self, x: Tensor, W_random: Tensor) -> Tensor:
        """Apply random projection + learned correction via single fused GEMM.
        
        Materializes W_eff = W_random + ΔW then does one F.linear.
        
        Three modes:
        1. Normal training: ΔW = scale * (A @ B)
        2. Late QAT training: ΔW = fake_quantize_int8(scale * (A @ B))
        3. Fused eval: ΔW = scale * _fused_delta_w (unscaled A@B from export)
        """
        # Check for fused delta_w (loaded from fused export — stored unscaled)
        if self._fused_delta_w is not None:
            delta_w = self.scale * self._fused_delta_w.to(device=x.device, dtype=x.dtype)
        else:
            delta_w = self.compute_delta_w(x.dtype)
            if self.qat_enabled and self.training:
                delta_w = fake_quantize_int8_per_row(delta_w)
        
        W_eff = W_random.to(x.dtype) + delta_w
        return F.linear(x, W_eff)


# =============================================================================
# Adapted Attention (uses random matrices + adapters for Q,K,V,O projections)
# =============================================================================

class AdaptedAttention(nn.Module):
    """Self-attention where Q, K, V, O projections use random matrices + adapters.
    
    Compatible with the existing SOTA stack's attention features:
    - GQA (grouped query attention)
    - Partial RoPE
    - XSA (cross-sequence attention)
    - QK normalization + gain
    """
    
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rank: int,
                 qk_gain_init: float = 1.5, adapter_alpha: float = 1.0):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        
        # Adapters for each projection (random matrices provided externally)
        self.q_adapter = LowRankAdapter(dim, dim, rank, alpha=adapter_alpha)
        self.k_adapter = LowRankAdapter(dim, kv_dim, rank, alpha=adapter_alpha)
        self.v_adapter = LowRankAdapter(dim, kv_dim, rank, alpha=adapter_alpha)
        self.o_adapter = LowRankAdapter(dim, dim, rank, alpha=adapter_alpha)
        
        # Non-adapted small params (same as SOTA)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        
        # XSA and RoPE settings (configured externally)
        self.use_xsa = False
        self.rope_dims = 0
    
    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        """Efficient XSA: subtract self-value projection via GQA-aware reshape."""
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)
    
    def forward(self, x: Tensor, 
                W_q: Tensor, W_k: Tensor, W_v: Tensor, W_o: Tensor,
                cos: Tensor, sin: Tensor,
                flash_attn_fn,
                v_embed: Tensor | None = None) -> Tensor:
        bsz, seqlen, dim = x.shape
        
        q = self.q_adapter(x, W_q).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.k_adapter(x, W_k).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.v_adapter(x, W_v)
        
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        
        # QK normalization
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        
        # Partial RoPE
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        
        # QK gain
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        
        # Flash attention
        y = flash_attn_fn(q, k, v, causal=True)
        
        # XSA
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        
        y = y.reshape(bsz, seqlen, dim)
        return self.o_adapter(y, W_o)


# =============================================================================
# Adapted MLP (uses random matrices + adapters for up/down projections)
# =============================================================================

class AdaptedMLP(nn.Module):
    """MLP with random matrices + adapters. Uses LeakyReLU(0.5)² activation."""
    
    def __init__(self, dim: int, mlp_dim: int, rank: int, adapter_alpha: float = 1.0):
        super().__init__()
        self.up_adapter = LowRankAdapter(dim, mlp_dim, rank, alpha=adapter_alpha)
        self.down_adapter = LowRankAdapter(mlp_dim, dim, rank, alpha=adapter_alpha)
    
    def forward(self, x: Tensor, W_up: Tensor, W_down: Tensor) -> Tensor:
        h = F.leaky_relu(self.up_adapter(x, W_up), negative_slope=0.5)
        return self.down_adapter(h.square(), W_down)


# =============================================================================
# Universal Transformer Block
# =============================================================================

class UTBlock(nn.Module):
    """A single Universal Transformer block with adapter-corrected random projections.
    
    This block is applied R times with different adapter sets (selected by step index).
    The random matrices are shared across all applications.
    
    Includes all SOTA features: RMSNorm, residual scaling, residual mixing,
    LN scale factor, SmearGate compatibility.
    """
    
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 mlp_mult: float, rank: int, num_adapter_sets: int = 1,
                 qk_gain_init: float = 1.5, adapter_alpha: float = 1.0,
                 ln_scale: bool = True, num_recurrence_steps: int = 11,
                 adapter_strategy: str = "phase"):
        super().__init__()
        self.dim = dim
        self.num_adapter_sets = num_adapter_sets
        self.num_recurrence_steps = num_recurrence_steps
        mlp_dim = int(mlp_mult * dim)
        
        # Build the adapter-to-step mapping
        # "modulo": step i uses adapter i % K (interleaved cycling)
        # "phase": steps are divided into K contiguous phases
        #          e.g. K=3, R=11 -> adapter 0 for steps 0-3, 1 for 4-7, 2 for 8-10
        self.adapter_strategy = adapter_strategy
        if adapter_strategy == "phase":
            steps_per_phase = num_recurrence_steps / num_adapter_sets
            self._step_to_adapter = [
                min(int(i / steps_per_phase), num_adapter_sets - 1)
                for i in range(num_recurrence_steps)
            ]
        else:  # modulo
            self._step_to_adapter = [i % num_adapter_sets for i in range(num_recurrence_steps)]
        
        # K sets of adapters (attention + MLP each)
        self.attn_sets = nn.ModuleList([
            AdaptedAttention(dim, num_heads, num_kv_heads, rank,
                           qk_gain_init=qk_gain_init, adapter_alpha=adapter_alpha)
            for _ in range(num_adapter_sets)
        ])
        self.mlp_sets = nn.ModuleList([
            AdaptedMLP(dim, mlp_dim, rank, adapter_alpha=adapter_alpha)
            for _ in range(num_adapter_sets)
        ])
        
        # Per-recurrence-step parameters (cheap, learned)
        # These allow each application of the block to behave differently
        # even when sharing the same adapter set
        self.step_scale = nn.Parameter(torch.ones(num_recurrence_steps, dim, dtype=torch.float32))
        self.step_shift = nn.Parameter(torch.zeros(num_recurrence_steps, dim, dtype=torch.float32))
        
        # Shared norms and scales (same as SOTA but single set)
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        
        # LN scale factors (per recurrence step, like SOTA's per-layer 1/sqrt(i+1))
        if ln_scale:
            self.ln_scale_factors = [1.0 / math.sqrt(i + 1) for i in range(num_recurrence_steps)]
        else:
            self.ln_scale_factors = [1.0] * num_recurrence_steps
    
    def forward(self, x: Tensor, x0: Tensor, step_idx: int,
                W_q: Tensor, W_k: Tensor, W_v: Tensor, W_o: Tensor,
                W_up: Tensor, W_down: Tensor,
                cos: Tensor, sin: Tensor,
                flash_attn_fn,
                v_embed: Tensor | None = None) -> Tensor:
        """
        Args:
            x: current hidden state (bsz, seq_len, dim)
            x0: original input (for residual mixing)
            step_idx: which recurrence step (0 to R-1)
            W_q, W_k, W_v, W_o: random matrices for attention
            W_up, W_down: random matrices for MLP
            cos, sin: RoPE embeddings
            flash_attn_fn: flash attention function
            v_embed: optional value embedding
        """
        # Select adapter set for this step
        adapter_idx = self._step_to_adapter[step_idx]
        attn = self.attn_sets[adapter_idx]
        mlp = self.mlp_sets[adapter_idx]
        
        ln_factor = self.ln_scale_factors[step_idx]
        
        # Per-step modulation
        s = self.step_scale[step_idx].to(dtype=x.dtype)[None, None, :]
        b = self.step_shift[step_idx].to(dtype=x.dtype)[None, None, :]
        
        # Residual mixing (blend current state with original input)
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        
        # Per-step modulation applied after mixing
        x_in = x_in * s + b
        
        # Attention
        attn_out = attn(self.attn_norm(x_in) * ln_factor,
                       W_q, W_k, W_v, W_o, cos, sin, flash_attn_fn,
                       v_embed=v_embed)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        
        # MLP
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * mlp(
            self.mlp_norm(x_out) * ln_factor, W_up, W_down)
        
        return x_out


# =============================================================================
# Helper: RoPE (copied from SOTA for self-containment)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


# =============================================================================
# Full Model: Universal Transformer GPT with Random Matrix Adapters
# =============================================================================

class UTGPT(nn.Module):
    """Universal Transformer GPT with Random Matrix Adapters.
    
    Architecture:
    - Token embedding (learned, tied with output)
    - BigramHash embedding (learned)
    - SmearGate
    - R applications of the shared UTBlock (with K cycling adapter sets)
    - U-Net style skip connections between first half and second half
    - RMSNorm + tied linear head
    
    Random matrices are generated deterministically from a seed and cached.
    Only adapter parameters + embeddings + small control tensors are stored.
    """
    
    def __init__(
        self,
        vocab_size: int = 1024,
        model_dim: int = 768,
        num_heads: int = 12,
        num_kv_heads: int = 6,
        mlp_mult: float = 3.0,
        num_recurrence_steps: int = 11,
        num_adapter_sets: int = 3,
        adapter_rank: int = 256,
        adapter_alpha: float = 1.0,
        tie_embeddings: bool = True,
        tied_embed_init_std: float = 0.005,
        logit_softcap: float = 30.0,
        rope_base: float = 10000.0,
        rope_dims: int = 16,
        qk_gain_init: float = 1.5,
        bigram_vocab_size: int = 3072,
        bigram_dim: int = 112,
        xsa_last_n: int = 11,
        ln_scale: bool = True,
        ve_enabled: bool = True,
        ve_dim: int = 128,
        ve_layers: str = "9,10",
        random_seed: int = 42,
        adapter_strategy: str = "phase",
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_recurrence_steps = num_recurrence_steps
        self.num_adapter_sets = num_adapter_sets
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        
        head_dim = model_dim // num_heads
        kv_dim = num_kv_heads * head_dim
        mlp_dim = int(mlp_mult * model_dim)
        
        # Token embedding (learned, stored)
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        if tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)
        
        # MTP heads placeholder (empty — satisfies training loop scaffolding)
        self.mtp_heads = nn.ModuleList()
        
        # BigramHash embedding (from SOTA)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim)
        
        # Universal Transformer block (shared, with K adapter sets)
        self.ut_block = UTBlock(
            dim=model_dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
            mlp_mult=mlp_mult, rank=adapter_rank, num_adapter_sets=num_adapter_sets,
            qk_gain_init=qk_gain_init, adapter_alpha=adapter_alpha,
            ln_scale=ln_scale, num_recurrence_steps=num_recurrence_steps,
            adapter_strategy=adapter_strategy,
        )
        
        # Configure XSA on deep layers
        if xsa_last_n > 0:
            for attn in self.ut_block.attn_sets:
                attn.use_xsa = True  # We'll selectively apply per step in forward
        self._xsa_start_step = max(0, num_recurrence_steps - xsa_last_n)
        
        # Configure partial RoPE
        for attn in self.ut_block.attn_sets:
            attn.rope_dims = rope_dims
        
        # RoPE module
        self.rotary = Rotary(head_dim, base=rope_base, train_seq_len=2048, rope_dims=rope_dims)
        
        # U-Net skip connections
        self.num_encoder_steps = num_recurrence_steps // 2
        self.num_decoder_steps = num_recurrence_steps - self.num_encoder_steps
        self.num_skip_weights = min(self.num_encoder_steps, self.num_decoder_steps)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        
        # Value embeddings (from SOTA)
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_enabled else []
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim)
            self.ve_layer_scales = nn.ParameterList(
                [nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices]
            )
        else:
            self.ve_shared = None
            self.ve_layer_scales = nn.ParameterList()
        
        # Output
        self.final_norm = RMSNorm()
        
        # Random matrix generator
        self.rng = RandomMatrixGenerator(seed=random_seed)
        _random_matrix_specs = {
            'W_q': (model_dim, model_dim),
            'W_k': (kv_dim, model_dim),
            'W_v': (kv_dim, model_dim),
            'W_o': (model_dim, model_dim),
            'W_up': (mlp_dim, model_dim),
            'W_down': (model_dim, mlp_dim),
        }
        # Generate and register random matrices as non-persistent buffers.
        # Using register_buffer ensures model.to(device) moves them automatically,
        # and Dynamo treats them as graph inputs without recompilation.
        self._random_matrix_names = []
        for name, (rows, cols) in _random_matrix_specs.items():
            mat = self.rng.get_matrix(name, rows, cols, device=torch.device('cpu'), dtype=torch.bfloat16)
            self.register_buffer(f'_rm_{name}', mat, persistent=False)
            self._random_matrix_names.append(name)
    
    def _get_random_matrices(self) -> dict[str, Tensor]:
        """Get all random matrices as a dict. Buffers are already on the correct device."""
        return {name: getattr(self, f'_rm_{name}') for name in self._random_matrix_names}
    
    def set_qat(self, enabled: bool) -> None:
        """Toggle Late QAT on all LowRankAdapter modules.
        
        When enabled, each adapter's forward() fake-quantizes ΔW = scale*(A@B)
        to int8 before adding to W_random, teaching the optimizer to find
        weight distributions that survive int8 truncation.
        """
        for module in self.modules():
            if isinstance(module, LowRankAdapter):
                module.qat_enabled = enabled
    
    @torch.no_grad()
    def fused_export_state_dict(self) -> dict[str, Tensor]:
        """Export state dict with fused ΔW matrices for int8-friendly storage.
        
        Exports unscaled A @ B (without the LoRA scale factor) so values are
        in a reasonable numeric range for int8 quantization. The scale factor
        is stored separately as a small scalar tensor per adapter.
        
        At eval time: ΔW = scale * dequant(A@B_int8), then W_eff = W_random + ΔW.
        """
        fused_sd = {}
        
        for name, param in self.state_dict().items():
            # Skip A and B params — we'll replace them with fused delta_w
            if '.A' in name or '.B' in name:
                continue
            fused_sd[name] = param.detach().cpu()
        
        # Fuse each adapter's A @ B into a dense matrix (WITHOUT scale)
        for module_name, module in self.named_modules():
            if isinstance(module, LowRankAdapter):
                # Store unscaled A @ B for better int8 dynamic range
                ab = (module.A.to(torch.bfloat16) @ module.B.to(torch.bfloat16))
                fused_sd[f"{module_name}.delta_w"] = ab.detach().cpu()
                # Store scale as a tiny scalar tensor (passthrough in int8 quantizer)
                fused_sd[f"{module_name}.delta_scale"] = torch.tensor(
                    module.scale, dtype=torch.float32)
        
        return fused_sd
    
    def _get_ve(self, step_idx: int, input_ids: Tensor, ve_cache: dict) -> Tensor | None:
        if self.ve_shared is None or step_idx not in self.ve_layer_indices:
            return None
        if 've' not in ve_cache:
            ve_cache['ve'] = self.ve_shared(input_ids)
        ve_idx = self.ve_layer_indices.index(step_idx)
        return ve_cache['ve'] * self.ve_layer_scales[ve_idx].to(dtype=ve_cache['ve'].dtype)
    
    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        bsz, seqlen = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x  # save for residual mixing
        
        # Get random matrices (pre-generated buffers)
        rm = self._get_random_matrices()
        
        # RoPE
        cos, sin = self.rotary(seqlen, device, x.dtype)
        
        # U-Net: encoder phase
        skips: list[Tensor] = []
        ve_cache: dict = {}
        
        for step in range(self.num_encoder_steps):
            # Toggle XSA based on step
            for attn in self.ut_block.attn_sets:
                attn.use_xsa = (step >= self._xsa_start_step)
            
            ve = self._get_ve(step, input_ids, ve_cache)
            x = self.ut_block(x, x0, step,
                            rm['W_q'], rm['W_k'], rm['W_v'], rm['W_o'],
                            rm['W_up'], rm['W_down'],
                            cos, sin, _flash_attn_fn, v_embed=ve)
            skips.append(x)
        
        # U-Net: decoder phase
        for i in range(self.num_decoder_steps):
            step = self.num_encoder_steps + i
            
            # Skip connection
            if skips:
                skip = skips.pop()
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skip
            
            for attn in self.ut_block.attn_sets:
                attn.use_xsa = (step >= self._xsa_start_step)
            
            ve = self._get_ve(step, input_ids, ve_cache)
            x = self.ut_block(x, x0, step,
                            rm['W_q'], rm['W_k'], rm['W_v'], rm['W_o'],
                            rm['W_up'], rm['W_down'],
                            cos, sin, _flash_attn_fn, v_embed=ve)
        
        # Output
        x = self.final_norm(x)
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        
        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            raise NotImplementedError("Non-tied embeddings not implemented yet")
        
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")
    
    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return logits without computing loss (for eval and GPTQ calibration)."""
        bsz, seqlen = input_ids.shape
        device = input_ids.device
        
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        
        rm = self._get_random_matrices()
        cos, sin = self.rotary(seqlen, device, x.dtype)
        
        skips: list[Tensor] = []
        ve_cache: dict = {}
        
        for step in range(self.num_encoder_steps):
            for attn in self.ut_block.attn_sets:
                attn.use_xsa = (step >= self._xsa_start_step)
            ve = self._get_ve(step, input_ids, ve_cache)
            x = self.ut_block(x, x0, step,
                            rm['W_q'], rm['W_k'], rm['W_v'], rm['W_o'],
                            rm['W_up'], rm['W_down'],
                            cos, sin, _flash_attn_fn, v_embed=ve)
            skips.append(x)
        
        for i in range(self.num_decoder_steps):
            step = self.num_encoder_steps + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            for attn in self.ut_block.attn_sets:
                attn.use_xsa = (step >= self._xsa_start_step)
            ve = self._get_ve(step, input_ids, ve_cache)
            x = self.ut_block(x, x0, step,
                            rm['W_q'], rm['W_k'], rm['W_v'], rm['W_o'],
                            rm['W_up'], rm['W_down'],
                            cos, sin, _flash_attn_fn, v_embed=ve)
        
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            raise NotImplementedError
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
    
    def load_fused_state_dict(self, fused_sd: dict[str, Tensor]) -> None:
        """Load a state dict produced by fused_export_state_dict().
        
        The fused state dict contains dense 'delta_w' tensors (unscaled A@B)
        and 'delta_scale' scalars instead of separate A and B parameters.
        We register delta_w as a buffer; forward() applies: ΔW = scale * delta_w.
        """
        current_sd = self.state_dict()
        load_sd = {}
        for key in current_sd:
            if '.A' in key or '.B' in key:
                load_sd[key] = current_sd[key]
            elif key in fused_sd:
                load_sd[key] = fused_sd[key]
            else:
                load_sd[key] = current_sd[key]
        
        self.load_state_dict(load_sd, strict=True)
        
        # Register fused delta_w buffers and restore scales
        for module_name, module in self.named_modules():
            if isinstance(module, LowRankAdapter):
                dw_key = f"{module_name}.delta_w"
                scale_key = f"{module_name}.delta_scale"
                if dw_key in fused_sd:
                    if hasattr(module, '_fused_delta_w'):
                        delattr(module, '_fused_delta_w')
                    module.register_buffer('_fused_delta_w', fused_sd[dw_key].clone())
                    if scale_key in fused_sd:
                        module.scale = float(fused_sd[scale_key].item())