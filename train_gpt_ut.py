"""
train_gpt_ut.py — Universal Transformer with Random Matrix Adapters
====================================================================

Training script for the Parameter Golf challenge.
Uses the SOTA infrastructure (data loading, eval, quantization) from train_gpt.py
but replaces the model architecture with UTGPT and the optimizer with pure AdamW.

Usage (single GPU, local development):
    RUN_ID=ut_smoke ITERATIONS=200 MAX_WALLCLOCK_SECONDS=0 VAL_LOSS_EVERY=0 \
    python3 train_gpt_ut.py

Usage (8xH100, submission):
    RUN_ID=ut_submit SEED=314 \
    torchrun --standalone --nproc_per_node=8 train_gpt_ut.py
"""

from __future__ import annotations
import contextlib
import copy
import io
import lzma
import math
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path
from collections import deque

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

# === Import infrastructure from the leader's train_gpt.py ===
from train_gpt import (
    # Data
    Hyperparameters,
    TokenStream,
    DistributedTokenLoader,
    load_data_shard,
    load_validation_tokens,
    # Eval
    build_sentencepiece_luts,
    eval_val,
    eval_val_sliding,
    # Quantization (int8 path — we may not need GPTQ for small adapters)
    quantize_state_dict_int8,
    dequantize_state_dict_int8,
    tensor_nbytes,
    CONTROL_TENSOR_NAME_PATTERNS,
    INT8_KEEP_FLOAT_FP32_NAME_PATTERNS,
    # Model components we reuse
    BigramHashEmbedding,
    SmearGate,
    ValueEmbedding,
    Rotary,
    RMSNorm,
    CastedLinear,
    restore_low_dim_params_to_fp32,
)

# === Import our novel architecture ===
from ut_random_adapter_modules import (
    UTGPT,
    RandomMatrixGenerator,
    LowRankAdapter,
    UTBlock,
    AdaptedAttention,
    AdaptedMLP,
)

# === Import water cycle (disabled by default) ===
from water_cycle_optimizer import WaterCycleWrapper


# =============================================================================
# UT-specific Hyperparameters (extend the base class)
# =============================================================================

class UTHyperparameters(Hyperparameters):
    """Extended hyperparameters for the Universal Transformer variant."""
    # Architecture
    ut_model_dim = int(os.environ.get("UT_MODEL_DIM", 768))
    ut_num_heads = int(os.environ.get("UT_NUM_HEADS", 12))
    ut_num_kv_heads = int(os.environ.get("UT_NUM_KV_HEADS", 6))
    ut_mlp_mult = float(os.environ.get("UT_MLP_MULT", 3.0))
    ut_recurrence_steps = int(os.environ.get("UT_RECURRENCE_STEPS", 11))
    ut_adapter_sets = int(os.environ.get("UT_ADAPTER_SETS", 3))
    ut_adapter_rank = int(os.environ.get("UT_ADAPTER_RANK", 256))
    ut_adapter_alpha = float(os.environ.get("UT_ADAPTER_ALPHA", 1.0))
    ut_adapter_strategy = os.environ.get("UT_ADAPTER_STRATEGY", "phase")
    ut_random_seed = int(os.environ.get("UT_RANDOM_SEED", 42))

    # Optimizer (all AdamW, no Muon)
    ut_adapter_lr = float(os.environ.get("UT_ADAPTER_LR", 3e-4))
    ut_embed_lr = float(os.environ.get("UT_EMBED_LR", 0.035))
    ut_scalar_lr = float(os.environ.get("UT_SCALAR_LR", 0.025))

    # Water cycle
    wc_enabled = bool(int(os.environ.get("WC_ENABLED", "0")))
    wc_trigger_fraction = float(os.environ.get("WC_TRIGGER_FRACTION", 0.80))
    wc_num_particles = int(os.environ.get("WC_NUM_PARTICLES", 5))
    wc_noise_scale = float(os.environ.get("WC_NOISE_SCALE", 0.01))
    wc_atmosphere_ratio = float(os.environ.get("WC_ATMOSPHERE_RATIO", 0.1))


# =============================================================================
# Training
# =============================================================================

def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    args = UTHyperparameters()

    # --- Distributed setup (same as SOTA) ---
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    grad_accum_steps = max(1, 8 // world_size)
    grad_scale = 1.0 / grad_accum_steps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()

    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- Logging ---
    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0("=" * 100, console=False)

    # --- Seed ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # --- Tokenizer ---
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE mismatch: {args.vocab_size} vs {int(sp.vocab_size())}")

    # --- Validation data ---
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece")
    log0(f"val_tokens:{val_tokens.numel() - 1}")

    # =================================================================
    # MODEL: Universal Transformer with Random Matrix Adapters
    # =================================================================

    base_model = UTGPT(
        vocab_size=args.vocab_size,
        model_dim=args.ut_model_dim,
        num_heads=args.ut_num_heads,
        num_kv_heads=args.ut_num_kv_heads,
        mlp_mult=args.ut_mlp_mult,
        num_recurrence_steps=args.ut_recurrence_steps,
        num_adapter_sets=args.ut_adapter_sets,
        adapter_rank=args.ut_adapter_rank,
        adapter_alpha=args.ut_adapter_alpha,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        rope_dims=args.rope_dims,
        qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
        random_seed=args.ut_random_seed,
        adapter_strategy=args.ut_adapter_strategy,
    ).to(device).bfloat16()

    # Restore small params to FP32 (same as SOTA)
    restore_low_dim_params_to_fp32(base_model)
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()

    # DDP wrapper (standard — no fancy reduce-scatter needed for small adapters)
    if distributed:
        base_model = nn.parallel.DistributedDataParallel(
            base_model, device_ids=[local_rank],
            broadcast_buffers=False,
        )
    raw_model = base_model.module if distributed else base_model

    # Compile for speed
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)

    # =================================================================
    # OPTIMIZER: Pure AdamW (no Muon — adapters are low-rank, not square)
    # =================================================================

    # Split params into groups with different LRs
    adapter_params = []      # A, B matrices in LowRankAdapter
    step_params = []         # step_scale, step_shift
    embed_params = []        # tok_emb, bigram embed, VE embed
    scalar_params = []       # norms, scales, gates, skip weights, q_gain, etc.

    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if 'adapter' in name and ('A' in name.split('.')[-1] or 'B' in name.split('.')[-1]):
            adapter_params.append(param)
        elif 'step_scale' in name or 'step_shift' in name:
            step_params.append(param)
        elif 'tok_emb' in name or 'bigram.embed' in name or 've_shared.embed' in name:
            embed_params.append(param)
        else:
            scalar_params.append(param)

    param_groups = [
        {"params": adapter_params, "lr": args.ut_adapter_lr, "base_lr": args.ut_adapter_lr},
        {"params": step_params, "lr": args.ut_scalar_lr, "base_lr": args.ut_scalar_lr},
        {"params": embed_params, "lr": args.ut_embed_lr, "base_lr": args.ut_embed_lr},
        {"params": scalar_params, "lr": args.ut_scalar_lr, "base_lr": args.ut_scalar_lr},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )

    # =================================================================
    # WATER CYCLE (disabled by default, activated via WC_ENABLED=1)
    # =================================================================

    water_cycle = WaterCycleWrapper(
        enabled=args.wc_enabled,
        trigger_fraction=args.wc_trigger_fraction,
        num_particles=args.wc_num_particles,
        noise_scale=args.wc_noise_scale,
        atmosphere_dim_ratio=args.wc_atmosphere_ratio,
        random_seed=args.ut_random_seed + 9999,
    )

    # =================================================================
    # LOGGING
    # =================================================================

    n_params = sum(p.numel() for p in raw_model.parameters())
    n_adapter = sum(p.numel() for p in adapter_params)
    n_step = sum(p.numel() for p in step_params)
    n_embed = sum(p.numel() for p in embed_params)
    n_scalar = sum(p.numel() for p in scalar_params)

    log0(f"=== Universal Transformer with Random Matrix Adapters ===")
    log0(f"model_params:{n_params} (adapter:{n_adapter} step:{n_step} embed:{n_embed} scalar:{n_scalar})")
    log0(f"architecture: d={args.ut_model_dim} heads={args.ut_num_heads} kv={args.ut_num_kv_heads} "
         f"mlp={args.ut_mlp_mult}x R={args.ut_recurrence_steps} K={args.ut_adapter_sets} "
         f"rank={args.ut_adapter_rank} strategy={args.ut_adapter_strategy}")
    log0(f"random_seed:{args.ut_random_seed}")
    log0(f"optimizer: AdamW (adapter_lr={args.ut_adapter_lr} embed_lr={args.ut_embed_lr} "
         f"scalar_lr={args.ut_scalar_lr})")
    log0(f"water_cycle: enabled={args.wc_enabled}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
         f"iterations:{args.iterations}")
    log0(f"seed:{args.seed}")

    # =================================================================
    # TRAINING LOOP
    # =================================================================

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        """Warmdown learning rate schedule (same as SOTA)."""
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            if warmdown_start <= step < args.iterations:
                return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # EMA state for weight averaging
    ema_state = {name: t.detach().float().clone() for name, t in raw_model.state_dict().items()}
    ema_decay = 0.997

    training_time_ms = 0.0
    stop_after_step: int | None = None

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)

        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args, compiled_model if not distributed else base_model,
                rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                 f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms")
            
            # Flush Rotary's cached cos/sin tensors — they were created under
            # torch.inference_mode() inside eval_val and will poison autograd
            # if reused in the next training forward pass.
            raw_model.rotary._cos_cached = None
            raw_model.rotary._sin_cached = None
            raw_model.rotary._seq_len_cached = 0
            
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}")
            break

        # --- LR schedule ---
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * scale

        # --- Forward/backward ---
        optimizer.zero_grad(set_to_none=True)
        train_loss = torch.zeros((), device=device)

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)

            # Prevent DDP AllReduce on every micro-step — only sync on the last one
            sync_context = (base_model.no_sync()
                           if distributed and micro_step < grad_accum_steps - 1
                           else contextlib.nullcontext())

            with sync_context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    loss = compiled_model(x, y) if not distributed else base_model(x, y)
                train_loss += loss.detach()
                (loss * grad_scale).backward()

        train_loss /= grad_accum_steps

        # --- Gradient clipping ---
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip_norm)

        # --- Optimizer step ---
        optimizer.step()

        # --- EMA update ---
        with torch.no_grad():
            for name, t in raw_model.state_dict().items():
                ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)

        # --- Water cycle check ---
        water_cycle.record_loss(step, train_loss.item())
        if water_cycle.should_evaporate(step, args.iterations):
            # Micro-eval: evaluate on a few training batches (cheap).
            # CRITICAL: All-reduce the loss so every DDP rank agrees on scores,
            # otherwise different ranks could pick different winning particles.
            def micro_eval(model):
                model.eval()
                total_loss = torch.zeros((), device=device)
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    for _ in range(4):
                        x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                        total_loss += model(x, y).detach()
                if distributed:
                    dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
                model.train()
                return total_loss.item() / 4

            water_cycle.evaporate_and_precipitate(
                raw_model, micro_eval, optimizer=optimizer,
                log_fn=lambda msg: log0(msg),
            )

        step += 1

        # --- Logging ---
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log = (args.train_log_every > 0 and
                      (step <= 10 or step % args.train_log_every == 0))
        if should_log:
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_training_time_ms:.0f}ms "
                 f"step_avg:{approx_training_time_ms / step:.2f}ms")

        # --- Wallclock cap ---
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    # =================================================================
    # POST-TRAINING: Apply EMA, quantize, evaluate
    # =================================================================

    log0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
         f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")

    # Apply EMA weights
    log0("ema:applying EMA weights")
    current_state = raw_model.state_dict()
    avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
    raw_model.load_state_dict(avg_state, strict=True)

    # Diagnostic eval before quantization
    torch.cuda.synchronize()
    t_diag = time.perf_counter()
    diag_val_loss, diag_val_bpb = eval_val(
        args, compiled_model if not distributed else base_model,
        rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(f"DIAGNOSTIC post_ema val_loss:{diag_val_loss:.4f} val_bpb:{diag_val_bpb:.4f} "
         f"eval_time:{1000.0 * (time.perf_counter() - t_diag):.0f}ms")

    # =================================================================
    # QUANTIZATION: Int8 for adapters (they're small, GPTQ likely overkill)
    # =================================================================

    # Export state dict (exclude random matrices — they're regenerated from seed)
    export_sd = {k: v.detach().cpu() for k, v in raw_model.state_dict().items()}

    if master_process:
        # Standard int8 quantization + compression
        quant_obj, quant_stats = quantize_state_dict_int8(export_sd)
        quant_buf = io.BytesIO()
        torch.save(quant_obj, quant_buf)
        quant_raw = quant_buf.getvalue()

        # Try both zlib and lzma, use whichever is smaller
        zlib_blob = __import__('zlib').compress(quant_raw, 9)
        lzma_blob = lzma.compress(quant_raw, preset=9)
        best_blob = lzma_blob if len(lzma_blob) < len(zlib_blob) else zlib_blob
        best_method = "lzma" if len(lzma_blob) < len(zlib_blob) else "zlib"

        with open("final_model_ut.ptz", "wb") as f:
            f.write(best_blob)

        code_bytes = len(code.encode("utf-8"))
        model_bytes = len(best_blob)
        log0(f"Serialized model int8+{best_method}: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes "
             f"({(model_bytes + code_bytes) / 1e6:.2f} MB)")
        log0(f"Quant stats: {quant_stats}")

    # =================================================================
    # ROUNDTRIP EVAL: Load quantized model and evaluate
    # =================================================================

    if distributed:
        dist.barrier()

    with open("final_model_ut.ptz", "rb") as f:
        blob = f.read()

    # Detect compression method
    if blob[:4] == b'\xfd7zX':  # LZMA magic bytes
        raw = lzma.decompress(blob)
    else:
        raw = __import__('zlib').decompress(blob)

    quant_obj_loaded = torch.load(io.BytesIO(raw), map_location="cpu")
    deq_sd = dequantize_state_dict_int8(quant_obj_loaded)

    eval_model = UTGPT(
        vocab_size=args.vocab_size,
        model_dim=args.ut_model_dim,
        num_heads=args.ut_num_heads,
        num_kv_heads=args.ut_num_kv_heads,
        mlp_mult=args.ut_mlp_mult,
        num_recurrence_steps=args.ut_recurrence_steps,
        num_adapter_sets=args.ut_adapter_sets,
        adapter_rank=args.ut_adapter_rank,
        adapter_alpha=args.ut_adapter_alpha,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        rope_dims=args.rope_dims,
        qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
        random_seed=args.ut_random_seed,
        adapter_strategy=args.ut_adapter_strategy,
    ).to(device).bfloat16()

    restore_low_dim_params_to_fp32(eval_model)
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    eval_model.load_state_dict(deq_sd, strict=True)
    compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True)

    # Standard eval
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, compiled_eval, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        eval_seq_len=effective_eval_seq_len,
    )
    torch.cuda.synchronize()
    log0(f"final_int8_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
         f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms")
    log0(f"final_int8_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    # Sliding window eval
    sw_seq_len = effective_eval_seq_len
    if args.eval_stride > 0 and args.eval_stride < sw_seq_len:
        torch.cuda.synchronize()
        t_slide = time.perf_counter()
        sw_val_loss, sw_val_bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride,
            eval_seq_len=sw_seq_len,
        )
        torch.cuda.synchronize()
        log0(f"final_sliding_window val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} "
             f"stride:{args.eval_stride} eval_time:{1000.0 * (time.perf_counter() - t_slide):.0f}ms")
        log0(f"final_int8_zlib_roundtrip_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()