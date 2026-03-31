"""
Water Cycle Optimizer
=====================

A meta-optimizer that wraps standard AdamW training with bio-inspired
evaporation/condensation dynamics for escaping sharp local minima.

Design philosophy:
- "Flow" phase (0-80% of training): pure AdamW, aggressive descent
- "Curvature check" (at trigger point): estimate basin sharpness
- "Evaporation" (if in sharp minimum): escape via random projection transport
- "Precipitation": spawn candidate particles from atmospheric distribution
- "Final sprint" (80-100%): best particle rides AdamW to finish

The key insight vs simulated annealing: SA perturbs in-place (jiggle where
you are). The water cycle TRANSPORTS through a different medium — parameters
are projected to a low-dimensional space, perturbed there, and projected back.
This enables long-range jumps in parameter space, not just local perturbations.

Basin sharpness is estimated via loss improvement rate (not gradient variance,
which is too noisy during warmdown).
"""

import hashlib
import math
import copy
import torch
import torch.nn as nn
from torch import Tensor
from collections import deque
from typing import Optional


class WaterCycleWrapper:
    """Wraps the training loop with water cycle dynamics.
    
    Usage:
        wc = WaterCycleWrapper(model, optimizer, enabled=True, ...)
        
        for step in range(total_steps):
            # Normal training step
            loss = train_step(...)
            wc.record_loss(step, loss)
            
            # Check if we should trigger evaporation
            if wc.should_evaporate(step, total_steps):
                wc.evaporate_and_precipitate(model, train_fn, eval_fn)
    
    Args:
        model: the UTGPT model (or any nn.Module)
        optimizer: the AdamW optimizer for adapter params
        enabled: if False, all methods are no-ops
        trigger_fraction: fraction of training at which to check curvature (default 0.8)
        num_particles: number of candidate particles to spawn during precipitation
        atmosphere_dim_ratio: dimensionality reduction ratio for atmospheric transport
        noise_scale: scale of Gaussian noise in atmosphere space
        eval_microbatches: number of microbatches to evaluate each particle
        improvement_window: number of steps to track loss improvement rate
        improvement_threshold: minimum loss improvement rate to consider "still flowing"
        random_seed: seed for reproducible atmospheric transport
    """
    
    def __init__(
        self,
        enabled: bool = False,
        trigger_fraction: float = 0.80,
        num_particles: int = 5,
        atmosphere_dim_ratio: float = 0.1,
        noise_scale: float = 0.01,
        eval_microbatches: int = 4,
        improvement_window: int = 100,
        improvement_threshold: float = 1e-5,
        random_seed: int = 12345,
    ):
        self.enabled = enabled
        self.trigger_fraction = trigger_fraction
        self.num_particles = num_particles
        self.atmosphere_dim_ratio = atmosphere_dim_ratio
        self.noise_scale = noise_scale
        self.eval_microbatches = eval_microbatches
        self.improvement_window = improvement_window
        self.improvement_threshold = improvement_threshold
        self.random_seed = random_seed
        
        # Loss tracking
        self._loss_history: deque[float] = deque(maxlen=improvement_window)
        self._has_evaporated = False
        
        # Atmospheric transport matrix (lazily initialized)
        self._projection_matrix: Optional[Tensor] = None
    
    def record_loss(self, step: int, loss: float) -> None:
        """Record training loss for improvement rate tracking."""
        if not self.enabled:
            return
        self._loss_history.append(loss)
    
    def _compute_improvement_rate(self) -> float:
        """Compute average loss improvement per step over recent history.
        
        Returns positive value if loss is decreasing (good),
        near-zero or negative if plateaued/increasing (candidate for evaporation).
        """
        if len(self._loss_history) < 10:
            return float('inf')  # Not enough data, assume still flowing
        
        history = list(self._loss_history)
        n = len(history)
        mid = n // 2
        
        first_half_mean = sum(history[:mid]) / mid
        second_half_mean = sum(history[mid:]) / (n - mid)
        
        # Improvement rate: positive means loss is going down
        return (first_half_mean - second_half_mean) / mid
    
    def should_evaporate(self, step: int, total_steps: int) -> bool:
        """Check if we should trigger the evaporation-condensation cycle.
        
        Triggers at most once, when:
        1. We've passed the trigger fraction of training
        2. Loss improvement has plateaued (we're stuck in a basin)
        """
        if not self.enabled or self._has_evaporated:
            return False
        
        progress = step / max(total_steps, 1)
        if progress < self.trigger_fraction:
            return False
        
        improvement_rate = self._compute_improvement_rate()
        
        # If still improving meaningfully, keep flowing
        if improvement_rate > self.improvement_threshold:
            return False
        
        return True
    
    def _get_adapter_params(self, model: nn.Module) -> list[nn.Parameter]:
        """Extract all adapter parameters (A and B matrices, step_scale, step_shift)."""
        adapter_params = []
        for name, param in model.named_parameters():
            if any(key in name for key in ['adapter', 'step_scale', 'step_shift']):
                adapter_params.append(param)
        return adapter_params
    
    def _flatten_params(self, params: list[nn.Parameter]) -> Tensor:
        """Flatten a list of parameters into a single 1D tensor."""
        return torch.cat([p.data.detach().reshape(-1) for p in params])
    
    def _unflatten_params(self, flat: Tensor, params: list[nn.Parameter]) -> None:
        """Write a flattened tensor back into parameter data."""
        offset = 0
        for p in params:
            numel = p.numel()
            p.data.copy_(flat[offset:offset + numel].reshape(p.shape))
            offset += numel
    
    def _get_projection_matrix(self, full_dim: int, device: torch.device) -> Tensor:
        """Get the random projection matrix for atmospheric transport.
        
        Uses Johnson-Lindenstrauss random projection: a matrix of shape
        (reduced_dim, full_dim) with entries drawn from N(0, 1/reduced_dim).
        
        This preserves pairwise distances in expectation while compressing
        the parameter space by atmosphere_dim_ratio.
        """
        if self._projection_matrix is not None and self._projection_matrix.shape[1] == full_dim:
            return self._projection_matrix.to(device)
        
        reduced_dim = max(int(full_dim * self.atmosphere_dim_ratio), 64)
        
        # Deterministic random projection
        rng = torch.Generator(device='cpu')
        name_hash = int(hashlib.md5(b'atmosphere_projection').hexdigest()[:8], 16)
        rng.manual_seed(self.random_seed ^ name_hash)
        
        # Sparse random projection (faster than dense for large dims)
        P = torch.randn(reduced_dim, full_dim, generator=rng) / math.sqrt(reduced_dim)
        self._projection_matrix = P
        return P.to(device)
    
    @torch.no_grad()
    def evaporate_and_precipitate(
        self,
        model: nn.Module,
        eval_fn,  # Callable: (model) -> float (returns loss)
        optimizer: torch.optim.Optimizer = None,  # AdamW optimizer for momentum reset
        log_fn = None,  # Optional: Callable: (str) -> None
    ) -> None:
        """Execute one evaporation-condensation cycle.
        
        1. Flatten current adapter params (the "water" in the current basin)
        2. Project to atmosphere (low-dim JL projection)
        3. Generate particles by adding noise in atmosphere space
        4. Project particles back to full parameter space (precipitation)
        5. Evaluate each particle, keep the best
        
        Args:
            model: the model whose adapter params to optimize
            eval_fn: function that evaluates model and returns loss (lower = better).
                     IMPORTANT: This must be a CHEAP micro-validation — e.g. 2-4
                     training batches — NOT a full validation pass. With 5 particles,
                     a full eval would eat 1-2 minutes of the 10-minute budget.
            log_fn: optional logging function
        """
        if not self.enabled:
            return
        
        self._has_evaporated = True
        
        def log(msg):
            if log_fn:
                log_fn(f"water_cycle: {msg}")
        
        adapter_params = self._get_adapter_params(model)
        if not adapter_params:
            log("no adapter params found, skipping")
            return
        
        # Save original state
        original_flat = self._flatten_params(adapter_params)
        original_loss = eval_fn(model)
        log(f"evaporation triggered. current loss: {original_loss:.6f}")
        
        device = original_flat.device
        full_dim = original_flat.numel()
        P = self._get_projection_matrix(full_dim, device)
        
        # Project to atmosphere
        atmosphere_point = P @ original_flat.float()
        log(f"projected {full_dim} dims -> {atmosphere_point.numel()} atmospheric dims")
        
        # Generate particles by adding noise in atmosphere space
        # Scale noise relative to the magnitude of the atmospheric representation
        atm_std = atmosphere_point.std().item()
        noise_magnitude = self.noise_scale * atm_std
        
        best_loss = original_loss
        best_flat = original_flat.clone()
        
        # CRITICAL: Use a seeded RNG for noise generation so all DDP ranks
        # generate identical particles. Without this, each GPU would precipitate
        # different parameters and the next backward pass would desync/hang.
        noise_rng = torch.Generator(device='cpu')
        
        for i in range(self.num_particles):
            # Seed deterministically per-particle so results are reproducible
            noise_rng.manual_seed(self.random_seed + i * 7919)  # 7919 is prime
            noise = torch.randn(
                atmosphere_point.shape, generator=noise_rng, 
                device='cpu', dtype=torch.float32
            ).to(device) * noise_magnitude
            perturbed_atm = atmosphere_point + noise
            
            # Precipitate: project back to full parameter space
            # Use pseudoinverse: P^T (P P^T)^{-1}, but since P is JL,
            # P^T is a good enough approximate inverse (up to scaling)
            precipitated = P.T @ perturbed_atm
            
            # The precipitation won't exactly reconstruct the original
            # because JL projection loses information. Add back the
            # component that was in the null space of P.
            null_component = original_flat.float() - P.T @ (P @ original_flat.float())
            precipitated = precipitated + null_component
            
            # Load into model and evaluate
            self._unflatten_params(precipitated.to(original_flat.dtype), adapter_params)
            particle_loss = eval_fn(model)
            
            log(f"particle {i+1}/{self.num_particles}: loss={particle_loss:.6f} "
                f"(delta={particle_loss - original_loss:+.6f})")
            
            if particle_loss < best_loss:
                best_loss = particle_loss
                best_flat = precipitated.to(original_flat.dtype).clone()
        
        # Load best particle (or original if none improved)
        self._unflatten_params(best_flat, adapter_params)
        improvement = original_loss - best_loss
        
        if improvement > 0:
            log(f"condensation complete. improvement: {improvement:.6f} "
                f"({original_loss:.6f} -> {best_loss:.6f})")
            # CRITICAL: Wipe AdamW's exp_avg and exp_avg_sq for teleported params.
            # Without this, stale momentum from the old basin would yank parameters
            # in the wrong direction on the very first post-condensation step.
            if optimizer is not None:
                for p in adapter_params:
                    if p in optimizer.state:
                        optimizer.state[p].clear()
                log("reset AdamW momentum buffers for adapter params")
        else:
            log(f"no improvement found. restoring original params. "
                f"(best particle: {best_loss:.6f} vs original: {original_loss:.6f})")
            self._unflatten_params(original_flat, adapter_params)
    
    def get_state(self) -> dict:
        """Serialize state for checkpointing."""
        return {
            'has_evaporated': self._has_evaporated,
            'loss_history': list(self._loss_history),
        }
    
    def load_state(self, state: dict) -> None:
        """Restore state from checkpoint."""
        self._has_evaporated = state.get('has_evaporated', False)
        self._loss_history = deque(state.get('loss_history', []), maxlen=self.improvement_window)