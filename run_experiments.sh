#!/bin/bash
# =============================================================================
# Parameter Golf — Experiment Runner
# =============================================================================
# Run the full battery of experiments overnight.
# Usage:
#   chmod +x run_experiments.sh
#   ./run_experiments.sh           # Run all phases
#   ./run_experiments.sh phase0    # Run only Phase 0
#   ./run_experiments.sh phase1    # Run only Phase 1
#   ...etc
#
# Results go to outputs/<RUN_ID>/<RUN_ID>.log and outputs/<RUN_ID>/<RUN_ID>.ptz
# =============================================================================

set +e  # Don't exit on error — failed runs should not kill the batch

# --- Common settings ---
export OUTPUT_DIR="outputs"
export MAX_WALLCLOCK_SECONDS=0       # No wallclock cap (local dev)
export TRAIN_BATCH_TOKENS=524288     # Override leader's default of 786432 to fit 32GB GPU
export TRAIN_SEQ_LEN=2048
export EVAL_STRIDE=0                 # Skip sliding window eval (saves ~36 min per run)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Number of training steps — adjust for quick smoke tests vs full runs
# For overnight: 500. For quick smoke: 50-100.
STEPS=${STEPS:-500}
VAL_EVERY=${VAL_EVERY:-100}

mkdir -p "$OUTPUT_DIR"

PHASE="${1:-all}"

run_ut() {
    local run_id="$1"
    shift
    
    # Skip if this run already completed successfully
    local logfile="$OUTPUT_DIR/$run_id/$run_id.log"
    if [ -f "$logfile" ] && grep -q "final_.*_roundtrip_exact" "$logfile" 2>/dev/null; then
        echo ""
        echo "  SKIPPING: $run_id (already completed — delete $OUTPUT_DIR/$run_id to rerun)"
        echo ""
        return 0
    fi
    
    echo ""
    echo "=================================================================="
    echo "  STARTING: $run_id"
    echo "  $(date)"
    echo "=================================================================="
    
    # Export all passed KEY=VALUE pairs as environment variables
    for arg in "$@"; do
        export "$arg"
    done
    
    export RUN_ID="$run_id"
    export ITERATIONS="$STEPS"
    export VAL_LOSS_EVERY="$VAL_EVERY"
    
    python3 train_gpt_ut.py
    local exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        echo "  FAILED: $run_id (exit code $exit_code) at $(date)"
        echo "  Continuing to next run..."
    else
        echo "  FINISHED: $run_id at $(date)"
    fi
    echo "=================================================================="
    echo ""
}

# =====================================================================
# PHASE 0 — Baselines
# =====================================================================
phase0() {
    echo "========== PHASE 0: Baselines =========="

    # B1: Current leader's train_gpt.py (reference — uncomment if you want)
    # echo "Skipping B1 (leader baseline) — run manually if desired:"
    # echo "  RUN_ID=B1_leader ITERATIONS=$STEPS MAX_WALLCLOCK_SECONDS=0 VAL_LOSS_EVERY=$VAL_EVERY python3 train_gpt.py"

    # B2: Our UT model, FP16 export (no quantization — true performance floor)
    run_ut "B2_fp16_baseline" \
        EXPORT_MODE=fp16 \
        QAT_ENABLED=0 \
        UT_ADAPTER_RANK=256 \
        UT_ADAPTER_SETS=3 \
        UT_ADAPTER_LR=0.01
}

# =====================================================================
# PHASE 1 — Fused ΔW Export (fix quantization)
# =====================================================================
phase1() {
    echo "========== PHASE 1: Fused ΔW Export =========="

    # F1: Fused ΔW int8, rank 256 (current rank)
    run_ut "F1_fused_r256" \
        EXPORT_MODE=fused_int8 \
        QAT_ENABLED=0 \
        UT_ADAPTER_RANK=256 \
        UT_ADAPTER_SETS=3 \
        UT_ADAPTER_LR=0.01

    # F2: Fused ΔW int8, rank 384
    run_ut "F2_fused_r384" \
        EXPORT_MODE=fused_int8 \
        QAT_ENABLED=0 \
        UT_ADAPTER_RANK=384 \
        UT_ADAPTER_SETS=3 \
        UT_ADAPTER_LR=0.01

    # F3: Fused ΔW int8, rank 512 (test VRAM limits)
    run_ut "F3_fused_r512" \
        EXPORT_MODE=fused_int8 \
        QAT_ENABLED=0 \
        UT_ADAPTER_RANK=512 \
        UT_ADAPTER_SETS=3 \
        UT_ADAPTER_LR=0.01
}

# =====================================================================
# PHASE 2 — Late QAT (harden against int8 error)
# =====================================================================
phase2() {
    echo "========== PHASE 2: Late QAT =========="

    # Q1: Fused + QAT at 85%, rank 256
    run_ut "Q1_fused_qat_r256" \
        EXPORT_MODE=fused_int8 \
        QAT_ENABLED=1 \
        QAT_THRESHOLD=0.85 \
        UT_ADAPTER_RANK=256 \
        UT_ADAPTER_SETS=3 \
        UT_ADAPTER_LR=0.01

    # Q2: Fused + QAT, best rank from Phase 1
    # (Adjust rank based on Phase 1 results — default to 384)
    run_ut "Q2_fused_qat_r384" \
        EXPORT_MODE=fused_int8 \
        QAT_ENABLED=1 \
        QAT_THRESHOLD=0.85 \
        UT_ADAPTER_RANK=384 \
        UT_ADAPTER_SETS=3 \
        UT_ADAPTER_LR=0.01
}

# =====================================================================
# PHASE 3 — Architecture expansion
# =====================================================================
phase3() {
    echo "========== PHASE 3: Architecture Expansion =========="

    # A1: Best config + K=4 adapter sets
    run_ut "A1_K4" \
        EXPORT_MODE=fused_int8 \
        QAT_ENABLED=1 \
        QAT_THRESHOLD=0.85 \
        UT_ADAPTER_RANK=384 \
        UT_ADAPTER_SETS=4 \
        UT_ADAPTER_LR=0.01

    # A2: Best config + K=5 adapter sets
    run_ut "A2_K5" \
        EXPORT_MODE=fused_int8 \
        QAT_ENABLED=1 \
        QAT_THRESHOLD=0.85 \
        UT_ADAPTER_RANK=384 \
        UT_ADAPTER_SETS=5 \
        UT_ADAPTER_LR=0.01

    # A3: Bigger bigram table
    run_ut "A3_bigram8k" \
        EXPORT_MODE=fused_int8 \
        QAT_ENABLED=1 \
        QAT_THRESHOLD=0.85 \
        UT_ADAPTER_RANK=384 \
        UT_ADAPTER_SETS=3 \
        UT_ADAPTER_LR=0.01 \
        BIGRAM_VOCAB_SIZE=8192 \
        BIGRAM_DIM=160
}

# =====================================================================
# PHASE 4 — Water Cycle
# =====================================================================
phase4() {
    echo "========== PHASE 4: Water Cycle =========="

    # W1: Water cycle 80/20 split
    run_ut "W1_wc_80_20" \
        EXPORT_MODE=fused_int8 \
        QAT_ENABLED=1 \
        QAT_THRESHOLD=0.85 \
        UT_ADAPTER_RANK=384 \
        UT_ADAPTER_SETS=3 \
        UT_ADAPTER_LR=0.01 \
        WC_ENABLED=1 \
        WC_TRIGGER_FRACTION=0.80 \
        WC_NUM_PARTICLES=5

    # W2: Water cycle 70/30, more particles
    run_ut "W2_wc_70_30" \
        EXPORT_MODE=fused_int8 \
        QAT_ENABLED=1 \
        QAT_THRESHOLD=0.85 \
        UT_ADAPTER_RANK=384 \
        UT_ADAPTER_SETS=3 \
        UT_ADAPTER_LR=0.01 \
        WC_ENABLED=1 \
        WC_TRIGGER_FRACTION=0.70 \
        WC_NUM_PARTICLES=8 \
        WC_NOISE_SCALE=0.01
}

# =====================================================================
# Dispatch
# =====================================================================
case "$PHASE" in
    all)
        phase0
        phase1
        phase2
        phase3
        phase4
        ;;
    phase0) phase0 ;;
    phase1) phase1 ;;
    phase2) phase2 ;;
    phase3) phase3 ;;
    phase4) phase4 ;;
    *)
        echo "Usage: $0 [all|phase0|phase1|phase2|phase3|phase4]"
        exit 1
        ;;
esac

echo ""
echo "========== ALL EXPERIMENTS COMPLETE =========="
echo "Results in: $OUTPUT_DIR/"
echo ""

# Print summary of all results
echo "=== RESULTS SUMMARY ==="
for logfile in "$OUTPUT_DIR"/*/*.log; do
    if [ -f "$logfile" ]; then
        run_name=$(basename "$(dirname "$logfile")")
        # Extract key metrics from log — use ^ to match start of line (actual output)
        # not source code lines that contain the same strings
        diag=$(grep "^DIAGNOSTIC" "$logfile" 2>/dev/null | tail -1)
        roundtrip=$(grep "^final_.*_roundtrip_exact" "$logfile" 2>/dev/null | head -1)
        size=$(grep "^Total submission size" "$logfile" 2>/dev/null | tail -1)
        echo "--- $run_name ---"
        [ -n "$diag" ] && echo "  $diag"
        [ -n "$roundtrip" ] && echo "  $roundtrip"
        [ -n "$size" ] && echo "  $size"
        [ -z "$diag" ] && [ -z "$roundtrip" ] && echo "  (incomplete or failed)"
        echo ""
    fi
done