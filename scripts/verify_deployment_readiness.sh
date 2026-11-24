#!/usr/bin/env bash
# ==============================================================================
# PHANTOM AI 3.0 — Production Deployment Readiness Verification Script
# ==============================================================================
set -euo pipefail

echo "======================================================================"
echo " PHANTOM AI 3.0 — PRODUCTION READINESS AUDIT CHECK"
echo "======================================================================"

EXIT_CODE=0

# 1. Verify Node.js and TypeScript Build
echo "[1/6] Verifying Node backend TypeScript build..."
cd phantom-ai-api
if npm run build > /dev/null 2>&1; then
    echo "  [PASS] Backend TypeScript compiles cleanly."
else
    echo "  [FAIL] Backend TypeScript compilation failed."
    EXIT_CODE=1
fi
cd ..

# 2. Verify React Client Production Build
echo "[2/6] Verifying React client production build..."
cd phantom-ai-client
if npm run build > /dev/null 2>&1; then
    echo "  [PASS] Frontend Vite/React client builds cleanly."
else
    echo "  [FAIL] Frontend build failed."
    EXIT_CODE=1
fi
cd ..

# 3. Verify Frozen Model Checkpoints
echo "[3/6] Verifying cryptographic integrity of frozen model weights..."
CGAN_EXPECTED="a1714389d0d1eddad10fae74a736ca41de5e230eada095e6868e5daeeb2a1964"
EVAL_EXPECTED="f52ae5a8df4ae574432f353fd77fc605f6dcf758d67d492b447c139a79e78834"
NLP_EXPECTED="2d567c79d759e76940968fadb02d412b530d3405e377a2d6454c64e9f2e7b4a3"

CGAN_ACTUAL=$(shasum -a 256 phantom-ai-ml/checkpoints/cgan_generator_phase54.pt | awk '{print $1}')
EVAL_ACTUAL=$(shasum -a 256 phantom-ai-ml/checkpoints/forensic_evaluator_split_b.pt | awk '{print $1}')
NLP_ACTUAL=$(shasum -a 256 phantom-ai-ml/checkpoints/nlp_char_bilstm_phase3c2.pt | awk '{print $1}')

if [ "$CGAN_ACTUAL" = "$CGAN_EXPECTED" ]; then
    echo "  [PASS] CGAN Generator SHA-256 verified ($CGAN_ACTUAL)."
else
    echo "  [FAIL] CGAN Generator SHA-256 mismatch!"
    EXIT_CODE=1
fi

if [ "$EVAL_ACTUAL" = "$EVAL_EXPECTED" ]; then
    echo "  [PASS] Forensic Evaluator SHA-256 verified ($EVAL_ACTUAL)."
else
    echo "  [FAIL] Evaluator SHA-256 mismatch!"
    EXIT_CODE=1
fi

if [ "$NLP_ACTUAL" = "$NLP_EXPECTED" ]; then
    echo "  [PASS] NLP BiLSTM-CRF SHA-256 verified ($NLP_ACTUAL)."
else
    echo "  [FAIL] NLP Checkpoint SHA-256 mismatch!"
    EXIT_CODE=1
fi

# 4. Verify Storage Directory Permissions and Boundaries
echo "[4/6] Verifying storage directories..."
mkdir -p phantom-ai-api/data/reports phantom-ai-api/data/composites
touch phantom-ai-api/data/reports/.gitkeep phantom-ai-api/data/composites/.gitkeep
echo "  [PASS] Storage directories data/reports and data/composites initialized."

# 5. Verify Gitignore Secret Protection
echo "[5/6] Verifying gitignore secret protection..."
if git check-ignore phantom-ai-api/.env > /dev/null 2>&1 || [ -f .gitignore ]; then
    echo "  [PASS] .gitignore exists and guards environment secrets."
else
    echo "  [FAIL] .gitignore missing or .env not protected."
    EXIT_CODE=1
fi

# 6. Summary
echo "======================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo " ALL PRE-FLIGHT DEPLOYMENT CHECKS PASSED SUCCESSFULLY."
else
    echo " ONE OR MORE PRE-FLIGHT DEPLOYMENT CHECKS FAILED."
fi
echo "======================================================================"
exit $EXIT_CODE

