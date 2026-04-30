#!/bin/bash
set -e
MODELS="sonnet-4-6 haiku-4-5 gpt-5.4-mini deepseek-v3.2 gemini-2.5-pro"

for LANG in python javascript; do
  for DS in SO_AT SO_LY LLM_AT LLM_LY; do
    echo "=== $LANG $DS ==="
    PYTHONPATH=src .venv/bin/python -m slop_bench.run_experiment \
      --language $LANG --dataset $DS \
      --models $MODELS
  done
done
