#!/bin/bash

set -v

cd to-tfjs

if [ ! -d "env" ]; then
    python -m venv env
fi

source env/bin/activate

if [ ! -d "env" ]; then
    pip install -r requirements.txt
fi

MODEL_ID=${1:-"paraphrase-MiniLM-L3-v2"}

python export.py \
    --out-dir ../exported \
    --model-id sentence-transformers/"$MODEL_ID"
