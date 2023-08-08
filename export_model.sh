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

python export.py \
    --model-id "sentence-transformers/all-MiniLM-L6-v2" \
    --out-dir ../exported
