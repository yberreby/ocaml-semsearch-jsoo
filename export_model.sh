#!/bin/bash

set -v

cd to-tfjs

python -m venv env
source env/bin/activate
pip install -r requirements.txt

python export.py \
    --model-id "sentence-transformers/all-MiniLM-L6-v2" \
    --out-dir ../exported \
    --max-tokens 128

deactivate
