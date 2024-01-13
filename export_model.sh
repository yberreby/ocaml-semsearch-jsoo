#!/bin/bash

set -ve

cd to-tfjs

env_exists=0
if [ -d "env" ]; then
    env_exists=1
else
    python -m venv env
fi

source env/bin/activate

if [ $env_exists -eq 0 ]; then
    pip install -r requirements.txt
fi

MODEL_ID=${1:-"all-MiniLM-L6-v2"}

python export.py \
    --out-dir ../exported \
    --model-id sentence-transformers/"$MODEL_ID"
