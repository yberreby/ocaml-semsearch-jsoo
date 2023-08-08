import argparse
import subprocess
from pathlib import Path
from transformers import AutoTokenizer
from model import TFSentenceTransformer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--max-tokens",
    type=int,
    default=128,
    help="Max sequence length, including start and end tokens",
)
parser.add_argument(
    "--model-id",
    type=str,
    default="sentence-transformers/all-MiniLM-L6-v2",
    help="model id on HuggingFace",
)
parser.add_argument("--out-dir", type=Path, default="exported/")
args = parser.parse_args()


if __name__ == "__main__":
    model_id = args.model_id
    model_id_path_friendly = model_id.replace("/", "-")

    # Prepare output directories
    dirs = list(
        map(
            lambda d: Path(args.out_dir / d),
            ["sm", "js"],
        )
    )
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    sm_out_path, js_out_path = map(lambda d: d / model_id_path_friendly, dirs)

    print(f"Exporting {model_id} to {sm_out_path} with max_tokens={args.max_tokens}")
    print()

    print("Loading model")
    st = TFSentenceTransformer(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("Running model on dummy input to infer shape")
    # Don't count START and END, hence -2
    payload = "@ " * (args.max_tokens - 2)
    inputs = tokenizer([payload], padding=True, truncation=True, return_tensors="tf")
    inputs = {
        "input_ids": inputs["input_ids"].numpy(),
        "token_type_ids": inputs["token_type_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy(),
    }
    embeddings = st(inputs)

    print(f"Exporting to TensorFlow SavedModel: {sm_out_path}")
    st.save(sm_out_path, save_format="tf")

    print(f"Converting to TF.js: {js_out_path}")
    subprocess.run(
        [
            "tensorflowjs_converter",
            "--input_format",
            "tf_saved_model",
            sm_out_path,
            js_out_path,
        ]
    )
