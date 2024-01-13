import argparse
import subprocess
from pathlib import Path
from transformers import AutoTokenizer
from model import TFSentenceTransformer
import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Disable GPU support explicitly
tf.config.set_visible_devices([], "GPU")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-id",
    type=str,
    default="sentence-transformers/all-MiniLM-L6-v2",
    help="model id on HuggingFace",
)
parser.add_argument("--out-dir", type=Path, default="exported/")
args = parser.parse_args()


def build_transformer(model_name_or_path, **kwargs):
    input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name="attention_mask"
    )
    token_type_ids = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name="token_type_ids"
    )
    transformer = TFSentenceTransformer(model_name_or_path, **kwargs)
    embeddings = transformer(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
    )
    model = tf.keras.Model(
        inputs=[input_ids, attention_mask, token_type_ids], outputs=embeddings
    )
    return model


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

    print(f"Exporting {model_id} to {sm_out_path}")
    print()

    print("Loading model")
    st = build_transformer(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

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
