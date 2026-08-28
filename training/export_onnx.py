"""Export the trained FallClassifier checkpoint to ONNX and verify it matches
the PyTorch output, so it can be dropped into the backend's models/ folder.
"""
import os
import numpy as np
import torch
import onnxruntime as ort

from model import FallClassifier, FEATURES_PER_FRAME
from dataset import WINDOW_SIZE

CKPT_PATH = os.environ.get(
    "EXPORT_CKPT_PATH", os.path.join(os.path.dirname(__file__), "data", "best_model.pt")
)
ONNX_OUT = os.environ.get(
    "EXPORT_ONNX_OUT", os.path.join(os.path.dirname(__file__), "data", "fall_classifier_v3.onnx")
)


def main():
    hidden = int(os.environ.get("HIDDEN_SIZE", "128"))
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = FallClassifier(hidden=hidden)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, val_f1={ckpt['val_f1']:.3f}, val_acc={ckpt['val_acc']:.3f}")

    dummy = torch.randn(1, WINDOW_SIZE, FEATURES_PER_FRAME)

    torch.onnx.export(
        model,
        dummy,
        ONNX_OUT,
        input_names=["input"],
        output_names=["logit"],
        dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=13,
    )
    print(f"Exported ONNX model to: {ONNX_OUT}")

    # Verify PyTorch vs ONNX outputs match
    with torch.no_grad():
        torch_out = model(dummy).numpy()

    sess = ort.InferenceSession(ONNX_OUT, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["logit"], {"input": dummy.numpy().astype(np.float32)})[0]

    diff = np.abs(torch_out - onnx_out).max()
    print(f"Max diff between torch and onnx outputs: {diff:.6f}")
    assert diff < 1e-4, "ONNX export mismatch!"
    print("ONNX export verified OK.")


if __name__ == "__main__":
    main()
