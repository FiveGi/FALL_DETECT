import os
import sys
import torch
from ultralytics import YOLO

DATA_YAML = os.path.join(os.path.dirname(__file__), "data", "ft_dataset", "data.yaml")


def main():
    torch.set_num_threads(8)
    model_name = sys.argv[1] if len(sys.argv) > 1 else "yolo26m.pt"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    model = YOLO(model_name)
    model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=640,
        batch=8,
        patience=10,
        lr0=0.001,       # low LR for fine-tuning, not training from scratch
        workers=0,       # avoid Windows spawn-multiprocessing overhead/pitfalls entirely
        project=os.path.join(os.path.dirname(__file__), "data", "ft_runs"),
        name=model_name.replace(".pt", "") + "_person_finetune",
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
