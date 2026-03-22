# PyTorch to ONNX Export and Inference

This project demonstrates how to export a PyTorch model to ONNX format and run optimized inference using ONNX Runtime.

---

## What is ONNX?

ONNX (Open Neural Network Exchange) is an open format for machine learning models. It allows models trained in one framework (e.g. PyTorch) to be run in another runtime or on different hardware without retraining.

---

## Project Structure

```
project/
├── export.py        # Export PyTorch model to ONNX
├── infer.py         # Run inference with ONNX Runtime
├── model.onnx       # Exported ONNX model
└── requirements.txt
```

---

## Setup

```bash
pip install torch torchvision onnx onnxruntime numpy pillow
```

---

## Export Model

```bash
python export.py
```

Internally this does:

```python
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}}
)
```

---

## Run Inference

```bash
python infer.py
```

ONNX Runtime loads the model and runs inference:

```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
outputs = session.run(None, {"input": input_array})
```

---

## Why ONNX?

| Benefit | Details |
|---|---|
| Framework-agnostic | Export from PyTorch, run anywhere |
| Faster inference | ONNX Runtime is optimized for CPU and GPU |
| Smaller footprint | No PyTorch dependency at inference time |
| Cross-platform | Works on edge devices, cloud, and mobile |

---

## Learnings

- ONNX decouples training from deployment
- ONNX Runtime is significantly faster than native PyTorch on CPU
- Dynamic axes allow flexible batch sizes without re-exporting

---

## Future Improvements

- Quantize the ONNX model for further speedup
- Deploy with TensorRT for GPU acceleration
- Benchmark ONNX Runtime vs PyTorch vs JIT
