import onnxruntime as ort
import numpy as np

# Step 1: Load ONNX model
session = ort.InferenceSession("ann_model.onnx")

# Step 2: Prepare input data
input_data = np.random.randn(1, 10).astype(np.float32)

# Step 3: Run inference
outputs = session.run(None, {"input": input_data})

# Step 4: Print output
print("ONNX Prediction:", outputs)
