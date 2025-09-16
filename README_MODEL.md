# Tomato Disease Classification Model

## Model Description

This is a PyTorch model trained to classify tomato diseases with 3 classes accuracy. The model is specifically designed for agricultural applications in tomato disease detection and management.

## Model Files

- `best_model_3class.pth`: Pre-trained PyTorch model weights

## Usage

```python
import torch

# Load the model
model = torch.load('best_model_3class.pth')
model.eval()

# Use the model for inference
# Add your inference code here
```

## Model Architecture

- 3-class classification model
- PyTorch framework
- Optimized for tomato disease detection

## Training Data

The model was trained on tomato disease images for classification tasks related to plant pathology research.

## Citation

If you use this model, please cite appropriately in your research.

## License

Please respect the terms of use for this model.
