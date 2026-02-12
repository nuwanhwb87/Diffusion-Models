# Diffusion Models Learning & Implementation Project

## Project Overview
Educational codebase for understanding and implementing diffusion models across multiple complexity levels:
- **Part 1**: Unconditional diffusion on synthetic 2D data (circles/moons)
- **Part 2**: MNIST digit generation (Part 2.1: PyTorch from scratch, Part 2.2: HuggingFace Diffusers library)
- **Part 3**: Celebrity face generation with conditional diffusion (gender conditioning)

All work conducted in Jupyter notebooks with GPU acceleration support (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`).

## Architecture & Key Components

### Core Diffusion Process
1. **Forward diffusion**: Progressively add Gaussian noise over T=1000 timesteps
   - `betas`: noise schedule (typically 1e-4 to 0.01 linearly)
   - `alphas = 1 - betas`
   - `alpha_bars = cumprod(alphas)`: cumulative product for direct sampling
   - Formula: `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise`

2. **Reverse diffusion**: UNet2D predicts noise at each timestep to reconstruct original data

3. **Schedulers**: DDPMScheduler (beta_start=0.00085, beta_end=0.012, scaled_linear schedule)

### Data Processing Patterns
- **Normalization**: All data normalized to [-1, 1] range: `normalize = (data - 0.5) / 0.5`
- **Unnormalization**: `unnormalize = normalized * 0.5 + 0.5` (for visualization)
- **Image sizes**: 28×28 (MNIST), 64×64 (CelebFaces)
- **Batching**: Collate functions handle dynamic tensor stacking (see `part_2_2_huggingface_diffusers.ipynb` and `part_3_1_celeb_face.ipynb`)

### Conditional Diffusion (Part 3)
- Celebrity faces use gender conditioning (1 or 2 → class labels 0 or 1)
- UNet2DModel configured with `num_class_embeds=2` for binary gender classification
- Dataset: ashraq/tmdb-celeb-10k (HuggingFace datasets library)
- Collate functions must subtract 1 from gender labels: `torch.tensor(labels).long() - 1` to convert [1,2] → [0,1]

## Development Workflows

### Running Notebooks
- Notebooks execute top-to-bottom; GPU availability checked at startup
- Clear visualizations with `clear_output(wait=True)` and matplotlib for intermediate steps
- Progress tracking with `tqdm.auto.tqdm` for long loops

### Common Tasks
1. **Data loading**: Use `torchvision.transforms.Compose()` with `ToTensor()` + `Normalize()`; wrap in `DataLoader`
2. **Model inspection**: Print parameter count with `sum(p.numel() for p in model.parameters())`
3. **Noise visualization**: Every N steps (~50 timesteps) to show progression
4. **Model state**: Always move tensors/models to `device` before operations

## Project-Specific Patterns

### Import Organization
```python
# Standard imports (always first)
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, matplotlib.pyplot as plt
# Visualization & utilities
import seaborn as sns
from IPython.display import clear_output
from tqdm.auto import tqdm
# Domain-specific
from diffusers import DDPMScheduler, UNet2DModel  # Part 2+
from datasets import load_dataset  # Part 3
```

### Time Embedding Handling
- Timesteps passed as `torch.tensor(t).long()` or direct indices
- UNet automatically handles time embedding internally via `time_embedding_dim=128`

### Dataset Paths
- MNIST downloads to `mnist/MNIST/raw/` (automatic with `download=True`)
- Keep relative paths consistent across notebooks

## Synthetic Data Patterns (Part 1)
- **Data generation**: Use `sklearn.datasets.make_circles()` or `make_moons()` for 2D benchmarks
- **Normalization**: Standardize synthetic data via `(data - mean) / std` (zero mean, unit std)
- **Visualization**: Every 20 timesteps (~50 total plots) shows diffusion progression clearly
- **Tracking single points**: Store first sample index separately to visualize individual trajectory (shown in green)

## Common Issues & Solutions

### Dataset Loading (Part 3)
- **Issue**: `load_dataset()` fails without internet or with auth. 
  - **Solution**: Install `datasets` library first (`%pip install datasets`), run from environment with internet access
- **Issue**: Gender filtering returns empty dataset
  - **Solution**: Verify dataset structure first: `ds[0].keys()` should contain 'gender' and 'image'

### Normalization Mismatches
- **Forward pass**: Always normalize to [-1, 1] before feeding to scheduler/UNet
- **Visualization**: Unnormalize output with `* 0.5 + 0.5` to recover [0, 1] range for matplotlib
- **Common bug**: Forgetting `.long()` when passing timestep tensors to UNet causes shape mismatches

### GPU/Device Handling
- **Check early**: Cell 1 should print device; if "cpu" appears unexpectedly, verify CUDA availability
- **Tensor moves**: Always `.to(device)` for both model AND input tensors before forward pass
- **Model state**: Use `model.eval()` inside generation loops to disable batch norm/dropout; training uses `model.train()`

## Integration Points & Dependencies
- **PyTorch 2.0+**: Core framework for all computations
- **TorchVision**: Dataset loading (MNIST) and visualization utilities
- **HuggingFace Diffusers**: Pre-built schedulers and UNet2D (Part 2+)
- **HuggingFace Datasets**: Remote dataset loading (Part 3: CelebA)
- **Scikit-learn**: Synthetic data generation (Part 1)
- **tqdm**: Progress bars in loops
- **torchinfo**: Model architecture summary (imported but rarely used)

## Critical File References
- [diffusion_process.ipynb](diffusion_process.ipynb): Unconditional diffusion primer
- [part_2_1_diffusion_from_scratch_pytorch.ipynb](part_2_mnist_diffusion/part_2_1_diffusion_from_scratch_pytorch.ipynb): Manual scheduler implementation
- [part_2_2_huggingface_diffusers.ipynb](part_2_mnist_diffusion/part_2_2_huggingface_diffusers.ipynb): Diffusers library scheduler
- [part_3_1_celeb_face.ipynb](part_3_1_celeb_face.ipynb): Conditional diffusion with real images

---
**Last Updated**: 2026-02-12 | **Focus**: Educational diffusion model implementation progression
