# MAI Demo 2425 - Interpretability

## Setup

```bash
conda create -n mai-demo-2425-interpretability python=3.12
conda activate mai-demo-2425-interpretability
```

### GPU on Mac

```python
>>> import torch
>>> torch.cuda.is_available()
False
>>> torch.backends.mps.is_available()
True
>>> device = torch.device("mps")  # Use Metal Performance Shaders
>>> x = torch.tensor([1.0, 2.0, 3.0], device=device)
>>> y = x * 2
>>> print(y)
>>> y = x * 2
>>> print(y)
tensor([2., 4., 6.], device='mps:0')
```
