import torch, importlib.util as u
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("torch_geometric", u.find_spec("torch_geometric") is not None,
      "| h5py", u.find_spec("h5py") is not None)
import physicsnemo; print("physicsnemo", physicsnemo.__version__)
