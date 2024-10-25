# Ref: To verify if Metal Performance Shaders (MPS) is available on the system.
# https://pytorch.org/get-started/locally/
# https://developer.apple.com/metal/pytorch/

import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")