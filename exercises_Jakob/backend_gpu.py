from deepxde.backend import pytorch
# import torch as pytorch

# # Check that MPS is available
# if not pytorch.backends.mps.is_available():
#     if not pytorch.backends.mps.is_built():
#         print("MPS not available because the current PyTorch install was not "
#               "built with MPS enabled.")
#     else:
#         print("MPS not available because the current MacOS version is not 12.3+ "
#               "and/or you do not have an MPS-enabled device on this machine.")


# else:
#     print("MPS is available and compiled")
#     mps_device = pytorch.device("mps")