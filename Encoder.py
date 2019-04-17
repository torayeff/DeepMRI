import torch.nn
import Conv3DLSTM


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()