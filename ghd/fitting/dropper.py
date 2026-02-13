import torch


class Do_Dropper:
    def __init__(self, args, weights_attention, drop_num=25, drop_rate=0.75):
        self.args = args
        self.weights_attention = weights_attention
        self.drop_num = drop_num
        self.drop_rate = drop_rate
        self.full_index = None
        if torch.is_tensor(weights_attention):
            # weights_attention expected shape [1, N]
            self.full_index = torch.arange(weights_attention.shape[1], device=weights_attention.device)

    def forward(self, epoch):
        # Minimal implementation: no dropping, return full index.
        if self.full_index is None:
            return 1, False
        return self.full_index, False
