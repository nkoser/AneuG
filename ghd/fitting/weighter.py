import copy


class base_loss_weighter:
    def __init__(self, args, glo_loss_weighting, style="constant"):
        self.args = args
        self.glo_loss_weighting = copy.deepcopy(glo_loss_weighting or {})
        self.style = style or "constant"

    def easy_weighting(self, epoch):
        # Minimal implementation: keep weights constant.
        # This preserves existing behavior if no custom schedule is required.
        return copy.deepcopy(self.glo_loss_weighting)
