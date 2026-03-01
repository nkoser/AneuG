import copy


class base_loss_weighter:
    def __init__(self, args, glo_loss_weighting, style="constant"):
        self.args = args
        self.base_weighting = copy.deepcopy(glo_loss_weighting or {})
        self.style = (style or "constant").strip().lower()
        self.epochs = int(getattr(args, "epochs", 1) or 1)

    @staticmethod
    def _lerp(start, end, t):
        return (1.0 - t) * float(start) + t * float(end)

    def easy_weighting(self, epoch):
        weights = copy.deepcopy(self.base_weighting)
        if not weights:
            return {}

        if self.style in ("constant", "static"):
            return weights

        if self.style == "strategy_v1_linear":
            # Main training goal: keep data terms stable while gradually
            # relaxing strong shape priors, so deformation can move away from
            # canonical when required by target evidence.
            progress = min(max(float(epoch) / max(self.epochs - 1, 1), 0.0), 1.0)
            decay_targets = {
                "loss_rigid": 0.20,
                "loss_laplacian": 0.35,
                "loss_edge": 0.35,
                "loss_consistency": 0.35,
            }
            for key, end_ratio in decay_targets.items():
                if key in weights:
                    weights[key] = self._lerp(weights[key], weights[key] * end_ratio, progress)
            return weights

        if self.style == "exp_decay":
            gamma = float(getattr(self.args, "weighter_gamma", 0.995))
            scale = gamma ** max(int(epoch), 0)
            for key in weights:
                weights[key] = float(weights[key]) * scale
            return weights

        if self.style == "milestone_decay":
            milestones = list(getattr(self.args, "weighter_milestones", []) or [])
            decay = float(getattr(self.args, "weighter_decay", 0.5))
            multiplier = 1.0
            for milestone in milestones:
                if int(epoch) >= int(milestone):
                    multiplier *= decay
            for key in weights:
                weights[key] = float(weights[key]) * multiplier
            return weights

        return weights
