import copy


class base_loss_weighter:
    def __init__(
        self,
        args,
        glo_loss_weighting,
        style="constant",
        case_scale_overrides=None,
        warmup_epochs_override=None,
        fit_start_epoch=0,
    ):
        self.args = args
        self.base_weighting = copy.deepcopy(glo_loss_weighting or {})
        self.style = (style or "constant").strip().lower()
        self.epochs = int(getattr(args, "epochs", 1) or 1)
        self.case_scale_overrides = copy.deepcopy(case_scale_overrides or {})
        self.opening_warmup_epochs = int(
            warmup_epochs_override
            if warmup_epochs_override is not None
            else (getattr(args, "opening_warmup_epochs", 0) or 0)
        )
        self.fit_start_epoch = int(fit_start_epoch or 0)

    @staticmethod
    def _lerp(start, end, t):
        return (1.0 - t) * float(start) + t * float(end)

    def _scale_towards_one(self, start_scale, epoch):
        warmup_epochs = max(int(self.opening_warmup_epochs), 0)
        if warmup_epochs <= 0:
            return 1.0
        local_epoch = max(int(epoch) - int(self.fit_start_epoch), 0)
        t = min(max(float(local_epoch) / max(warmup_epochs, 1), 0.0), 1.0)
        return self._lerp(start_scale, 1.0, t)

    def _apply_opening_warmup(self, weights, epoch):
        key_to_default = {
            "loss_openings_p": float(getattr(self.args, "opening_start_scale", 1.0)),
            "loss_openings_surface_p": float(getattr(self.args, "opening_surface_start_scale", 1.0)),
            "loss_openings_n": float(getattr(self.args, "opening_normal_start_scale", 1.0)),
            "loss_openings_plane": float(getattr(self.args, "opening_plane_start_scale", 1.0)),
            "loss_openings_rim_curvature": float(getattr(self.args, "opening_rim_start_scale", 1.0)),
            "loss_diff_centreline": float(getattr(self.args, "centreline_start_scale", 1.0)),
        }
        for key, default_scale in key_to_default.items():
            if key not in weights:
                continue
            start_scale = float(self.case_scale_overrides.get(key, default_scale))
            weights[key] = float(weights[key]) * self._scale_towards_one(start_scale, epoch)
        return weights

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

        if self.style in ("strategy_v2_robust_opening", "strategy_v2_robust"):
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
            return self._apply_opening_warmup(weights, epoch)

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
