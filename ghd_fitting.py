import argparse
import copy
from ghd.fitting.fitter import fit_ghd
import os
import logging
from ghd.fitting.registration import RegistrationwOpeningAlignmentwDifferentiableCentreline
import torch
import random
import yaml

# conf
DEFAULTS = {
    "register": True,
    "chk_num": 4,  # number of checkpoints during fitting
    "epochs": 15000,
    "device": "cuda:0",
    "root_template": "./checkpoints/alignment_male_medium",
    "root_target": "./checkpoints/alignment_male_medium",
    "name_canonical": "canonical_typeB",
    "name_target": "AN213_full_clean",
    "viz_freq": 200,
    "chk_freq": None,
    "log_freq": 100,
    "lr": 0.00075,
    "lr_scheduler": "step",  # step | plateau | none
    "step_size": 2500,
    "gamma": 0.75,
    "plateau_factor": 0.5,
    "plateau_patience": 300,
    "plateau_threshold": 1e-4,
    "plateau_cooldown": 100,
    "min_lr": 1e-6,
    "early_stopping": 0,
    "early_stopping_patience": 1200,
    "early_stopping_min_delta": 1e-5,
    "early_stopping_min_epochs": 2000,
    "num_op": 3,
    "num_Basis": 13 ** 2,
    "mix_lap_weights": [1.0, 0.1, 0.1],
    "sample_num": int(2.5e5),
    "op_sample_num": int(1e3),
    "op_clean_threshold": 0.2,
    "op_bold": 0,
    "save_root": "./checkpoints/ghb_fitting_male_medium",
    "meta": "cut_446_decreasing_centrelineloss_10",
    "num_sp": 2,
    "do_dpi": 4,
    "do_style": "number_control_v2",
    "do_loss_type": "dice_loss_attention",
    "use_do_dropper": 0,
    "attention_max_w": 3.0,
    "attention_smooth": 0.02,
    "do_number": 25000,
    "weighter_style": "strategy_v1_linear",
    "mesh_filename": "part_aligned.obj",
    "loss_weighting": {
        "loss_do": 1.0,
        "loss_p0": 1.0 * 1,
        "loss_n1": 0.8 * 1,
        "loss_laplacian": 0.1,
        "loss_edge": 0.1,
        "loss_consistency": 0.1,
        "loss_rigid": 100.0,
        "loss_openings_p": 5,
        "loss_openings_n": 0.1,
        "loss_diff_centreline": 10.0,
    },
}


def deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path):
    if not path:
        return {}
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def compute_defaults(cfg):
    cfg = copy.deepcopy(cfg)
    if cfg.get("chk_freq") is None:
        cfg["chk_freq"] = round(cfg["epochs"] / cfg["chk_num"])
    return cfg


parser = argparse.ArgumentParser("ghb_fitting_oa")
parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
parser.add_argument("--register", type=int, default=1)
parser.add_argument("--chk_num", type=int, default=DEFAULTS["chk_num"])
parser.add_argument('--device', type=str, default=DEFAULTS["device"]) # device to use for fitting
parser.add_argument('--root_template', type=str, default=DEFAULTS["root_template"]) # root directory for template meshes, should contain a subdirectory named as args.name_canonical
parser.add_argument('--root_target', type=str, default=DEFAULTS["root_target"]) # root directory for target meshes, should contain subdirectories named as args.name_target, each of which contains the registered mesh and the cleaned centreline for the corresponding case. The registered mesh should be named "mesh_target_p3d.pt", and the cleaned centreline should be named "centreline_clean.pt". If registration is not performed, these files will be generated during registration.
parser.add_argument('--name_canonical', type=str, default=DEFAULTS["name_canonical"]) # name of the canonical template mesh, should be a subdirectory of args.root_template, and should contain a file named "mesh_template_p3d.pt" for the template mesh. If registration is not performed, this file will be generated during registration.
parser.add_argument('--name_target', type=str, default=DEFAULTS["name_target"]) # name of the target mesh, should be a subdirectory of args.root_target, and should contain the registered mesh and the cleaned centreline for the corresponding case if registration is performed. If registration is not performed, these files will be generated during registration.
parser.add_argument('--viz_freq', type=int, default=DEFAULTS["viz_freq"]) # frequency for visualization during fitting, in number of epochs
parser.add_argument('--chk_freq', type=int, default=None) # frequency for saving checkpoints during fitting, in number of epochs. Should be set such that epochs / chk_freq = chk_num, i.e., the total number of checkpoints saved during fitting is equal to chk_num.
parser.add_argument('--log_freq', type=int, default=DEFAULTS["log_freq"]) # frequency for logging during fitting, in number of epochs
parser.add_argument('--lr', type=float, default=DEFAULTS["lr"]) # learning rate for fitting, can be adjusted based on the scale of the meshes and the number of epochs. A smaller learning rate may be needed for larger meshes or more epochs to ensure stable convergence.
parser.add_argument('--lr_scheduler', type=str, default=DEFAULTS["lr_scheduler"]) # step | plateau | none
parser.add_argument('--step_size', type=int, default=DEFAULTS["step_size"]) # StepLR step size
parser.add_argument('--gamma', type=float, default=DEFAULTS["gamma"]) # StepLR gamma
parser.add_argument('--plateau_factor', type=float, default=DEFAULTS["plateau_factor"]) # ReduceLROnPlateau factor
parser.add_argument('--plateau_patience', type=int, default=DEFAULTS["plateau_patience"]) # ReduceLROnPlateau patience (epochs)
parser.add_argument('--plateau_threshold', type=float, default=DEFAULTS["plateau_threshold"]) # ReduceLROnPlateau threshold
parser.add_argument('--plateau_cooldown', type=int, default=DEFAULTS["plateau_cooldown"]) # ReduceLROnPlateau cooldown (epochs)
parser.add_argument('--min_lr', type=float, default=DEFAULTS["min_lr"]) # minimum learning rate
parser.add_argument('--early_stopping', type=int, default=DEFAULTS["early_stopping"]) # 1 enables early stopping
parser.add_argument('--early_stopping_patience', type=int, default=DEFAULTS["early_stopping_patience"]) # epochs with no sufficient improvement
parser.add_argument('--early_stopping_min_delta', type=float, default=DEFAULTS["early_stopping_min_delta"]) # minimal improvement to reset patience
parser.add_argument('--early_stopping_min_epochs', type=int, default=DEFAULTS["early_stopping_min_epochs"]) # warmup epochs before early stopping can trigger
parser.add_argument('--num_op', type=int, default=DEFAULTS["num_op"]) # number of operators to use for fitting, can be adjusted based on the complexity of the meshes and the desired level of detail in the fitting. More operators may allow for a better fit but may also increase the risk of overfitting and require more careful tuning of the loss weights.
parser.add_argument('--num_Basis', type=int, default=DEFAULTS["num_Basis"]) # number of basis functions to use for fitting, can be adjusted based on the desired level of detail in the fitting.
parser.add_argument('--mix_lap_weights', type=list, default=DEFAULTS["mix_lap_weights"]) # weights for the mixed Laplacian regularization terms, can be adjusted to balance the smoothness and fidelity of the fitting.
parser.add_argument('--sample_num', type=int, default=DEFAULTS["sample_num"]) # number of samples to use for fitting, can be adjusted based on the scale of the meshes and the desired level of detail in the fitting.
parser.add_argument('--op_sample_num', type=int, default=DEFAULTS["op_sample_num"]) # number of operator samples to use for fitting, can be adjusted based on the complexity of the meshes and the desired level of detail in the fitting.
parser.add_argument('--op_clean_threshold', type=float, default=DEFAULTS["op_clean_threshold"]) # threshold for cleaning operator samples, can be adjusted based on the desired level of detail in the fitting.
parser.add_argument('--op_bold', type=int, default=DEFAULTS["op_bold"])  # confidence that trimesh offers uniform mesh normal directions
parser.add_argument('--save_root', type=str, default=DEFAULTS["save_root"]) # root directory for saving fitting results, should contain subdirectories named as args.name_target, each of which contains the fitting results for the corresponding case. The fitting results will be saved in a subdirectory named as args.meta under each target subdirectory, and the checkpoint for each case will be saved as "ghd_fitting_checkpoint.pkl" under the corresponding meta subdirectory.
parser.add_argument('--meta', type=str, default=DEFAULTS["meta"]) # meta name for saving fitting results, can be adjusted to distinguish different fitting settings. The fitting results will be saved in a subdirectory named as args.meta under each target subdirectory.
parser.add_argument('--epochs', type=int, default=DEFAULTS["epochs"]) # number of epochs for fitting, can be adjusted based on the desired level of detail in the fitting and the computational resources available. More epochs may allow for a better fit but may also increase the risk of overfitting and require more careful tuning of the learning rate and loss weights.
parser.add_argument('--num_sp', type=int, default=DEFAULTS["num_sp"]) # number of spectral components to use for fitting, can be adjusted based on the desired level of detail in the fitting. More spectral components may allow for a better fit but may also increase the risk of overfitting and require more careful tuning of the loss weights.
parser.add_argument('--do_dpi', type=int, default=DEFAULTS["do_dpi"]) # number of iterations for the differentiable point cloud registration, can be adjusted based on the desired level of detail in the fitting and the computational resources available. More iterations may allow for a better fit but may also increase the risk of overfitting and require more careful tuning of the learning rate and loss weights.
parser.add_argument('--do_style', type=str, default=DEFAULTS["do_style"]) # style for the differentiable point cloud registration, can be adjusted based on the desired level of detail in the fitting and the characteristics of the meshes. Different styles may use different loss functions and optimization strategies for the registration, and may require different tuning of the learning rate and loss weights.
parser.add_argument('--do_loss_type', type=str, default=DEFAULTS["do_loss_type"]) # loss type for the differentiable point cloud registration, can be adjusted based on the desired level of detail in the fitting and the characteristics of the meshes. Different loss types may have different sensitivities to outliers and may require different tuning of the learning rate and loss weights.
parser.add_argument('--use_do_dropper', type=int, default=DEFAULTS["use_do_dropper"]) # whether to use a dropper for the differentiable point cloud registration, can be adjusted based on the desired level of detail in the fitting and the characteristics of the meshes. Using a dropper may help to improve the robustness of the registration by reducing the influence of outliers, but may also require more careful tuning of the learning rate and loss weights.
parser.add_argument('--attention_max_w', type=float, default=DEFAULTS["attention_max_w"]) # maximum weight for the attention mechanism in the differentiable point cloud registration, can be adjusted based on the desired level of detail in the fitting and the characteristics of the meshes. A larger maximum weight may allow for a better fit but may also increase the risk of overfitting and require more careful tuning of the learning rate and loss weights.
parser.add_argument('--attention_smooth', type=float, default=DEFAULTS["attention_smooth"]) # smoothing factor for the attention mechanism in the differentiable point cloud registration, can be adjusted based on the desired level of detail in the fitting and the characteristics of the meshes. A larger smoothing factor may help to improve the robustness of the registration by reducing the influence of outliers, but may also require more careful tuning of the learning rate and loss weights.
parser.add_argument('--do_number', type=int, default=DEFAULTS["do_number"]) # number of points to use for the differentiable point cloud registration, can be adjusted based on the desired level of detail in the fitting and the computational resources available. More points may allow for a better fit but may also increase the risk of overfitting and require more careful tuning of the learning rate and loss weights.
parser.add_argument('--weighter_style', type=str, default=DEFAULTS["weighter_style"]) # style for the weighter in the differentiable point cloud registration, can be adjusted based on the desired level of detail in the fitting and the characteristics of the meshes. Different styles may use different weighting strategies and may require different tuning of the learning rate and loss weights.
parser.add_argument('--mesh_filename', type=str, default=DEFAULTS["mesh_filename"]) # mesh filename inside each target subfolder (e.g., part_aligned.obj). If not found, falls back to <root>/<label>.obj

pre_args, _ = parser.parse_known_args()
config_path = pre_args.config
if config_path is None:
    # Use repository-local config by default when available.
    inferred_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ghd_fitting_config.yaml")
    if os.path.isfile(inferred_config):
        config_path = inferred_config
config_data = load_config(config_path)
merged = compute_defaults(deep_merge(DEFAULTS, config_data))
parser.set_defaults(**{k: merged[k] for k in vars(pre_args).keys() if k in merged})
args = parser.parse_args()

register = bool(args.register)
chk_num = int(args.chk_num)
if args.chk_freq is None:
    args.chk_freq = round(args.epochs / chk_num)

loss_weighting = copy.deepcopy(merged.get("loss_weighting", DEFAULTS["loss_weighting"]))

if not os.path.isdir(args.root_target):
    raise FileNotFoundError(
        f"Target root directory not found: '{args.root_target}'. "
        f"Pass --root_target or --config pointing to a valid path."
    )

label_list = [label for label in os.listdir(args.root_target) if
              os.path.isdir(os.path.join(args.root_target, label)) and label != args.name_canonical]
exclude_list = []
random.shuffle(label_list)
mesh_filename = getattr(args, "mesh_filename", None)
if mesh_filename:
    filtered = []
    for label in label_list:
        mesh_path = os.path.join(args.root_target, label, mesh_filename)
        if os.path.isfile(mesh_path):
            filtered.append(label)
        else:
            print(f"Skipping case {label}: missing {mesh_filename}")
    label_list = filtered

# perform registration for opa classes and differentiable centrelines
if register:
    for label in label_list:
        args.name_target = label
        if (os.path.exists(os.path.join(args.root_target, args.name_target, "opa_checkpoint.pkl")) and os.path.exists(os.path.join(args.root_target, args.name_target, "diff_centreline_checkpoint.pkl"))):
            print("Registration for case {} has been found, skipping".format(label))
        else:
            print("Registration for case {} not found".format(label))
            target = RegistrationwOpeningAlignmentwDifferentiableCentreline(args, args.root_target, args.name_target)
            target.load_checkpoint_opa(None, redo=False)
            target.load_checkpoint_centreline(None, redo=False)
            norm_target = torch.max(torch.norm(getattr(target, "mesh_target_p3d").verts_packed(), dim=-1)).detach().item()
            target.class_normalize(norm=norm_target)
            target.centreline_clean(radius=0.5 / norm_target)
            target.visualize_centreline(norm_target)

# perform ghd fitting
for label in label_list:
    args.name_target = label
    print('Now performing ghd fitting for case {}'.format(label))
    case_log_root = os.path.join(args.save_root, args.name_target, args.meta)
    done_markers = [
        os.path.join(case_log_root, "ghb_fitting_checkpoint.pkl"),
        os.path.join(case_log_root, "ghd_fitting_checkpoint.pkl"),  # backward compatibility
    ]
    if not any(os.path.exists(path) for path in done_markers):
        copied_loss_weighting = loss_weighting.copy()
        fit_ghd(args, copied_loss_weighting, hard_normalize=True, keep_size=True)
    else:
        print("Skipping ghd fitting for case {}".format(label))
