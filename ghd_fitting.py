import argparse
from ghd.fitting.fitter import fit_ghd
import os
import logging
from ghd.fitting.registration import RegistrationwOpeningAlignmentwDifferentiableCentreline
import torch
import random

# conf 
epochs = 15000
chk_num = 4  # number of checkpoints during fitting
register = True  
parser = argparse.ArgumentParser("ghb_fitting_oa")
parser.add_argument('--device', type=str, default='cuda:0') # device to use for fitting
parser.add_argument('--root_template', type=str, default='./checkpoints/alignment_male_medium') # root directory for template meshes, should contain a subdirectory named as args.name_canonical
parser.add_argument('--root_target', type=str, default='./checkpoints/alignment_male_medium') # root directory for target meshes, should contain subdirectories named as args.name_target, each of which contains the registered mesh and the cleaned centreline for the corresponding case. The registered mesh should be named "mesh_target_p3d.pt", and the cleaned centreline should be named "centreline_clean.pt". If registration is not performed, these files will be generated during registration.
parser.add_argument('--name_canonical', type=str, default='canonical_typeB') # name of the canonical template mesh, should be a subdirectory of args.root_template, and should contain a file named "mesh_template_p3d.pt" for the template mesh. If registration is not performed, this file will be generated during registration.
parser.add_argument('--name_target', type=str, default='AN213_full_clean') # name of the target mesh, should be a subdirectory of args.root_target, and should contain the registered mesh and the cleaned centreline for the corresponding case if registration is performed. If registration is not performed, these files will be generated during registration.
parser.add_argument('--viz_freq', type=int, default=200) # frequency for visualization during fitting, in number of epochs
parser.add_argument('--chk_freq', type=int, default=round(epochs / chk_num)) # frequency for saving checkpoints during fitting, in number of epochs. Should be set such that epochs / chk_freq = chk_num, i.e., the total number of checkpoints saved during fitting is equal to chk_num.
parser.add_argument('--log_freq', type=int, default=100) # frequency for logging during fitting, in number of epochs
parser.add_argument('--lr', type=float, default=0.00075) # learning rate for fitting, can be adjusted based on the scale of the meshes and the number of epochs. A smaller learning rate may be needed for larger meshes or more epochs to ensure stable convergence.
parser.add_argument('--num_op', type=int, default=3) # number of operators to use for fitting, can be adjusted based on the complexity of the meshes and the desired level of detail in the fitting. More operators may allow for a better fit but may also increase the risk of overfitting and require more careful tuning of the loss weights.
parser.add_argument('--num_Basis', type=int, default=13 ** 2) # number of basis functions to use for fitting, can be adjusted based on the desired level of detail in the fitting.
parser.add_argument('--mix_lap_weights', type=list, default=[1.0, 0.1, 0.1]) # weights for the mixed Laplacian regularization terms, can be adjusted to balance the smoothness and fidelity of the fitting.
parser.add_argument('--sample_num', type=int, default=int(2.5e5)) # number of samples to use for fitting, can be adjusted based on the scale of the meshes and the desired level of detail in the fitting.
parser.add_argument('--op_sample_num', type=int, default=int(1e3)) # number of operator samples to use for fitting, can be adjusted based on the complexity of the meshes and the desired level of detail in the fitting.
parser.add_argument('--op_clean_threshold', type=float, default=0.2) # threshold for cleaning operator samples, can be adjusted based on the desired level of detail in the fitting.
parser.add_argument('--op_bold', type=int, default=0)  # confidence that trimesh offers uniform mesh normal directions
parser.add_argument('--save_root', type=str, default='./checkpoints/ghb_fitting_male_medium') # root directory for saving fitting results, should contain subdirectories named as args.name_target, each of which contains the fitting results for the corresponding case. The fitting results will be saved in a subdirectory named as args.meta under each target subdirectory, and the checkpoint for each case will be saved as "ghd_fitting_checkpoint.pkl" under the corresponding meta subdirectory.
parser.add_argument('--meta', type=str, default='cut_446_decreasing_centrelineloss_10') # meta name for saving fitting results, can be adjusted to distinguish different fitting settings. The fitting results will be saved in a subdirectory named as args.meta under each target subdirectory.
parser.add_argument('--epochs', type=int, default=int(epochs)) # number of epochs for fitting, can be adjusted based on the desired level of detail in the fitting and the computational resources available. More epochs may allow for a better fit but may also increase the risk of overfitting and require more careful tuning of the learning rate and loss weights.
parser.add_argument('--num_sp', type=int, default=2) # number of spectral components to use for fitting, can be adjusted based on the desired level of detail in the fitting. More spectral components may allow for a better fit but may also increase the risk of overfitting and require more careful tuning of the loss weights.
parser.add_argument('--do_dpi', type=int, default=4) # number of iterations for the differentiable point cloud registration, can be adjusted based on the desired level of detail in the fitting and the computational resources available. More iterations may allow for a better fit but may also increase the risk of overfitting and require more careful tuning of the learning rate and loss weights.
parser.add_argument('--do_style', type=str, default='number_control_v2') # style for the differentiable point cloud registration, can be adjusted based on the desired level of detail in the fitting and the characteristics of the meshes. Different styles may use different loss functions and optimization strategies for the registration, and may require different tuning of the learning rate and loss weights.
parser.add_argument('--do_loss_type', type=str, default='dice_loss_attention') # loss type for the differentiable point cloud registration, can be adjusted based on the desired level of detail in the fitting and the characteristics of the meshes. Different loss types may have different sensitivities to outliers and may require different tuning of the learning rate and loss weights.
parser.add_argument('--use_do_dropper', type=int, default=0) # whether to use a dropper for the differentiable point cloud registration, can be adjusted based on the desired level of detail in the fitting and the characteristics of the meshes. Using a dropper may help to improve the robustness of the registration by reducing the influence of outliers, but may also require more careful tuning of the learning rate and loss weights.
parser.add_argument('--attention_max_w', type=float, default=3.0) # maximum weight for the attention mechanism in the differentiable point cloud registration, can be adjusted based on the desired level of detail in the fitting and the characteristics of the meshes. A larger maximum weight may allow for a better fit but may also increase the risk of overfitting and require more careful tuning of the learning rate and loss weights.
parser.add_argument('--attention_smooth', type=float, default=0.02) # smoothing factor for the attention mechanism in the differentiable point cloud registration, can be adjusted based on the desired level of detail in the fitting and the characteristics of the meshes. A larger smoothing factor may help to improve the robustness of the registration by reducing the influence of outliers, but may also require more careful tuning of the learning rate and loss weights.
parser.add_argument('--do_number', type=int, default=25000) # number of points to use for the differentiable point cloud registration, can be adjusted based on the desired level of detail in the fitting and the computational resources available. More points may allow for a better fit but may also increase the risk of overfitting and require more careful tuning of the learning rate and loss weights.
parser.add_argument('--weighter_style', type=str, default='strategy_v1_linear') # style for the weighter in the differentiable point cloud registration, can be adjusted based on the desired level of detail in the fitting and the characteristics of the meshes. Different styles may use different weighting strategies and may require different tuning of the learning rate and loss weights.
args = parser.parse_args()

loss_weighting = {
    'loss_do': 1.0,
    'loss_p0': 1.0 * 1, 'loss_n1': 0.8 * 1,
    'loss_laplacian': 0.1, 'loss_edge': 0.1, 'loss_consistency': 0.1,
    'loss_rigid': 100.0,
    'loss_openings_p': 5,
    'loss_openings_n': 0.1,
    'loss_diff_centreline': 10.0
}

label_list = [label for label in os.listdir(args.root_target) if
              os.path.isdir(os.path.join(args.root_target, label)) and label != args.name_canonical]
exclude_list = []
random.shuffle(label_list)

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
    if not os.path.exists(os.path.join(args.save_root, args.name_target, args.meta, "ghd_fitting_checkpoint.pkl")):
        copied_loss_weighting = loss_weighting.copy()
        fit_ghd(args, copied_loss_weighting, hard_normalize=True, keep_size=True)
    else:
        print("Skipping ghd fitting for case {}".format(label))