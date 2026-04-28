#!/usr/bin/env python
"""Patch optional imports for legacy glibc server resume runs.

This rewrites the import header instead of doing a tiny string replacement,
because failed manual patches can leave malformed try/except blocks behind.
"""
from pathlib import Path


REGISTRATION_HEADER = '''"""
Resigration classes for ghd fitting.
Truth mesh is registered to enable opening alignment & differentiable centreline losses during ghd fitting.
"""
import open3d as o3d
import numpy as np
import numpy
import logging
import os
import itertools
import trimesh
import shapely
from utils import utils_registration as u_register
import pickle
from pytorch3d.structures import Meshes
import torch
import sys
from utils.utils import o3d_mesh_to_pytorch3d
import vtk
import pytorch3d as p3d
import igraph as ig
from tqdm import tqdm
try:
    from skeletor.utilities import make_trimesh
except Exception:
    def make_trimesh(mesh, validate=False):
        return mesh
import pyvista as pv
from pytorch3d.io import save_obj, load_objs_as_meshes
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from typing import Dict, List, Optional, Tuple
'''

UTILS_HEADER = '''import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
try:
    import point_cloud_utils as pcu
except Exception:
    pcu = None
import open3d as o3d
import torch
import os
from pytorch3d.structures import Meshes
import pytorch3d
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from tqdm import trange
'''


def rewrite_before(path, marker, header):
    p = Path(path)
    text = p.read_text()
    idx = text.index(marker)
    p.write_text(header + text[idx:])
    print(f"rewrote header in {path}")


rewrite_before("ghd/fitting/registration.py", "\nclass RegistrationwOpeningAlignment", REGISTRATION_HEADER)
rewrite_before("utils/utils.py", "\ndef o3d_mesh_to_pytorch3d", UTILS_HEADER)
