import logging
import os

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except Exception:
    plt = None
    Poly3DCollection = None

import numpy as np


def log_dict_printer(log_dict):
    if not log_dict:
        return

    def _format_value(value):
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    parts = []
    for key, value in log_dict.items():
        parts.append(f"{key}: {_format_value(value)}")
    # Print directly so values are visible even when logging is not configured to INFO.
    print(" | ".join(parts))


def viz_fitting_static(epoch, log_path, warped_mesh, target_mesh, args):
    if plt is None or Poly3DCollection is None:
        return

    def _mesh_to_numpy(mesh):
        if mesh is None:
            return None, None
        try:
            if hasattr(mesh, "verts_list") and hasattr(mesh, "faces_list"):
                verts_list = mesh.verts_list()
                faces_list = mesh.faces_list()
                if len(verts_list) == 0 or len(faces_list) == 0:
                    return None, None
                verts = verts_list[0].detach().cpu().numpy()
                faces = faces_list[0].detach().cpu().numpy()
                return verts, faces
            if hasattr(mesh, "vertices") and hasattr(mesh, "triangles"):
                verts = np.asarray(mesh.vertices, dtype=np.float64)
                faces = np.asarray(mesh.triangles, dtype=np.int64)
                return verts, faces
        except Exception:
            return None, None
        return None, None

    def _set_equal_axes(ax, verts):
        mins = np.min(verts, axis=0)
        maxs = np.max(verts, axis=0)
        center = 0.5 * (mins + maxs)
        span = float(np.max(maxs - mins))
        half = 0.55 * span if span > 0 else 1.0
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.set_box_aspect((1.0, 1.0, 1.0))

    def _draw_mesh(ax, verts, faces, title, color="deepskyblue", alpha=0.55):
        if verts is None or faces is None:
            ax.set_title(f"{title} (unavailable)")
            return
        if verts.ndim != 2 or verts.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
            ax.set_title(f"{title} (invalid)")
            return
        if verts.shape[0] < 3 or faces.shape[0] < 1:
            ax.set_title(f"{title} (empty)")
            return
        if not np.isfinite(verts).all():
            ax.set_title(f"{title} (non-finite)")
            return
        valid = np.all((faces >= 0) & (faces < verts.shape[0]), axis=1)
        faces = faces[valid]
        if faces.shape[0] < 1:
            ax.set_title(f"{title} (faces invalid)")
            return
        tris = verts[faces]
        coll = Poly3DCollection(tris, facecolors=color, edgecolors="none", alpha=alpha)
        ax.add_collection3d(coll)
        _set_equal_axes(ax, verts)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    try:
        warped_v, warped_f = _mesh_to_numpy(warped_mesh)
        target_v, target_f = _mesh_to_numpy(target_mesh)

        fig = plt.figure(figsize=(12, 6), dpi=160)
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        _draw_mesh(ax1, target_v, target_f, "target", color="lightgray", alpha=0.35)
        _draw_mesh(ax2, warped_v, warped_f, "warped", color="deepskyblue", alpha=0.65)
        fig.suptitle(f"Fitting preview | epoch={int(epoch)}")
        fig.tight_layout()
        os.makedirs(log_path, exist_ok=True)
        out_file = os.path.join(log_path, f"fitting_preview_epoch_{int(epoch):06d}.png")
        fig.savefig(out_file, dpi=180, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        return


def viz_fitting_debug(*args, **kwargs):
    # Placeholder: kept for API compatibility.
    return


def update_and_plot_loss_history(
    loss_history,
    log_dict,
    log_path,
    epoch,
    plot_every=100,
    filename="loss_components.png",
    title="Fitting Loss Components",
    ylabel="Loss",
):
    """
    Keep per-term loss history and periodically save a combined loss plot.
    Output file: <log_path>/<filename>
    """
    if not log_dict:
        return loss_history

    for key, value in log_dict.items():
        if key == "epoch":
            continue
        if key not in loss_history:
            loss_history[key] = []
        try:
            loss_history[key].append(float(value))
        except Exception:
            pass

    if plt is None:
        return loss_history

    if epoch % max(1, int(plot_every)) != 0:
        return loss_history

    fig = plt.figure(figsize=(10, 6), dpi=140)
    ax = fig.add_subplot(111)
    for key, values in loss_history.items():
        if len(values) == 0:
            continue
        ax.plot(values, label=key, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(log_path, filename))
    plt.close(fig)
    return loss_history
