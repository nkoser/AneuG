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


def viz_fitting_static(
    epoch,
    log_path,
    warped_mesh,
    target_mesh,
    args,
    target_openings=None,
    warped_openings=None,
    warped_opening_points=None,
    warped_opening_normal_points=None,
    warped_opening_normals=None,
    target_opening_normal_points=None,
    target_opening_normals=None,
):
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

    def _set_equal_axes(ax, mins, maxs):
        center = 0.5 * (mins + maxs)
        span = float(np.max(maxs - mins))
        half = 0.55 * span if span > 0 else 1.0
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.set_box_aspect((1.0, 1.0, 1.0))

    def _draw_mesh(ax, verts, faces, title, color="deepskyblue", alpha=0.55, shared_bounds=None):
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
        if shared_bounds is None:
            mins = np.min(verts, axis=0)
            maxs = np.max(verts, axis=0)
        else:
            mins, maxs = shared_bounds
        _set_equal_axes(ax, mins, maxs)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    def _iter_openings(openings):
        if openings is None:
            return []
        if isinstance(openings, (list, tuple)):
            return list(openings)
        return [openings]

    def _opening_display_normal(verts, faces):
        if verts is None or faces is None or faces.shape[0] == 0:
            return None
        try:
            tris = verts[faces]
            e1 = tris[:, 1] - tris[:, 0]
            e2 = tris[:, 2] - tris[:, 0]
            face_normals = np.cross(e1, e2)
            norm = np.linalg.norm(face_normals, axis=1, keepdims=True)
            valid = norm[:, 0] > 1e-10
            if not np.any(valid):
                return None
            normal = np.sum(face_normals[valid], axis=0)
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm <= 1e-10:
                return None
            return normal / normal_norm
        except Exception:
            return None

    def _draw_opening_faces(
        ax,
        openings=None,
        color="crimson",
        alpha=0.16,
        edgecolor=None,
        linewidth=0.25,
        shared_bounds=None,
        lift_scale=0.0,
    ):
        span = 1.0
        if shared_bounds is not None:
            mins, maxs = shared_bounds
            span = max(float(np.max(maxs - mins)), 1.0)
        for opening in _iter_openings(openings):
            verts, faces = _mesh_to_numpy(opening)
            if verts is None or faces is None:
                continue
            if verts.ndim != 2 or faces.ndim != 2 or verts.shape[0] < 3 or faces.shape[0] < 1:
                continue
            valid = np.all((faces >= 0) & (faces < verts.shape[0]), axis=1)
            faces = faces[valid]
            if faces.shape[0] < 1:
                continue
            if lift_scale > 0.0:
                opening_normal = _opening_display_normal(verts, faces)
                if opening_normal is not None:
                    verts = verts + opening_normal[None, :] * (lift_scale * span)
            tris = verts[faces]
            coll = Poly3DCollection(
                tris,
                facecolors=color,
                edgecolors=edgecolor if edgecolor is not None else color,
                linewidths=linewidth,
                alpha=alpha,
            )
            ax.add_collection3d(coll)

    def _points_to_numpy(points):
        if points is None:
            return None
        try:
            if hasattr(points, "detach"):
                points = points.detach().cpu().numpy()
            else:
                points = np.asarray(points)
        except Exception:
            return None
        points = np.asarray(points, dtype=np.float64)
        if points.ndim == 3 and points.shape[0] == 1:
            points = points[0]
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            return None
        return points

    def _boundary_points_from_mesh(mesh):
        if mesh is None:
            return None
        verts, faces = _mesh_to_numpy(mesh)
        if verts is None or verts.ndim != 2 or verts.shape[0] == 0:
            return None
        if faces is None or faces.ndim != 2 or faces.shape[0] == 0:
            return verts
        try:
            edges = np.concatenate(
                (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]),
                axis=0,
            )
            edges = np.sort(edges, axis=1)
            unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
            boundary_edges = unique_edges[counts == 1]
            if boundary_edges.shape[0] == 0:
                return verts
            boundary_idx = np.unique(boundary_edges.reshape(-1))
            return verts[boundary_idx]
        except Exception:
            return verts

    def _select_opening_marker_points(source, num_points=4, source_is_points=False):
        pts = _points_to_numpy(source) if source_is_points else _boundary_points_from_mesh(source)
        if pts is None or pts.ndim != 2 or pts.shape[0] == 0:
            return None
        if pts.shape[0] <= num_points:
            return pts
        try:
            idx = np.linspace(0, pts.shape[0] - 1, num_points, dtype=int)
            return pts[idx]
        except Exception:
            idx = np.linspace(0, pts.shape[0] - 1, num_points, dtype=int)
            return pts[idx]

    def _draw_opening_markers(ax, openings=None, opening_points=None, color="crimson"):
        if opening_points:
            for pts in opening_points:
                marker_pts = _points_to_numpy(pts)
                if marker_pts is None or marker_pts.shape[0] == 0:
                    continue
                ax.scatter(
                    marker_pts[:, 0],
                    marker_pts[:, 1],
                    marker_pts[:, 2],
                    s=4,
                    c=color,
                    marker="o",
                    alpha=0.9,
                    edgecolors="none",
                    depthshade=False,
                )
            return
        if not openings:
            return
        for opening in openings:
            marker_pts = _select_opening_marker_points(opening, num_points=4, source_is_points=False)
            if marker_pts is None or marker_pts.shape[0] == 0:
                continue
            ax.scatter(
                marker_pts[:, 0],
                marker_pts[:, 1],
                marker_pts[:, 2],
                s=22,
                c=color,
                marker="o",
                edgecolors="white",
                linewidths=0.4,
                depthshade=False,
            )

    def _draw_opening_normals(ax, opening_points=None, opening_normals=None, color="crimson", shared_bounds=None):
        if not opening_points or not opening_normals:
            return
        if shared_bounds is not None:
            mins, maxs = shared_bounds
            arrow_length = max(float(np.max(maxs - mins)) * 0.05, 1e-3)
        else:
            arrow_length = 0.05
        for pts_src, nrm_src in zip(opening_points, opening_normals):
            pts = _points_to_numpy(pts_src)
            nrms = _points_to_numpy(nrm_src)
            if pts is None or nrms is None or pts.shape[0] == 0 or nrms.shape[0] == 0:
                continue
            count = min(pts.shape[0], nrms.shape[0])
            pts = pts[:count]
            nrms = nrms[:count]
            if count > 48:
                idx = np.linspace(0, count - 1, 48, dtype=int)
                pts = pts[idx]
                nrms = nrms[idx]
            lengths = np.linalg.norm(nrms, axis=1, keepdims=True)
            lengths = np.clip(lengths, 1e-8, None)
            nrms = nrms / lengths
            ax.quiver(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                nrms[:, 0],
                nrms[:, 1],
                nrms[:, 2],
                length=arrow_length,
                normalize=True,
                color=color,
                linewidth=0.5,
                arrow_length_ratio=0.25,
            )

    try:
        warped_v, warped_f = _mesh_to_numpy(warped_mesh)
        target_v, target_f = _mesh_to_numpy(target_mesh)

        shared_bounds = None
        bounds_verts = [v for v in (target_v, warped_v) if v is not None and v.size > 0]
        if bounds_verts:
            all_v = np.concatenate(bounds_verts, axis=0)
            shared_bounds = (np.min(all_v, axis=0), np.max(all_v, axis=0))

        fig = plt.figure(figsize=(18, 6), dpi=160)
        ax1 = fig.add_subplot(1, 3, 1, projection="3d")
        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        _draw_mesh(ax1, target_v, target_f, "target", color="lightgray", alpha=0.55, shared_bounds=shared_bounds)
        _draw_mesh(ax2, warped_v, warped_f, "warped", color="deepskyblue", alpha=0.65, shared_bounds=shared_bounds)
        _draw_mesh(ax3, target_v, target_f, "overlay", color="lightgray", alpha=0.18, shared_bounds=shared_bounds)
        _draw_mesh(ax3, warped_v, warped_f, "overlay", color="deepskyblue", alpha=0.65, shared_bounds=shared_bounds)
        _draw_opening_faces(
            ax1,
            openings=target_openings,
            color="silver",
            alpha=0.45,
            edgecolor="black",
            linewidth=0.7,
            shared_bounds=shared_bounds,
            lift_scale=0.01,
        )
        _draw_opening_faces(
            ax2,
            openings=warped_openings,
            color="gold",
            alpha=0.5,
            edgecolor="darkred",
            linewidth=0.8,
            shared_bounds=shared_bounds,
            lift_scale=0.01,
        )
        _draw_opening_faces(
            ax3,
            openings=target_openings,
            color="silver",
            alpha=0.28,
            edgecolor="black",
            linewidth=0.55,
            shared_bounds=shared_bounds,
            lift_scale=0.01,
        )
        _draw_opening_faces(
            ax3,
            openings=warped_openings,
            color="gold",
            alpha=0.34,
            edgecolor="darkred",
            linewidth=0.65,
            shared_bounds=shared_bounds,
            lift_scale=0.01,
        )
        _draw_opening_markers(ax2, openings=warped_openings, opening_points=warped_opening_points, color="crimson")
        _draw_opening_markers(ax3, openings=warped_openings, opening_points=warped_opening_points, color="crimson")
        _draw_opening_normals(
            ax1,
            opening_points=target_opening_normal_points,
            opening_normals=target_opening_normals,
            color="dimgray",
            shared_bounds=shared_bounds,
        )
        _draw_opening_normals(
            ax2,
            opening_points=warped_opening_normal_points,
            opening_normals=warped_opening_normals,
            color="crimson",
            shared_bounds=shared_bounds,
        )
        _draw_opening_normals(
            ax3,
            opening_points=target_opening_normal_points,
            opening_normals=target_opening_normals,
            color="dimgray",
            shared_bounds=shared_bounds,
        )
        _draw_opening_normals(
            ax3,
            opening_points=warped_opening_normal_points,
            opening_normals=warped_opening_normals,
            color="crimson",
            shared_bounds=shared_bounds,
        )
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
