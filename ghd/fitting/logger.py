import logging
import os

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


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
    # Placeholder: visualization is optional for headless runs.
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
