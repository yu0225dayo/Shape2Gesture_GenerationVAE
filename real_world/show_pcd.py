import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_csv_point_cloud(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    has_header = any(ch.isalpha() for ch in first_line)
    if has_header:
        columns = [c.strip().lower() for c in first_line.split(",")]
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    else:
        columns = []
        data = np.loadtxt(csv_path, delimiter=",")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if columns and all(name in columns for name in ("x", "y", "z")):
        x_idx = columns.index("x")
        y_idx = columns.index("y")
        z_idx = columns.index("z")
        points = data[:, [x_idx, y_idx, z_idx]]
    else:
        points = data[:, :3]

    colors = None
    if columns and all(name in columns for name in ("r", "g", "b")):
        r_idx = columns.index("r")
        g_idx = columns.index("g")
        b_idx = columns.index("b")
        colors = data[:, [r_idx, g_idx, b_idx]]
    elif data.shape[1] >= 6 and not columns:
        colors = data[:, 3:6]

    if colors is not None and colors.max() > 1.0:
        colors = colors / 255.0

    return points, colors


def show_csv(csv_path):
    points, colors = load_csv_point_cloud(csv_path)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    if colors is not None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors, marker=".")
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c="tab:blue", marker=".")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(Path(csv_path).name)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    csv_path = input("CSVファイルのパスを入力: ").strip()
    if not csv_path:
        raise SystemExit("CSVパスが空です")

    show_csv(csv_path)
