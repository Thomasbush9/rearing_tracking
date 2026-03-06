import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import xarray as xr 
from movement.io.load_poses import from_file
from typing import List
import argparse

def project_point_onto_line(P, A, B):
    """
    Projects a point P (can be a single point or multiple points) onto a line defined by points A and B.
    Args:
        P (np.array): The point(s) to project, shape (..., d) where d=2 or 3 (e.g., [x, y] or [x, y, z]).
        A (np.array): A point on the line, shape (d,)
        B (np.array): Another point on the line, shape (d,)
    Returns:
        np.array: The projected point(s) Q, with same leading shape as P.
    """
    P = np.array(P)
    A = np.array(A)
    B = np.array(B)
    AB = B - A  # (d,)
    AP = P - A  # (..., d)

    l2 = np.sum(AB ** 2)
    if l2 == 0:
        raise ValueError("Line points A and B must be distinct.")

    # Compute t for each point
    t = np.dot(AP, AB) / l2  # (...,)

    # Ensure shape compatibility for broadcasting
    Q = A + np.expand_dims(t, axis=-1) * AB  # (..., d)
    return Q


def _assert_arena_shape(arena: np.ndarray) -> None:
    """Arena must be (8, 2) from central view position."""
    assert arena.shape == (8, 2), f"arena must be (8, 2), got {arena.shape}"


def _assert_points_shape(points: np.ndarray) -> None:
    """Points must be (frames, 2); squeeze keypoint dim if single keypoint."""
    assert points.ndim == 2 and points.shape[1] == 2, (
        f"points must be (frames, 2), got {points.shape}"
    )


def _arena_wall_geometry(arena_coordinates: np.ndarray):
    """subset_arena (4,2) in cyclic order and list of wall (A,B) endpoints (consecutive edges)."""
    idx_upper_arena = [0, 1, 4, 5]
    subset_arena = arena_coordinates[idx_upper_arena, :]
    centroid = np.nanmean(subset_arena, axis=0)
    angles = np.arctan2(
        subset_arena[:, 1] - centroid[1],
        subset_arena[:, 0] - centroid[0],
    )
    order = np.argsort(angles)
    subset_arena = subset_arena[order]
    idx_wall = [(0, 1), (1, 2), (2, 3), (3, 0)]
    walls = [subset_arena[[a, b], :] for a, b in idx_wall]
    return subset_arena, walls


def dist_arena(arena_coordinates: np.ndarray, points: np.ndarray):
    """
    Compute distances from each point in 'points' to each wall of the arena.

    Args:
        arena_coordinates (np.ndarray): Arena vertices, shape (N, 2).
        points (np.ndarray): Points to measure from, shape (M, 2).

    Returns:
        List[np.ndarray]: List of distance arrays, one per wall (each shape: (M,))
    """
    _assert_arena_shape(arena_coordinates)
    _assert_points_shape(points)
    subset_arena, walls = _arena_wall_geometry(arena_coordinates)
    distances = []
    for wall in walls:
        if wall.shape[0] < 2:
            continue
        proj = project_point_onto_line(points, wall[0], wall[1])
        dist = np.linalg.norm(points - proj, axis=1)
        dist[np.isnan(points).any(axis=1) | np.isnan(proj).any(axis=1)] = np.nan
        distances.append(dist)
    return distances


def projections_onto_walls(arena_coordinates: np.ndarray, points: np.ndarray) -> List[np.ndarray]:
    """Per-wall projection of points onto wall line. Returns list of (n, 2) arrays."""
    _assert_arena_shape(arena_coordinates)
    _assert_points_shape(points)
    _, walls = _arena_wall_geometry(arena_coordinates)
    return [
        project_point_onto_line(points, w[0], w[1])
        for w in walls if w.shape[0] >= 2
    ]


def nearest_corner_coords(arena_coordinates: np.ndarray, points: np.ndarray) -> np.ndarray:
    """For each point, (n, 2) coordinates of the nearest arena corner. All-NaN frames -> NaN row."""
    _assert_arena_shape(arena_coordinates)
    _assert_points_shape(points)
    subset_arena, _ = _arena_wall_geometry(arena_coordinates)
    dists = np.stack([np.linalg.norm(points - subset_arena[i], axis=1) for i in range(4)], axis=0)
    out = np.full((points.shape[0], 2), np.nan)
    valid = np.any(~np.isnan(dists), axis=0)
    idx = np.nanargmin(dists[:, valid], axis=0)
    out[valid] = subset_arena[idx, :]
    return out


def dist_nearest_corner(arena_coordinates: np.ndarray, points: np.ndarray) -> np.ndarray:
    _assert_arena_shape(arena_coordinates)
    _assert_points_shape(points)
    subset_arena, _ = _arena_wall_geometry(arena_coordinates)
    distances = []
    for idx in range(subset_arena.shape[0]):
        dist = np.linalg.norm(points - subset_arena[idx], axis=1)
        distances.append(dist)
    stacked = np.stack(distances, axis=0)
    return np.nanmin(stacked, axis=0)


def pick_min_distance(distances: List[np.ndarray]) -> np.ndarray:
    """
    Given multiple arrays of the same shape, return the elementwise minimum,
    ignoring NaNs. If all values at a position are NaN, the output is NaN.
    """
    stacked = np.stack(distances, axis=0)
    return np.nanmin(stacked, axis=0)


def points_from_keypoints(coordinates, keypoints: List[str]) -> np.ndarray:
    """
    Build (frames, 2) points from coordinates: one keypoint -> squeeze; multiple -> centroid.
    position dims from movement are (time, space, keypoints, individuals).
    """
    pos = coordinates.sel(keypoints=keypoints).position.values
    if len(keypoints) == 1:
        xy = pos.squeeze()
    else:
        # position shape (time, space, keypoints, individuals); mean over keypoints
        xy = np.nanmean(pos, axis=2).squeeze()
    if xy.ndim > 2:
        xy = xy.squeeze()
    if xy.shape[1] > 2:
        xy = xy[:, :2]
    return xy


def plot_centroid_to_wall_corner(
    arena: np.ndarray,
    xy: np.ndarray,
    n_frames: int = 100,
    out_path: str | None = None,
) -> None:
    """Plot first n_frames: centroid, line to nearest wall projection, line to nearest corner."""
    subset_arena, walls = _arena_wall_geometry(arena)
    wall_dists = dist_arena(arena, xy)
    projs = projections_onto_walls(arena, xy)
    corner_coords = nearest_corner_coords(arena, xy)
    # For each frame, which wall is nearest (all-NaN -> 0, we skip in plot)
    stacked = np.stack(wall_dists, axis=0)
    nearest_wall_idx = np.zeros(xy.shape[0], dtype=int)
    valid = np.any(~np.isnan(stacked), axis=0)
    nearest_wall_idx[valid] = np.nanargmin(stacked[:, valid], axis=0)
    n = min(n_frames, xy.shape[0])
    wall_proj = np.full((n, 2), np.nan)
    for i in range(n):
        if not valid[i]:
            continue
        w = int(nearest_wall_idx[i])
        wall_proj[i] = projs[w][i]

    fig, ax = plt.subplots()
    # Arena outline (close the rectangle)
    outline = np.vstack([subset_arena, subset_arena[0:1]])
    ax.plot(outline[:, 0], outline[:, 1], "k-", lw=1.5, label="arena")
    # Centroids
    ax.scatter(xy[:n, 0], xy[:n, 1], c=np.arange(n), s=8, cmap="viridis", alpha=0.8)
    # Lines centroid -> wall
    for i in range(n):
        if np.any(np.isnan(xy[i])) or np.any(np.isnan(wall_proj[i])):
            continue
        ax.plot([xy[i, 0], wall_proj[i, 0]], [xy[i, 1], wall_proj[i, 1]], "b-", alpha=0.25, lw=0.8)
    # Lines centroid -> corner
    for i in range(n):
        if np.any(np.isnan(xy[i])) or np.any(np.isnan(corner_coords[i])):
            continue
        ax.plot([xy[i, 0], corner_coords[i, 0]], [xy[i, 1], corner_coords[i, 1]], "r-", alpha=0.25, lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_title(f"First {n} frames: centroid → wall (blue), centroid → corner (red)")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved plot to {out_path}")
    else:
        plt.show()


def _parse_args():
    p = argparse.ArgumentParser(description="Compute wall/corner distances from pose keypoints.")
    p.add_argument("--input", required=True, help="Poses file path (for from_file)")
    p.add_argument("--source-software", default="SLEAP", help="Source software for from_file")
    p.add_argument("--arena-path", required=True, help="Path to arena dataset (e.g. .h5)")
    p.add_argument("--keypoints", nargs="+", required=True, help="One or more keypoint names")
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument("--plot-first", type=int, default=None, help="If set, plot first N frames (centroid–wall–corner) and save")
    p.add_argument("--plot-output", default=None, help="Path for plot image (used with --plot-first)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    coordinates = from_file(args.input, source_software=args.source_software)
    arena_coordinates = xr.load_dataset(args.arena_path)
    arena = arena_coordinates.sel(view="central").position.values.squeeze().reshape((8, 2))
    xy = points_from_keypoints(coordinates, args.keypoints)
    _assert_arena_shape(arena)
    _assert_points_shape(xy)
    wall_dists = dist_arena(arena, xy)
    dist_wall_min = pick_min_distance(wall_dists)
    dist_nearest_corner_arr = dist_nearest_corner(arena, xy)
    n_frames = xy.shape[0]
    pd.DataFrame({
        "frame": np.arange(n_frames),
        "dist_wall_min": dist_wall_min,
        "dist_nearest_corner": dist_nearest_corner_arr,
    }).to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({n_frames} frames)")
    if args.plot_first is not None:
        plot_path = args.plot_output or (args.output.rsplit(".", 1)[0] + "_plot.png")
        plot_centroid_to_wall_corner(arena, xy, n_frames=args.plot_first, out_path=plot_path)



