#!/usr/bin/env python3
import argparse
import os
import numpy as np
import trimesh
import matplotlib.cm as cm

def compute_curvature_weights(mesh, radius_val, factor):
    """
    Computes face weights based on curvature for adaptive sampling.
    """
    print("Computing curvature weights...")
    
    # 2) Determine curvature radius
    radius = radius_val if radius_val is not None else mesh.scale * 0.01

    # 3) Compute per-vertex mean curvature H_v
    vert_curv = trimesh.curvature.discrete_mean_curvature_measure(
        mesh, mesh.vertices, radius
    )

    # 4) Compute per-face curvature using barycentric area weights
    faces = mesh.faces
    # vertex coordinates for each face
    v0 = mesh.vertices[faces[:, 0]]
    v1 = mesh.vertices[faces[:, 1]]
    v2 = mesh.vertices[faces[:, 2]]
    # face centroids
    centroids = (v0 + v1 + v2) / 3.0

    # sub-triangle areas at each vertex
    a0 = 0.5 * np.linalg.norm(np.cross(v1 - centroids, v2 - centroids), axis=1)
    a1 = 0.5 * np.linalg.norm(np.cross(v2 - centroids, v0 - centroids), axis=1)
    a2 = 0.5 * np.linalg.norm(np.cross(v0 - centroids, v1 - centroids), axis=1)
    total_area = a0 + a1 + a2
    
    # Avoid division by zero for degenerate faces
    total_area[total_area == 0] = 1.0

    # barycentric weights
    w0 = a0 / total_area
    w1 = a1 / total_area
    w2 = a2 / total_area

    # weighted face curvature
    face_curv = (
        w0 * vert_curv[faces[:, 0]] +
        w1 * vert_curv[faces[:, 1]] +
        w2 * vert_curv[faces[:, 2]]
    )
    abs_face_curv = np.clip(np.abs(face_curv), a_min=0, a_max=None)

    # 5) Apply exponent factor to curvature
    face_weight = np.power(abs_face_curv, factor)

    # 6) Incorporate face area
    face_weight *= mesh.area_faces
    
    return face_weight, vert_curv


def visualize_results(mesh, points, vert_curv):
    """
    Visualizes the mesh colored by curvature and the sampled points.
    """
    print("Visualizing curvature and sampled points...")
    
    # Color map on mesh by vertex curvature
    abs_vert = np.clip(np.abs(vert_curv), a_min=0, a_max=None)
    vmin, vmax = abs_vert.min(), abs_vert.max()
    vnorm = (abs_vert - vmin) / (vmax - vmin + 1e-8)
    vcolors = cm.jet(vnorm)[:, :3]
    mesh.visual.vertex_colors = (vcolors * 255).astype(np.uint8)

    # Sampled points cloud in yellow
    pc = trimesh.PointCloud(points, colors=[255, 255, 0, 255])
    
    # Create and show the scene
    scene = trimesh.Scene([mesh, pc])
    scene.show()


def save_results(points, normals, num_points, input_path):
    """
    Saves the sampled points and normals to an .xyz file.
    """
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_name = f"n{num_points}_{base}_inputPoints.xyz"
    
    print(f"Saving {len(points)} samples to '{out_name}'...")
    with open(out_name, 'w') as f:
        for p, n in zip(points, normals):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Curvature-adaptive face-based sampling on a mesh (Trimesh)"
    )
    parser.add_argument(
        "input_mesh",
        help="Input mesh file (OBJ/PLY)"
    )
    parser.add_argument(
        "-n", "--num_points",
        type=int,
        required=True,
        help="Number of points to sample"
    )
    parser.add_argument(
        "--curvature_radius",
        type=float,
        default=None,
        help="Radius for mean curvature measure (default: mesh.scale * 0.01)"
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=1.0,
        help="Exponent factor to adjust curvature influence (default: 1.0)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize curvature and sampled points on the mesh"
    )
    args = parser.parse_args()

    # 1) Load mesh
    print(f"Loading mesh from '{args.input_mesh}'...")
    mesh = trimesh.load(args.input_mesh, process=False)
    if mesh.is_empty:
        raise RuntimeError(f"Failed to load mesh: '{args.input_mesh}'")

    print(f"Mesh '{args.input_mesh}' watertight: {mesh.is_watertight}")
    if not mesh.is_watertight:
        print("Warning: Mesh is not watertight. Curvature results may be inaccurate.")

    # 2) Compute weights
    face_weights, vert_curv_for_viz = compute_curvature_weights(
        mesh, args.curvature_radius, args.factor
    )

    # 3) Perform weighted face-based surface sampling
    print(f"Sampling {args.num_points} points with curvature-adaptive weights...")
    points, face_indices = trimesh.sample.sample_surface(
        mesh, args.num_points, face_weight=face_weights
    )
    normals = mesh.face_normals[face_indices]

    # 4) Save results
    save_results(points, normals, args.num_points, args.input_mesh)
    
    # 5) Optional visualization
    if args.curvature_visualize:
        visualize_results(mesh, points, vert_curv_for_viz)

# Usage
# python sampling/curvature_based_sampling.py input_mesh.obj -n 1000 --curvature_radius 0.05 --factor 1.5 --visualize