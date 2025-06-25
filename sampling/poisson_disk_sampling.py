import argparse
import os
import numpy as np
import open3d as o3d

def process_mesh(input_path, num_points, init_factor):
    """
    Loads a mesh, samples points using Poisson disk sampling, and returns the mesh and point cloud.
    """
    # 1) Load mesh
    print(f"Loading mesh from '{input_path}'...")
    mesh = o3d.io.read_triangle_mesh(input_path)
    if mesh.is_empty():
        raise RuntimeError(f"Failed to load a valid mesh from '{input_path}'")

    # 2) Check for and compute vertex normals
    if not mesh.has_vertex_normals() or len(mesh.vertex_normals) == 0:
        print("Computing vertex normals...")
        mesh.compute_vertex_normals()
        if not mesh.has_vertex_normals():
            raise RuntimeError("Failed to compute vertex normals on the mesh.")

    # 3) Poisson-disk sampling
    print(f"Sampling {num_points} points...")
    pcd = mesh.sample_points_poisson_disk(
        number_of_points=num_points,
        init_factor=init_factor
    )

    # 4) Validate results
    if not pcd.has_points() or not pcd.has_normals():
        raise RuntimeError(
            "Sampling failed to generate points or normals. "
            "Ensure vertex normals exist on the mesh."
        )

    return mesh, pcd

def save_point_cloud(pcd, input_path, num_points):
    """
    Saves the point cloud to an .xyz file.
    """
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_name = f"n{num_points}_{base}_inputPoints.xyz"
    
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    print(f"Saving {len(points)} points to '{out_name}'...")
    with open(out_name, 'w') as f:
        for p, n in zip(points, normals):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n"
            )

def visualize_results(mesh, pcd):
    """
    Visualizes the mesh and the sampled point cloud.
    """
    print("Visualizing the sampled points on the mesh...")
    # Set point color to red for visibility
    pcd.paint_uniform_color([1.0, 0.0, 0.0]) # More direct way to set color

    # Use explicit visualizer for more control
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    try:
        vis.add_geometry(mesh)
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.mesh_show_wireframe = True
        opt.light_on = False  # Turn off lighting to show raw colors
        vis.run()
    finally:
        vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample points (and normals) on a mesh surface using Poisson disk sampling (Open3D)"
    )
    parser.add_argument(
        "input_mesh",
        help="Path to the input mesh file (e.g. OBJ, PLY)"
    )
    parser.add_argument(
        "-n", "--num_points",
        type=int,
        required=True,
        help="Number of points to sample"
    )
    parser.add_argument(
        "--init_factor",
        type=float,
        default=5.0,
        help="Initial oversampling factor (default: 5.0)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the mesh and the sampled point cloud"
    )
    
    # Parse arguments
    args = parser.parse_args()

    # Orchestrate the sampling process directly
    try:
        mesh, pcd = process_mesh(args.input_mesh, args.num_points, args.init_factor)
        save_point_cloud(pcd, args.input_mesh, args.num_points)

        if args.visualize:
            visualize_results(mesh, pcd)
    except Exception as e:
        print(f"An error occurred: {e}")


# Usage
# python sampling/poisson_disk_sampling.py input_mesh.obj -n 1000 --init_factor 5.0 --visualize