"""
Computes various quality metrics to compare an original and a simplified mesh.

- Chamfer Distance
- Hausdorff Distance
- Edge Preservation
- Normal Consistency
"""
import argparse
import sys
import trimesh
from typing import Tuple, Dict, Any

from metrics import (
    chamfer_distance,
    hausdorff_distance,
    edge_preservation,
    normal_consistency
)

def setup_arg_parser() -> argparse.ArgumentParser:
    """Sets up command-line arguments and returns the parser."""
    parser = argparse.ArgumentParser(
        description="Computes quality metrics between an original and a simplified mesh.",
        formatter_class=argparse.RawTextHelpFormatter  # Preserve help message formatting
    )
    parser.add_argument(
        "original_mesh",
        type=str,
        help="Path to the original (high-res) mesh file."
    )
    parser.add_argument(
        "simplified_mesh",
        type=str,
        help="Path to the simplified mesh file."
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=10000,
        help="Number of sample points for distance-based metrics (default: 10000)."
    )
    parser.add_argument(
        "--angle_threshold", "-a",
        type=float,
        default=30.0,
        help="Dihedral angle threshold in degrees for identifying important edges (default: 30°)."
    )
    parser.add_argument(
        "--important_edge_factor", "-f",
        type=float,
        default=2.0,
        help="Weight factor for important edges in edge preservation assessment (default: 2.0)."
    )
    return parser

def load_meshes(original_path: str, simplified_path: str) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """
    Loads the original and simplified meshes from the specified paths.

    Args:
        original_path (str): Path to the original mesh file.
        simplified_path (str): Path to the simplified mesh file.

    Returns:
        A tuple containing (original_mesh, simplified_mesh).
    """
    try:
        print(f"Loading original mesh from '{original_path}'...")
        mesh_orig = trimesh.load(original_path, process=False)
        print(f"Loading simplified mesh from '{simplified_path}'...")
        mesh_simp = trimesh.load(simplified_path, process=False)
        
        if mesh_orig.is_empty or mesh_simp.is_empty:
            raise ValueError("One or more mesh files are empty or failed to load.")
            
        return mesh_orig, mesh_simp
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: A problem occurred while loading meshes - {e}", file=sys.stderr)
        sys.exit(1)

def calculate_all_metrics(
    mesh_orig: trimesh.Trimesh, 
    mesh_simp: trimesh.Trimesh, 
    args: argparse.Namespace
) -> Dict[str, float]:
    """
    Calculates all quality metrics and returns them as a dictionary.

    Args:
        mesh_orig (trimesh.Trimesh): The original mesh.
        mesh_simp (trimesh.Trimesh): The simplified mesh.
        args (argparse.Namespace): Command-line arguments needed for calculations (e.g., samples).

    Returns:
        A dictionary of metric names and their calculated values.
    """
    print("\nStarting quality metric calculation...")
    
    # 1) Chamfer Distance
    cd = chamfer_distance(mesh_orig, mesh_simp, samples=args.samples)

    # 2) Hausdorff Distance
    hd = hausdorff_distance(mesh_orig, mesh_simp, samples=args.samples)

    # 3) Edge Preservation
    ep = edge_preservation(
        mesh_orig,
        mesh_simp,
        angle_threshold=args.angle_threshold,
        important_edge_factor=args.important_edge_factor
    )

    # 4) Normal Consistency (original vs simplified)
    nc_orig = normal_consistency(mesh_orig, samples=args.samples)
    nc_simp = normal_consistency(mesh_simp, samples=args.samples)
    
    return {
        "Chamfer Distance": cd,
        "Hausdorff Distance": hd,
        "Edge Preservation": ep,
        "Normal Consistency (orig)": nc_orig,
        "Normal Consistency (simp)": nc_simp,
        "Normal Consistency Δ": nc_orig - nc_simp,
    }

def print_results(metrics: Dict[str, Any]) -> None:
    """Prints the calculated metrics in a well-formatted table."""
    print("\n" + "="*38)
    print("      Mesh Quality Metrics      ")
    print("="*38)
    
    # Align the output based on the longest label length
    max_label_len = max(len(label) for label in metrics.keys())
    
    for label, value in metrics.items():
        # Left-align the label and right-align the value, formatted to 6 decimal places
        print(f"{label:<{max_label_len}} : {value:>10.6f}")
        
    print("="*38)


if __name__ == "__main__":
    # 1. Parse command-line arguments
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()

    # 2. Load mesh files
    original_mesh, simplified_mesh = load_meshes(args.original_mesh, args.simplified_mesh)
    
    # 3. Calculate all metrics
    calculated_metrics = calculate_all_metrics(original_mesh, simplified_mesh, args)
    
    # 4. Print the results
    print_results(calculated_metrics)

# Usage
# python compute_metrics.py original_mesh.obj simplified_mesh.obj --samples 100000 --angle_threshold 30 --important_edge_factor 2.0