import argparse
import time


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the existing L5 planning flow with the unified visualizer."
    )
    parser.add_argument(
        "--preset",
        choices=["default"],
        default="default",
        help="Choose the overall visual treatment for the unified viewer.",
    )
    parser.add_argument(
        "--screw-mode",
        choices=["threaded", "cylinder", "none"],
        default="threaded",
        help="Choose how screws are rendered in the unified viewer.",
    )
    parser.add_argument(
        "--theme",
        choices=["light", "dark"],
        default="dark",
        help="Choose the Plotly theme used by the unified viewer.",
    )
    parser.add_argument(
        "--mesh-opacity",
        type=float,
        default=0.25,
        help="Starting opacity for the vertebra mesh.",
    )
    parser.add_argument(
        "--show-safety-planes",
        action="store_true",
        help="Render the optional tip safety planes from the unified viewer.",
    )
    parser.add_argument(
        "--hide-bounding-box",
        action="store_true",
        help="Disable the anatomy bounding box overlay.",
    )
    parser.add_argument(
        "--hide-trajectory-lines",
        action="store_true",
        help="Disable screw trajectory line overlays.",
    )
    parser.add_argument(
        "--hide-entry-markers",
        action="store_true",
        help="Disable screw entry-point markers.",
    )
    parser.add_argument(
        "--show-tip-markers",
        action="store_true",
        help="Render tip-point markers at the distal end of each trajectory.",
    )
    parser.add_argument(
        "--v2-neon-trajectories",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the bright layered trajectory treatment inspired by visualizerV2.",
    )
    parser.add_argument(
        "--v2-gold-screws",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the shared metallic gold screw palette from visualizerV2.",
    )
    parser.add_argument(
        "--v2-threaded-screws",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable V2-style threaded screw geometry.",
    )
    parser.add_argument(
        "--v2-safety-planes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the V2-style safety planes at the screw tips.",
    )
    parser.add_argument(
        "--fallback-diameter",
        type=float,
        default=None,
        help="Optional visual-only fallback screw diameter when the planner result omits one.",
    )
    parser.add_argument(
        "--renderer",
        choices=["auto", "browser"],
        default="auto",
        help="Viewer backend for the final display step.",
    )
    return parser


def build_visualizer_adapter(args, unified_visualize):
    def patched_visualizer(verts_world, faces, results_list, volume_path=None):
        import inspect
        
        mask_meshes = None
        vis_volume_path = volume_path
        try:
            caller_frame = inspect.currentframe().f_back
            local_vis = caller_frame.f_locals.get("vis_volume_path")
            if local_vis:
                vis_volume_path = local_vis
        except Exception:
            pass
            
        if vis_volume_path:
            try:
                import numpy as np
                import nibabel as nib
                from skimage.measure import marching_cubes
                
                nii = nib.load(vis_volume_path)
                data = np.asanyarray(nii.dataobj)
                affine = nii.affine
                
                labels = [value for value in np.unique(data) if value > 0]
                if len(labels) > 0:
                    mask_meshes = []
                    label_map = {5: "L1", 4: "L2", 3: "L3", 2: "L4", 1: "L5"}
                    for label_val in sorted(labels, reverse=True):
                        mask = np.isclose(data, label_val)
                        if not mask.any(): continue
                        coords = np.argwhere(mask)
                        if coords.size == 0: continue
                        mins = np.maximum(coords.min(axis=0) - 2, 0)
                        maxs = np.minimum(coords.max(axis=0) + 3, np.array(mask.shape, dtype=int))
                        slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(mins, maxs))
                        cropped = np.asarray(mask[slices], dtype=np.uint8)
                        
                        if np.any(cropped):
                            try:
                                v, f, _, _ = marching_cubes(cropped, level=0.5)
                            except Exception:
                                continue
                            c_aff = np.array(affine, dtype=float, copy=True)
                            c_aff[:3, 3] = nib.affines.apply_affine(affine, mins)
                            v_world = nib.affines.apply_affine(c_aff, v)
                            
                            lbl_int = int(round(float(label_val)))
                            mask_meshes.append({
                                "label": label_map.get(lbl_int, f"Label {lbl_int}"),
                                "value": float(label_val),
                                "verts_world": v_world,
                                "faces": f,
                                "visible": True
                            })
                    
                    if mask_meshes:
                        verts_world = None
                        faces = None
            except Exception as e:
                print(f"[Unified Setup] Note: separate mask_meshes optional build failed: {e}")

        fig, show_figure = unified_visualize(
            verts_world,
            faces,
            results_list,
            mask_meshes=mask_meshes,
            volume_path=volume_path,
            screw_mode=args.screw_mode,
            theme=args.theme,
            visual_preset=args.preset,
            mesh_opacity=args.mesh_opacity,
            show_safety_planes=args.show_safety_planes,
            show_bounding_box=not args.hide_bounding_box,
            show_trajectory_lines=not args.hide_trajectory_lines,
            show_entry_markers=not args.hide_entry_markers,
            show_tip_markers=args.show_tip_markers,
            v2_neon_trajectories=args.v2_neon_trajectories,
            v2_gold_screws=args.v2_gold_screws,
            v2_threaded_screws=args.v2_threaded_screws,
            v2_safety_planes=args.v2_safety_planes,
            fallback_diameter=args.fallback_diameter,
        )

        def wrapped_show_figure(fig_obj=None, renderer=None):
            target_fig = fig if fig_obj is None else fig_obj
            target_renderer = args.renderer if renderer is None else renderer
            return show_figure(target_fig, renderer=target_renderer)

        return fig, wrapped_show_figure

    return patched_visualizer


def main():
    # Lazy-load heavy modules only when main() is called
    import sys
    import os
    
    # Add parent directory to path for relative imports
    sys.path.insert(0, os.path.dirname(__file__))
    
    import plan_and_visualize_l5 as l5_runner
    from visualizer_unified import visualize_plan as unified_visualize
    
    args = build_parser().parse_args()
    l5_runner.visualize_surgical_plan = build_visualizer_adapter(args, unified_visualize)
    l5_runner.start_time = time.time()
    l5_runner.plan_and_visualize_l5()


if __name__ == "__main__":
    main()
