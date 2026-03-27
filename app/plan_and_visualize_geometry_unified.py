import argparse

import plan_and_visualize_geometry as geometry_runner
from visualizer_unified import visualize_surgical_plan as unified_visualize


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the existing geometry planning flow with the unified visualizer."
    )
    parser.add_argument(
        "--preset",
        choices=["classic", "surgical", "cinematic"],
        default="cinematic",
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


def build_visualizer_adapter(args):
    def patched_visualizer(verts_world, faces, results_list, volume_path=None):
        fig, show_figure = unified_visualize(
            verts_world,
            faces,
            results_list,
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
        show_figure(fig, renderer=args.renderer)
        return fig, show_figure

    return patched_visualizer


def main():
    args = build_parser().parse_args()
    geometry_runner.visualize_surgical_plan = build_visualizer_adapter(args)
    geometry_runner.plan_and_visualize_geometry()


if __name__ == "__main__":
    main()
