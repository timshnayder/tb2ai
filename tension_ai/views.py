import boardlib.api.aurora
import flask
import tension_ai.db
import json
import pandas
from pathlib import Path

blueprint = flask.Blueprint("view", __name__)


@blueprint.route("/board-images/<board_name>/<path:filename>")
def serve_board_image(board_name, filename):
    """Serve local board images"""
    script_dir = Path(__file__).parent.parent
    image_path = script_dir / "data" / board_name / "images" / filename
    
    if not image_path.exists():
        flask.abort(404)
    
    return flask.send_file(str(image_path))


@blueprint.route("/")
def index():
    configs = {}
    for layout_id in [10, 11]:
        configs[str(layout_id)] = {}
        for size_id in [6, 7, 8, 9, 10]:
            try:
                colors = tension_ai.db.get_data("tension", "colors", {"layout_id": layout_id})
                led_colors = get_led_colors("tension", layout_id)
                placement_positions = get_placement_positions("tension", layout_id, size_id)
                
                # Query sets dynamically for this layout/size combination
                sets_data = tension_ai.db.get_data("tension", "sets", {"layout_id": layout_id, "size_id": size_id})
                set_ids = [row[0] for row in sets_data]
                draw_kwargs = get_draw_board_kwargs("tension", layout_id, size_id, set_ids)
                
                configs[str(layout_id)][str(size_id)] = {
                    "colors": colors,
                    "led_colors": led_colors,
                    "placement_positions": placement_positions,
                    "images_to_holds": draw_kwargs["images_to_holds"],
                    "edge_left": draw_kwargs["edge_left"],
                    "edge_right": draw_kwargs["edge_right"],
                    "edge_bottom": draw_kwargs["edge_bottom"],
                    "edge_top": draw_kwargs["edge_top"],
                }
            except Exception:
                continue
        
    return flask.render_template(
        "aiDashboard.html.j2",
        configs=configs,
    )

def get_draw_board_kwargs(board_name, layout_id, size_id, set_ids):
    images_to_holds = {}
    for set_id in set_ids:
        image_filename = tension_ai.db.get_data(
            board_name,
            "image_filename",
            {"layout_id": layout_id, "size_id": size_id, "set_id": set_id},
        )[0][0]
        # Use local image route instead of Aurora API
        image_url = flask.url_for('view.serve_board_image', board_name=board_name, filename=image_filename)
        images_to_holds[image_url] = tension_ai.db.get_data(
            board_name, "holds", {"layout_id": layout_id, "set_id": set_id}
        )

    size_dimensions = tension_ai.db.get_data(
        board_name, "size_dimensions", {"size_id": size_id}
    )[0]
    return {
        "images_to_holds": images_to_holds,
        "edge_left": size_dimensions[0],
        "edge_right": size_dimensions[1],
        "edge_bottom": size_dimensions[2],
        "edge_top": size_dimensions[3],
    }


def get_placement_positions(board_name, layout_id, size_id):
    binds = {"layout_id": layout_id, "size_id": size_id}
    return {
        placement_id: position
        for placement_id, position in tension_ai.db.get_data(board_name, "leds", binds)
    }


def get_led_colors(board_name, layout_id):
    binds = {"layout_id": layout_id}
    return {
        role_id: color
        for role_id, color in tension_ai.db.get_data(board_name, "led_colors", binds)
    }
