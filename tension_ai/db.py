import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "data" / "tension" / "board_metadata.json"

# Load metadata once on startup
metadata = {}
if METADATA_PATH.exists():
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

def get_data(board_name, query_name, binds={}):
    layout_id = str(binds.get("layout_id", 11))
    size_id = str(binds.get("size_id", 8))
    set_id = binds.get("set_id")
    
    if query_name == "colors":
        # Returns list of tuples: (placement_role_id, hex_color)
        layout_data = metadata.get("layouts", {}).get(layout_id, {})
        return layout_data.get("colors", [])
        
    elif query_name == "led_colors":
        # Returns list of tuples: (placement_role_id, led_color)
        layout_data = metadata.get("layouts", {}).get(layout_id, {})
        return [(int(role_id), led_color) for role_id, led_color in layout_data.get("led_colors", {}).items()]
        
    elif query_name == "leds":
        # Returns list of tuples: (placement_id, led_position)
        layout_size_data = metadata.get("layouts", {}).get(layout_id, {}).get("sizes", {}).get(size_id, {})
        return [(int(p_id), led_pos) for p_id, led_pos in layout_size_data.get("placement_positions", {}).items()]
        
    elif query_name == "sets":
        layout_size_data = metadata.get("layouts", {}).get(layout_id, {}).get("sizes", {}).get(size_id, {})
        images_to_holds = layout_size_data.get("images_to_holds", {})
        set_names = {
            12: "Wood Holds",
            13: "Plastic Holds",
            14: "Wood Expansion",
            15: "Plastic Expansion"
        }
        return [(int(s_id), set_names.get(int(s_id), f"Set {s_id}")) for s_id in images_to_holds.keys()]
        
    elif query_name == "image_filename":
        # Returns filename matching layout, size and set
        layout_size_data = metadata.get("layouts", {}).get(layout_id, {}).get("sizes", {}).get(size_id, {})
        images_to_holds = layout_size_data.get("images_to_holds", {})
        
        # Look up by set_id directly (which is stored as string keys '10' or '11')
        set_data = images_to_holds.get(str(set_id))
        if set_data:
            url = set_data.get("image_url", "")
            filename = url.replace(f"/board-images/{board_name}/", "")
            return [(filename,)]
        return [("product_sizes_layouts_sets/8x12-tb2-plastic.png",)]

    elif query_name == "holds":
        # Returns coordinates list for layout and set
        # Using size "10" as full baseline containing all placement coordinates
        layout_size_data = metadata.get("layouts", {}).get(layout_id, {}).get("sizes", {}).get("10", {})
        if not layout_size_data:
            layout_size_data = metadata.get("layouts", {}).get(layout_id, {}).get("sizes", {}).get(size_id, {})
        
        images_to_holds = layout_size_data.get("images_to_holds", {})
        set_data = images_to_holds.get(str(set_id))
        if set_data:
            return set_data.get("holds", [])
        return []
        
    elif query_name == "size_dimensions":
        # Edge boundaries [edge_left, edge_right, edge_bottom, edge_top]
        layout_size_data = metadata.get("layouts", {}).get("11", {}).get("sizes", {}).get(size_id, {})
        return [(
            layout_size_data.get("edge_left", -40),
            layout_size_data.get("edge_right", 40),
            layout_size_data.get("edge_bottom", 4),
            layout_size_data.get("edge_top", 140)
        )]
        
    return []
