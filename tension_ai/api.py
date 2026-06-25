import flask
import logging
import tension_ai.generator

import time
from collections import defaultdict

blueprint = flask.Blueprint("api", __name__)

# In-memory storage for request timestamps by IP
_rate_limit_records = defaultdict(list)

def get_client_ip():
    cf_ip = flask.request.headers.get("CF-Connecting-IP")
    if cf_ip:
        return cf_ip
    forwarded = flask.request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return flask.request.remote_addr

def is_rate_limited(ip, limit=10, period=60):
    now = time.time()
    # Filter out timestamps older than the tracking period
    timestamps = [t for t in _rate_limit_records[ip] if now - t < period]
    _rate_limit_records[ip] = timestamps
    
    if len(timestamps) >= limit:
        return True
    
    _rate_limit_records[ip].append(now)
    return False


@blueprint.route("/api/v1/generate", methods=["POST"])
def api_generate():
    # Enforce rate limit (max 10 requests per 60 seconds per IP)
    ip = get_client_ip()
    if is_rate_limited(ip, limit=10, period=60):
        return flask.jsonify({
            "error": True,
            "description": "Too many requests. Please wait before generating more climbs."
        }), 429

    try:
        req_data = flask.request.get_json() or {}
        layout_id = int(req_data.get("layout_id", 11))
        size_id = int(req_data.get("size_id", 8))
        grade = req_data.get("grade", "V5").strip()
        temperature = float(req_data.get("temperature", 0.7))
        angle = int(req_data.get("angle", 40))
        is_nomatch = bool(req_data.get("is_nomatch", False))
        max_len = int(req_data.get("max_len", 20))
        beam_width = int(req_data.get("beam_width", 4))
        
        # Run AI generation
        result = tension_ai.generator.generate_climb_for_grade(
            grade_name=grade,
            temperature=temperature,
            target_layout_id=layout_id,
            size_id=size_id,
            angle=angle,
            is_nomatch=is_nomatch,
            max_len=max_len,
            beam_width=beam_width
        )
        return flask.jsonify(result)
    except KeyError as e:
        return flask.jsonify({"error": True, "description": str(e)}), 400
    except Exception as e:
        logging.error(f"Error in api_generate: {str(e)}", exc_info=True)
        return flask.jsonify({"error": True, "description": "AI Inference failed: " + str(e)}), 500