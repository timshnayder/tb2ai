import sys
import flask
import logging

import tension_ai.api
import tension_ai.views
import tension_ai.generator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

def create_app():
    app = flask.Flask(__name__, instance_relative_config=True)
    app.url_map.strict_slashes = False
    app.register_blueprint(tension_ai.api.blueprint)
    app.register_blueprint(tension_ai.views.blueprint)
    
    # Initialize AI generator
    tension_ai.generator.init_generator()
    
    return app
