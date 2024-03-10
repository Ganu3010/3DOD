import os

import open3d as o3d
from flask import Flask

ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = os.path.join(ROOT_FOLDER, 'static', 'input')

print(f"{ROOT_FOLDER =}")
print(f"{UPLOAD_FOLDER =}")

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'website.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    app.config['ROOT_FOLDER'] = ROOT_FOLDER
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    
    # Initialize open3d webserver for visualization
    o3d.visualization.webrtc_server.enable_webrtc()

    from . import db
    db.init_app(app)

    from . import auth
    app.register_blueprint(auth.bp)

    from . import blog
    app.register_blueprint(blog.bp)
    app.add_url_rule('/', endpoint='index')

    return app
