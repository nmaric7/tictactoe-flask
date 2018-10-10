import os

from flask import Flask, jsonify
from . import db, auth, blog, tictactoeAPI
from .tictactoe import init_model

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
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

    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello, World!'
        
    # init app with database
    db.init_app(app)
    # register blueprints
    app.register_blueprint(auth.bp)
    app.register_blueprint(blog.bp)
    app.register_blueprint(tictactoeAPI.bp)

    app.add_url_rule('/', endpoint='index')

    @app.before_first_request
    def init_tf():
        print("Run this function before first request")
        init_model()

    # cors = CORS(app, resources={r"/tictactoe/api/*": {"origins": "localhost:9100"}})

    return app