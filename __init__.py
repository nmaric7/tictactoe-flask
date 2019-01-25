import os

from flask import Flask, jsonify, url_for
from . import db, auth, blog
from .v1 import tictactoeAPI as tictactoeAPI_v1
from .v2 import tictactoeAPI as tictactoeAPI_v2

# from .tictactoe import init_model
# from .model_v2 import init_model

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
    # app.register_blueprint(auth.bp)
    # app.register_blueprint(blog.bp)

    app.register_blueprint(tictactoeAPI_v1.bp)
    app.register_blueprint(tictactoeAPI_v2.bp)

    app.add_url_rule('/', endpoint='index')

    @app.route('/site-map')
    def site_map():
        import urllib.parse
        output = []
        for rule in app.url_map.iter_rules():

            options = {}
            for arg in rule.arguments:
                options[arg] = "[{0}]".format(arg)

            methods = ','.join(rule.methods)
            # url = url_for(rule.endpoint, **options)
            line = urllib.parse.unquote("{:30s} {:25s} {}".format(rule.endpoint, methods, rule))
            output.append(line)
        
        for line in sorted(output):
            print(line)
        
        return jsonify({'output': sorted(output)})
      
    return app