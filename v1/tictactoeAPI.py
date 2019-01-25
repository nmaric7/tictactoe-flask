import functools

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for, jsonify
)

from flask_cors import CORS
from . import model as m
from ..helpers import model as model_helper

bp = Blueprint('tictactoe_v1', __name__, url_prefix='/tictactoe')
CORS(bp)

@bp.before_app_first_request
def init_tf():
    print("Run this function before first request")
    m.init_model()
    if model_helper.checkpoint_exists(m.get_dictionary()) == False:
        model_helper.create_and_save_default_model(m.get_dictionary())

@bp.route('/api/v1.0/test', methods=['GET'])
def test():
    return jsonify({'hello world': 'from v1'})

@bp.route('/api/v1.0/accuracy', methods=['POST'])
def get_accuracy():
    try:
        acc = model_helper.accuracy(m.get_dictionary())
        return jsonify({'accuracy': acc})
    except Exception as e:
        return jsonify({'error': str(e)})  

@bp.route('/api/v1.0/train', methods=['POST'])
def train_NN():
    try:
        model_helper.train_NN(m.get_dictionary())
        acc = model_helper.accuracy(m.get_dictionary())
        return jsonify({'accuracy': acc})
    except Exception as e:
        return jsonify({'error': str(e)})   

@bp.route('/api/v1.0/move', methods=['POST'])
def get_move():
    board = request.form.get('board')
    try:
        move = model_helper.predict(board, m.get_dictionary())
        return jsonify({'move': move})
    except Exception as e:
        return jsonify({'error': str(e)})      
