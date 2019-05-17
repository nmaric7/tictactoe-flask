import functools

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for, jsonify
)

from flask_cors import CORS
from . import model as m
from ..helpers import model as model_helper
from .. import unbeatable_player as up 
from ..Move import Move
from ..helpers import board as b 

bp = Blueprint('tictactoe_v2', __name__, url_prefix='/tictactoe')
CORS(bp)


@bp.before_app_first_request
def init_tf():
    print("Run this function before first request")
    m.init_model()
    if model_helper.checkpoint_exists(m.get_dictionary()) == False:
        model_helper.create_and_save_default_model(m.get_dictionary())

@bp.route('/api/v2.0/test', methods=['GET'])
def test():
    return jsonify({'hello world': 'from v2'})

@bp.route('/api/v2.0/accuracy', methods=['POST'])
def get_accuracy():
    try:
        acc = model_helper.accuracy(m.get_dictionary())
        return jsonify({'accuracy': acc})
    except Exception as e:
        return jsonify({'error': str(e)})    

@bp.route('/api/v2.0/move', methods=['POST'])
def get_move():
    board = request.form.get('board')
    try:
        move = model_helper.predict(board, m.get_dictionary())
        return jsonify({'move': move})
    except Exception as e:
        return jsonify({'error': str(e)})    

@bp.route('/api/v2.0/train', methods=['POST'])
def train_NN():
    try:
        model_helper.train_NN(m.get_dictionary())
        acc = model_helper.accuracy(m.get_dictionary())
        return jsonify({'accuracy': acc})
    except Exception as e:
        return jsonify({'error': str(e)})    

@bp.route('/api/v2.0/unbeat', methods=['POST'])
def get_unbeat_move():
    board = request.form.get('board')
    mark = request.form.get('mark')
    try:
        move = up.get_next_move(board, mark)
        return jsonify({'move': move})
    except Exception as e:
        return jsonify({'error': str(e)})    

@bp.route('/api/v2.0/trainByMoves', methods=['POST'])
def train_by_moves():
    try:
        moves = request.json['moves']
        mark = request.json['mark']
        unbeat_moves = [(move['board'], move['move'], up.get_next_move(move['board'], mark)) for move in moves if move['mark'] == mark]
        new_boards = [(move[0] + ',' + str(move[2])) for move in unbeat_moves if str(move[1]) != str(move[2])]
        b.append_training_data(m.get_dictionary().get('data_path'), m.get_dictionary().get('data_name'), new_boards)

        return train_NN()
    except Exception as e:
        return jsonify({'error': str(e)})    

@bp.route('/api/v2.0/addTrainingData', methods=['POST'])
def add_training_data():
    try:
        moves = request.json['moves']
        mark = request.json['mark']
        unbeat_moves = [(move['board'], move['move'], up.get_next_move(move['board'], mark)) for move in moves if move['mark'] == mark]
        new_boards = [(move[0] + ',' + str(move[2])) for move in unbeat_moves if str(move[1]) != str(move[2])]
        b.append_training_data(m.get_dictionary().get('data_path'), m.get_dictionary().get('data_name'), new_boards)

        return jsonify({'message': 'moves are addes'})
    except Exception as e:
        return jsonify({'error': str(e)})  