import functools

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for, jsonify
)
from flask_cors import CORS
from .tictactoe import predict, train
from .tictactoe_v2 import accuracy
from .UnbeatablePlayer import get_next_move

bp = Blueprint('tictactoe', __name__, url_prefix='/tictactoe')
CORS(bp)

tasks = [
    {
        'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol', 
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web', 
        'done': False
    }
]
    
@bp.route('/todo/api/v1.0/tasks', methods=['GET'])
def get_tasks():
    return jsonify({'tasks': tasks})

"""     @app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['GET'])
    def get_task(task_id):
        task = [task for task in tasks if task['id'] == task_id]
        if len(task) == 0:
            abort(404)
        return jsonify({'task': task[0]})    
 """

@bp.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    task = [task for task in tasks if task['id'] == task_id]
    if len(task) == 0:
        abort(404)
    return jsonify({'task': task[0]})

@bp.route('/api/v1.0/move', methods=['POST'])
def get_move():
    board = request.form.get('board')
    try:
        move = predict(board)
        return jsonify({'move': move})
    except Exception as e:
        return jsonify({'error': str(e)})    

@bp.route('/api/v1.0/accuracy', methods=['POST'])
def get_accuracy():
    try:
        acc = accuracy()
        return jsonify({'accuracy': acc})
    except Exception as e:
        return jsonify({'error': str(e)})    

@bp.route('/api/v1.0/train', methods=['POST'])
def train_NN():
    try:
        train()
        acc = accuracy()
        return jsonify({'accuracy': acc})
    except Exception as e:
        return jsonify({'error': str(e)})    

@bp.route('/api/v1.0/unbeat', methods=['POST'])
def get_unbeat_move():
    board = request.form.get('board')
    mark = request.form.get('mark')
    try:
        move = get_next_move(board, mark)
        return jsonify({'move': move})
    except Exception as e:
        return jsonify({'error': str(e)})    