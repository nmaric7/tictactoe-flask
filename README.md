"# tictactoe-flask" 

This application is Rest API for TicTacToe

For run this on Windows following actions should be called from parent folder of application.

<ul>
  <li>set FLASK_APP=tictactoe-flask</li>
  <li>set FLASK_ENV=development</li>
  <li>flask run</li>
</ul>

There is two available version of NN in this app

<p>
<h3>v1</h3>

<div>
  First version is based on prepared training dataset. And it could be only trained on that data. <br/>
  Format of dataset is board position, and move that X player should play (based on tictactoe algorithm, implemented in unbeaten_player.py ). <br/>
  ex: <br />
  ---------,4 <br />
  --OXO-X--,0 <br />
  ... <br />
  
<h5>/tictactoe/api/v1.0/accuracy</h5>
<p>Method return accuracy of neural network</p>

<h5>/tictactoe/api/v1.0/move</h5>
<p>Method returns next move for X player</p>
  
<h5>/tictactoe/api/v1.0/train</h5>
<p></p>
</p>
v2

/tictactoe/api/v2.0/accuracy
/tictactoe/api/v2.0/move
/tictactoe/api/v2.0/unbeat
/tictactoe/api/v2.0/train
/tictactoe/api/v2.0/trainByMoves
