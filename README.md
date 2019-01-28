"# tictactoe-flask" 

This application is Rest API for TicTacToe

For run this on Windows following actions should be called from parent folder of application.

<ul>
  <li>set FLASK_APP=tictactoe-flask</li>
  <li>set FLASK_ENV=development</li>
  <li>flask run</li>
</ul>

There is two available versions of NN in this app

<p>
  <h3>v1</h3>
  <div>
    First version is based on prepared training dataset. And it could be only trained on that data. <br/>
    Format of dataset is board position, and move that X player should play (based on tictactoe algorithm, implemented in unbeaten_player.py ). <br/>
    ex: <br />
    ---------,4 <br />
    --OXO-X--,0 <br />
    ... <br />
    Training set is available in /v1/data/XBoardsWithResults.txt
  </div>
  <p>
    <h5>/tictactoe/api/v1.0/accuracy</h5>
    Method return accuracy of neural network
  </p>

  <p>
    <h5>/tictactoe/api/v1.0/move</h5>
    Method returns next move for X player
  </p>

  <p>
    <h5>/tictactoe/api/v1.0/train</h5>
    Train neural network
  </p>
</p>

<p>
  <h3>v2</hr>
  <p>
    <h5>/tictactoe/api/v2.0/accuracy</h5>
  <p>
  
  <p>
    <h5>/tictactoe/api/v2.0/move</h5>
  </p>
  
  <p>
    <h5>/tictactoe/api/v2.0/unbeat</h5>
  </p>
  
  <p>
    <h5>/tictactoe/api/v2.0/train</h5>
  </p>
  
  <p>
    <h5>/tictactoe/api/v2.0/trainByMoves</h5>
  </p>  
</p>
