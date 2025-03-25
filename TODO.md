# TODO
* tracking progress

## in progress

### housekeeping

* rules is outdated
* readmes probably are too

### standarizing the model inputs and outputs to compare effectiveness
* move from tic tac toe to rps card game - I thought this was done
* scripts to compar inputs and outputs of models running
  * this seems to be drifting
  * get run outputs to be comparable in a diff view
* time the runs
  * make sure they have the same configuration - maybe we can standardize the input format as well as the output format

### shared game
* it seems like the alpha go version has blended tic tac toe with rock paper scissors, let's explore that game mash up idea a little more, maybe try to define the game in a consistent way so all neural network models can play against the same game, maybe create a golang service that the cpp and golang code can call so it's shared
