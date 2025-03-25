* move from tic tac toe to rps card game
* ✅ we should be able to compare the 3 models using consistent output
  * Created shared_output_format.md with standardized format specification
  * Created example outputs for each implementation
  * Provided implementation instructions in implementation_instructions.md
  * Added validation script validate_output_format.py


* get run outputs to be comparable in a diff view

* it seems like the alpha go version has blended tic tac toe with rock paper scissors, let's explore that game mash up idea a little more, maybe try to define the game in a consistent way so all neural network models can play against the same game, maybe create a golang service that the cpp and golang code can call so it's shared

* time the runs
  * make sure they have the same configuration - maybe we can standardize the input format as well as the output format