package neat

// Config holds hyperparameters for the NEAT trainer in the RPS/Tic-Tac-Toe domain.
// These parameters control population evolution, speciation, and fitness evaluation.
// Fields:
//  - PopSize: number of genomes per generation
//  - Generations: number of evolutionary cycles to run
//  - MutRate: probability of mutating each gene
//  - CxRate: probability of crossover vs. cloning
//  - CompatThreshold: threshold for speciation by compatibility distance
//  - EvalGames: number of self-play games per genome to estimate fitness
//  - WeightStd: standard deviation for Gaussian weight mutations
//  - HiddenSize: number of hidden units in the neural network

type Config struct {
    PopSize          int     `json:"pop_size"`
    Generations      int     `json:"generations"`
    MutRate          float64 `json:"mut_rate"`
    CxRate           float64 `json:"cx_rate"`
    CompatThreshold  float64 `json:"compat_threshold"`
    EvalGames        int     `json:"eval_games"`
    WeightStd        float64 `json:"weight_std"`
    HiddenSize       int     `json:"hidden_size"`
}
