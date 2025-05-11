# Module Reorganization Plan for Neural RPS

## Current Issues

1. **Circular Dependencies**: `golang_implementation` and `alphago_demo` packages have cross-dependencies.
2. **TensorFlow Integration Issues**: TensorFlow package resolution and linking are problematic.
3. **Module Structure**: Multiple Go modules in different directories with unclear boundaries.

## Proposed Structure

```
neural_rps/
├── cmd/                     # Command-line applications
│   ├── benchmark/           # Performance benchmarking tools
│   ├── mcts_play/           # MCTS-based game player
│   └── tournament/          # Tournament runner
├── pkg/                     # Shared packages
│   ├── common/              # Common types and utilities
│   ├── game/                # Game logic and state representation
│   ├── neural/              # Neural network implementations
│   │   ├── cpu/             # CPU-based neural networks
│   │   └── gpu/             # GPU-accelerated neural networks (TensorFlow)
│   └── agents/              # Agent implementations
│       ├── mcts/            # MCTS-based agents
│       ├── alphago/         # AlphaGo-style agents
│       └── ppo/             # PPO-based agents
├── internal/                # Internal implementation details
├── go.mod                   # Single module file
└── go.sum                   # Dependencies
```

## Migration Steps

1. **Create a Single Module**:
   - Consolidate into a single Go module at the root level
   - Remove individual module files from subdirectories

2. **Resolve TensorFlow Dependencies**:
   - Create a clean integration for TensorFlow that can be optionally included
   - Isolate TensorFlow code in the `pkg/neural/gpu` package
   - Implement feature detection for GPU availability

3. **Refactor Cross-Dependencies**:
   - Create shared interfaces in the `pkg/common` directory
   - Implement adapters between different agent implementations
   - Update import paths in all files

4. **Integration Testing**:
   - Create integration tests that verify all components work together
   - Ensure backward compatibility with previous implementations

5. **Documentation**:
   - Update documentation to reflect the new structure
   - Provide migration guide for existing code

## Benefits

1. **Cleaner Structure**: Single module with clear package boundaries
2. **Eliminated Circular Dependencies**: Well-defined interfaces between components
3. **Improved Maintainability**: Easier to understand and extend
4. **Better GPU Integration**: Isolated GPU code that can be optionally included
5. **Simplified Builds**: Single module means simplified build process

## Timeline

1. Create the new directory structure
2. Migrate the game logic and shared interfaces
3. Implement the CPU-based neural network code
4. Add the GPU-based neural network code
5. Migrate the agent implementations
6. Update the command-line applications
7. Create integration tests
8. Update documentation 