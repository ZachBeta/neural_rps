# Neural RPS Build System Guide

This document explains the build system structure and Makefile organization for the Neural RPS project.

## 1. Build System Overview

The Neural RPS project uses a hierarchical Makefile structure:
- A top-level Makefile with high-level commands
- Package-specific Makefiles for detailed operations

## 2. Top-Level Makefile

The top-level Makefile provides commands that apply across the whole project:

```
neural_rps/
└── Makefile  # Top-level Makefile
```

### 2.1. Main Build Targets

| Target | Description |
|--------|-------------|
| `build` | Build all implementations |
| `build-legacy-cpp` | Build legacy C++ implementation |
| `build-cpp` | Build C++ implementation |
| `build-go` | Build Golang implementation |
| `build-alphago` | Build AlphaGo demos |

### 2.2. Run Targets

| Target | Description |
|--------|-------------|
| `run-legacy-cpp` | Run legacy C++ implementation |
| `run-cpp` | Run C++ implementation |
| `run-go` | Run Golang implementation |
| `run-alphago-ttt` | Run AlphaGo Tic-Tac-Toe demo |
| `run-alphago-rps` | Run AlphaGo RPS Card Game demo |

### 2.3. Test Targets

| Target | Description |
|--------|-------------|
| `test` | Run all tests |

### 2.4. Package-Specific Targets

| Target | Description |
|--------|-------------|
| `alphago-train` | Train AlphaGo models |
| `alphago-tournament` | Run AlphaGo tournament |
| `golang-tournament` | Run Golang tournament |
| `golang-vs-alphago` | Run tournament between Golang and AlphaGo agents |

### 2.5. Utility Targets

| Target | Description |
|--------|-------------|
| `clean` | Clean build artifacts |
| `install-deps` | Install dependencies |
| `fmt` | Format code |
| `doc` | Generate documentation |

## 3. Package-Specific Makefiles

Each implementation has its own Makefile for package-specific operations:

```
neural_rps/
├── alphago_demo/
│   └── Makefile  # AlphaGo-specific Makefile
├── golang_implementation/
│   └── Makefile  # Golang-specific Makefile
└── cpp_implementation/
    └── Makefile  # C++-specific Makefile
```

### 3.1. AlphaGo Demo Makefile

| Target | Description |
|--------|-------------|
| `build` | Build all AlphaGo binaries |
| `test` | Run AlphaGo tests |
| `train` | Train AlphaGo models |
| `compare` | Compare AlphaGo models |
| `play` | Play against an AlphaGo agent |
| `clean` | Clean build artifacts |

### 3.2. Golang Implementation Makefile

| Target | Description |
|--------|-------------|
| `build` | Build all Golang binaries |
| `test` | Run Golang tests |
| `tournament` | Run tournament between different agents |
| `tournament-verbose` | Run tournament with verbose output |
| `alphago-vs-alphago` | Run tournament between AlphaGo models |
| `alphago-vs-alphago-verbose` | Run tournament with verbose output |
| `clean` | Clean build artifacts |

### 3.3. C++ Implementation Makefile

| Target | Description |
|--------|-------------|
| `all` | Build all C++ binaries |
| `neural_rps_demo` | Build the simplified demo |
| `neural_rps_full` | Build the full implementation |
| `clean` | Clean build artifacts |

## 4. Build Dependencies

The build system relies on these tools:

- **Go**: For building Go-based implementations
- **CMake**: For building C++ implementations
- **Make**: For running the build system
- **Git**: For version control

Required Go packages and C++ libraries are handled by the `install-deps` target.

## 5. Directory Organization

The build system produces artifacts in these locations:

```
neural_rps/
├── alphago_demo/
│   ├── bin/     # AlphaGo binaries
│   └── output/  # Trained models and results
├── golang_implementation/
│   ├── bin/     # Golang binaries
│   └── results/ # Tournament results
└── cpp_implementation/
    └── build/   # C++ build artifacts
```

## 6. Using the Build System

### 6.1. Basic Usage

To build all implementations:
```bash
make build
```

To run tests:
```bash
make test
```

To clean all artifacts:
```bash
make clean
```

### 6.2. Training AlphaGo Models

To train AlphaGo models:
```bash
make alphago-train
```

This will generate trained model files in `alphago_demo/output/`.

### 6.3. Running Tournaments

To run a tournament between different agents:
```bash
make golang-tournament
```

To run a tournament between AlphaGo models:
```bash
make golang-vs-alphago
```

### 6.4. Building Individual Implementations

To build just the Golang implementation:
```bash
make build-go
```

To build just the AlphaGo demos:
```bash
make build-alphago
```

## 7. Extending the Build System

### 7.1. Adding New Targets

To add a new target to the top-level Makefile:

1. Define the target and its dependencies
2. Add documentation to the .PHONY line
3. Update this build_system.md document

Example:
```makefile
.PHONY: ... new-target

# New target description
new-target:
    @echo "Running new target..."
    @cd some_directory && do_something
```

### 7.2. Adding New Package Makefiles

If you add a new implementation or package:

1. Create a Makefile in the package directory
2. Define targets specific to that package
3. Add targets in the top-level Makefile to call the package-specific targets

## 8. Troubleshooting

### 8.1. Command Not Found

If you see "command not found" errors:
```bash
make install-deps
```

### 8.2. Build Failures

For Go build failures:
```bash
cd <package_dir> && go mod tidy
```

For C++ build failures:
```bash
cd cpp_implementation && rm -rf build && mkdir build && cd build && cmake ..
```

### 8.3. Path Issues

If you encounter path-related errors:
- Ensure you're running commands from the project root
- Check relative paths in your Makefiles
- Use $(CURDIR) for absolute paths

## 9. Continuous Integration

For CI systems:
```bash
make build && make test
```

This will build all implementations and run tests.

## 10. Conclusion

The Neural RPS build system provides a structured way to build, test, and run different implementations of neural network-based RPS games. By understanding the hierarchy and organization, you can efficiently work with the codebase and extend it as needed. 