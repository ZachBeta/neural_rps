# NEAT Training: Sensible Defaults & Minimal Configuration

This document outlines a robust, user-friendly approach for configuring NEAT training on modern consumer hardware (e.g., Apple M1). The goal is to ensure NEAT “just works” out-of-the-box, with minimal CLI flags and no need for a high-end GPU.

---

## Principles for Sensible Defaults

- **Minimal CLI configuration:** Only expose the most impactful options.
- **Defaults work well on a modern CPU:** No need for user tuning for typical experiments.
- **Advanced users can override via config file or env vars (optional).**

---

## Recommended Defaults

| Parameter                  | Default Value         | Rationale                                        |
|----------------------------|----------------------|--------------------------------------------------|
| Population Size            | 100                  | Diversity + speed                                |
| Generations                | 30                   | Enough for visible progress                      |
| Round Robin Opponents      | 5                    | Robustness vs. speed                             |
| Games per Round Robin      | 2                    | Reduce noise, not excessive                      |
| Hall-of-Fame Size          | 5                    | Historical challenge, manageable compute         |
| Games per HOF Member       | 5                    | Robustness, not excessive                        |
| Threads                    | NumCPU() - 1         | Uses all but one core for responsiveness         |
| Hidden Units (network)     | 64                   | Small, fast, non-trivial                         |

---

## Example: What the User Sees

```sh
./train_models --method neat
```

- Uses all sensible defaults for a fast, interesting run on a modern CPU.
- No need to specify population, threads, or evaluation details.

---

## Implementation Guidance

- **Hardcode defaults in NEAT config struct.**
- **Only override if user passes a CLI flag.**
- **Print a summary of chosen parameters at the start of training.**
- **Auto-detect thread count:**
  ```go
  numThreads := runtime.NumCPU() - 1
  if numThreads < 1 { numThreads = 1 }
  ```
- **Document defaults in README/help text.**

---

## For Power Users

- Advanced config (e.g., round robin count, HOF size) can be exposed via config file or environment variables—not via CLI by default.
- Document how to override in advanced usage section.

---

## Summary

- NEAT training should be productive and interesting out-of-the-box.
- Minimal configuration, robust defaults, and full CPU utilization (without pinning the machine) are key for a great user experience.
