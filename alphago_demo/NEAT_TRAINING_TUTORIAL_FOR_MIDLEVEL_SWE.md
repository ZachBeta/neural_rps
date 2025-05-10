# NEAT Training Tutorial for Mid-Level SWE

This tutorial guides a mid-level software engineer through setting up and running NEAT training for RPS on a modern consumer machine (e.g., Apple M1). The process is designed to be robust and user-friendly, with sensible defaults and minimal configuration required.

---

## 1. Overview
- **Goal:** Train a NEAT agent for RPS using parallel evaluation, round robin, and Hall-of-Fame (HOF) matches.
- **Philosophy:** Minimal CLI flags, robust defaults, and full CPU utilization (without pinning the machine).

---

## 2. Quick Start

1. **Build the CLI tool:**
    ```sh
    go build -o bin/train_models cmd/train_models/main.go
    ```
2. **Run NEAT training with defaults:**
    ```sh
    ./bin/train_models --method neat
    ```
3. **Observe CLI output:**
    - Per-generation summaries (best/avg fitness, species count)
    - Models saved in the `output/` directory

---

## 3. How It Works (Under the Hood)
- **Population size:** 100
- **Generations:** 30
- **Evaluation:** Each genome faces 5 random round robin opponents (2 games each) and 5 HOF members (5 games each)
- **Threads:** Uses all but one CPU core (`runtime.NumCPU() - 1`)
- **Network size:** 64 hidden units
- **No GPU required!**

---

## 4. Customization (Advanced, Optional)
- Most users do not need to change defaults.
- Advanced users can override population size, generations, or evaluation details via a config file or environment variables (see `NEAT_DEFAULTS_AND_CONFIG.md`).

---

## 5. Best Practices
- **Let the defaults work for you!** Only tweak if you have a specific research or performance need.
- **Monitor progress:** Watch the CLI for per-generation summaries.
- **Pause or stop training:** Use Ctrl+C; partial results will be available in `output/`.

---

## 6. Troubleshooting
- **Runs too slowly?** Reduce population or generations in code/config.
- **Runs too quickly/overfits?** Increase population or generations.
- **Machine feels sluggish?** Lower thread count in code to leave more CPU free.

---

## 7. Summary
- NEAT training is designed to be approachable, efficient, and robust on a modern CPU.
- Minimal configuration, clear CLI feedback, and no need for a high-end GPU.

Happy evolving!
