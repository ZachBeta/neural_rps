# Hall of Fame Integration Plan

This document outlines how to incorporate a Hall-of-Fame (HOF) archive into NEAT training to prevent self-play overfitting.

## 1. CLI & Config

- Add flags:
  - `--hof-size N` (number of past champions to retain; default 5)
  - `--hof-games G` (games vs each HOF member; default = `--eval-games`)
- Extend `Config` with `HOFSize int` and `HOFGames int`.

## 2. Population Tracking

- In `Population`, add field:
  ```go
  HOF []*Genome
  ```
- After selecting each generation’s champion:
  ```go
  pop.HOF = append(pop.HOF, champion)
  if len(pop.HOF) > cfg.HOFSize {
      pop.HOF = pop.HOF[1:]
  }
  ```

## 3. Evaluation Logic

- Refactor evaluator:
  - `EvaluateSelf(g *Genome, games, threads int)` for pure self-play (gen 1 fallback).
  - `EvaluatePair(g1, g2 *Genome, games, threads int)` for head-to-head.
- In `Evolve`, for each genome `g`:
  ```go
  if len(pop.HOF)==0 {
      g.Fitness = EvaluateSelf(g, cfg.EvalGames, threads)
  } else {
      totalW, totalD := 0, 0
      for _, hof := range pop.HOF {
          w,d := EvaluatePair(g, hof, cfg.HOFGames, threads)
          totalW += w; totalD += d
      }
      g.Fitness = (float64(totalW) + 0.5*float64(totalD)) /
                  (float64(len(pop.HOF)) * float64(cfg.HOFGames))
  }
  ```

## 4. Defaults & Fallback

- Gen 1 (empty HOF): either seed HOF with a random baseline or use `EvaluateSelf` until HOF populates.

## 5. Outputs & Logging

- Maintain per-generation logs and checkpoints.
- Optionally log HOF size and per-generation evaluation totals.
