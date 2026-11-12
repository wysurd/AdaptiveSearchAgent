# Probabilistic Enemy Search AI for a Psychological Horror Game

**Author:** Nicoli Benson  
**Course:** CSC 546/746 Artificial Intelligence, Eastern Kentucky University

## Project overview

I built this prototype because I wanted a horror enemy that could respond to changing information rather than repeat an easily recognized pattern. The player collects 20 orbs while avoiding an enemy that uses A* navigation, Markov movement prediction, and a probability distribution over possible player locations.

Sightings, visible empty locations, and uncertain footstep reports update the enemy's belief. The default hybrid strategy checks the last sighting before exploring probable locations. Sprinting, stamina, silent sneaking, and an optional circular viewing window support the gameplay.

The report's controlled experiment tests **vision-only reacquisition**, meaning seeing the player again. It does not evaluate capture, hearing, player tension, or full-game win rate.

## Dependencies

The author ran the project on macOS with:

- Python 3.12.4 (verified) and 3.14.2 (original development)
- pygame-ce 2.5.8

`requirements.txt` pins the only third-party runtime dependency:

```text
pygame-ce==2.5.8
```

The package installs as `pygame-ce`, but the code imports it as `pygame`. Use a clean virtual environment to avoid mixing it with a separate Pygame installation. Other imports are Python standard-library modules. No external dataset, graphics assets, or trained model is required.

## Setup on macOS

Clone or download the repository, open the `horror-enemy-ai` folder in VS Code, and open its terminal. Run:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=:all: -r requirements.txt
```

Check the installed dependency:

```bash
python -c "import pygame; print(pygame.version.ver)"
```

The environment above is the author's tested setup. If installation reports no matching binary distribution, check the Python version and platform before attempting a source build.

When reopening VS Code, reactivate the existing environment:

```bash
source .venv/bin/activate
```

## Run the game

```bash
python main.py
```

Click the game window to give it keyboard focus.

| Key | Action |
|---|---|
| WASD or arrow keys | Move |
| Shift while moving | Sprint and spend stamina |
| Control while moving | Sneak slowly without footsteps |
| V | Toggle restricted player visibility and the orb hint |
| H | Toggle the relative belief heatmap |
| P | Toggle the A* route display |
| B | Cycle hybrid, probabilistic, and baseline modes and restart |
| R | Restart with the same seed and orb sequence |
| N | Generate a new seed and restart |
| Escape | Quit |

The initial game seed is 42. The same seed repeats the orb sequence, but human inputs and frame timing can change the round. The orb arrow points directly toward the objective, not along a route around walls. The heatmap represents relative belief strength, not absolute percentages. Footsteps are simulated sensory events without audible sound effects.

Each win or loss appends a row to `session_results.csv`. Restarting or quitting an unfinished round does not log that round. Local playtest records are not included in the repository and are not controlled comparisons.

## Reproduce the report's experiment

From the project folder, run:

```bash
python evaluate_realistic.py
```

This opens no game window. It generates 40 scenarios using seed 2026 and runs four strategies on each scenario, producing 160 trials. Twenty scenarios contain a hiding player and twenty contain a moving player.

Settings include a 12-second search budget, a 0.05-second timestep, configured enemy/player speeds of 240/220 pixels per second, a 240-pixel vision range, and a one-second commitment limit. The enemy remains fixed during the constructed initial unseen interval. During evaluation, it makes movement decisions at tile centers, which can reduce its effective speed slightly.

The command writes or replaces these files beside the script:

- `evaluation_realistic_results.csv`: one row per scenario and strategy.
- `evaluation_realistic_summary.csv`: results for all, hiding, and moving scenarios.
- `evaluation_realistic_pairs.csv`: hybrid's paired comparisons with each alternative.

Unchanged copies of the submitted final results are included in `reference_results/`, allowing comparison after a rerun.

### Expected results

| Strategy | Reacquired | Rate | Mean capped search time |
|---|---:|---:|---:|
| Baseline | 15/40 | 37.5% | 8.21 s |
| Probabilistic | 38/40 | 95% | 3.22 s |
| Hybrid without commitment | 38/40 | 95% | 3.14 s |
| Hybrid | 38/40 | 95% | 3.02 s |

The displayed times follow the report's two-decimal formatting. CSVs retain full precision.

Hybrid's paired outcomes are:

- Against baseline: 28 wins, 0 losses, 12 ties.
- Against probabilistic: 6 wins, 9 losses, 25 ties.
- Against hybrid without commitment: 5 wins, 9 losses, 26 ties.

The experiment is deterministic in the author's environment. Exact timings may vary with software/platform differences. If results differ, first check the seed, settings, dependency version, and whether the supplied scripts were edited.

## Metrics and interpretation

- **Reacquisition rate:** successful trials divided by the scenario count.
- **Reacquisition time:** simulated seconds until visibility returns. A failed trial has a blank time, not an observed success at 12 seconds.
- **Mean capped search time:** includes failures as the full 12-second budget.
- **Median successful time:** includes only successful trials and should be read alongside success rate.
- **Paired outcome:** success takes priority, then earlier reacquisition. Two failures tie.

Probability-based search outperformed the last-location baseline on this test set. Hybrid had the lowest capped mean, but its commitment rule did not consistently win paired comparisons. Results concern one map, one scenario-generation seed, and scripted routes. No statistical significance test or player study was performed.

## Files and development history

| File | Purpose |
|---|---|
| `main.py` | Interactive game, sensing, movement, objectives, and round logging |
| `pathfinding.py` | A*, open-neighbor lookup, and Manhattan heuristic |
| `belief.py` | Location belief, Markov prediction, and observation updates |
| `evaluate_realistic.py` | Final controlled experiment used in the report |
| `evaluate_expanded.py` | Earlier expanded experiment that exposed a ceiling effect |
| `evaluate.py` | Earlier four-scenario comparison including hybrid |
| `evaluation_*.csv` | Full generated experimental results and summaries |
| `session_results.csv` | Locally generated gameplay records (excluded from Git) |
| `reference_results/` | Preserved copies of the final experiment's three CSVs |


To rerun the earlier experiments:

```bash
python evaluate.py
python evaluate_expanded.py
```

`evaluate.py` writes `evaluation_hybrid_results.csv`. The historical `evaluation_results.csv` came from the initial two-strategy version before hybrid was added; the supplied `evaluate.py` is the later three-strategy version. `evaluate_expanded.py` writes the expanded results and summary. These earlier experiments use different movement and scenario settings and should not be treated as replications of the final experiment.

## Research report

A separate academic report documenting the project's methodology, experimental results, and limitations is available upon request.

## Data and attribution

The dataset consists of generated scenarios and their recorded outcomes. All supplied CSVs are small, so the package includes the full results rather than a sample. Scenario routes are regenerated from the seeded evaluator and its map. No external human dataset is used. Local playtest records contain round statistics without player names.

The project adapts established AI techniques. AI assistance supported implementation, debugging, and writing. The report explains the project-specific design choices, evaluation revisions, and limitations. Pygame Community Edition supplies the game framework.

Scholarly foundations cited in the report:

1. P. E. Hart, N. J. Nilsson, and B. Raphael, “A formal basis for the heuristic determination of minimum cost paths,” 1968. https://doi.org/10.1109/TSSC.1968.300136
2. D. Fox, W. Burgard, and S. Thrun, “Markov localization for mobile robots in dynamic environments,” 1999. https://doi.org/10.1613/jair.616
3. F. Dellaert, D. Fox, W. Burgard, and S. Thrun, “Monte Carlo localization for mobile robots,” 1999. https://doi.org/10.1109/ROBOT.1999.772544
