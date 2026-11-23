# Shared-capacity malware containment

Code and results for *Shared-Capacity Containment in an SLBS Malware Model: Critical Capacity and Sampled-Data Control*, by Xiaofan Yang and Lu-Xing Yang.

Install the dependencies in a Python 3.12 environment:

```sh
python -m pip install -r requirements.txt
python -m pytest -q
python run.py main
python run.py outbreak
```

By default, `run.py main` uses the balanced propagation scenario and `run.py outbreak` uses the Code Red fit through 15:12 UTC at half capacity. Both commands write costs, defense settings, and host trajectories to `outputs/`; use `--output PATH` to choose another directory.

| Paper | To reproduce | Command |
|---|---|---|
| §7.1, Appendix A | Compare containment policies across nine propagation regimes | `python run.py main --all` |
| §7.2, Appendix C | Compare responses under eleven Code Red settings, including reusing the central schedule under alternative fits | `python run.py outbreak --all` |
| §7.3, Appendix A | Vary installed capacity, workload, incident composition or update interval | `python studies.py capacity --all`; select `capacity_load`, `initial` or `calendar` for the other comparisons |
| §7.4, Appendix A | Test detection delay, observation errors or random host transitions | `python studies.py detection --all`; select `robustness` or `finite` for the other comparisons |
| §6.3, §7.4, Appendix B | Compare fitted updates on held-out and extrapolation scenarios | `python studies.py surrogate --all` |

`python studies.py --help` lists the remaining studies. To refit the predictors, use `python studies.py train` (add `--regenerate-experts` to recompute the training schedules); evaluate with `python studies.py surrogate --all --assets outputs/train/learning_assets.joblib`.

## Code Red observations

The outbreak comparison uses the fitted inputs in [settings.json](settings.json). To repeat the fit, download `codered-july.start.cdf.txt.gz` and `codered-july.end.cdf.txt.gz` from the dataset below. Run `python fitting.py --data /path/to/summaries --all`. At each minute the script subtracts cumulative last sightings from cumulative first sightings. It also accepts a CSV with columns `time,observed`, with time in hours since 10:00 UTC on 19 July 2001.

Source: David Moore and Colleen Shannon, [The CAIDA UCSD Dataset on the Code-Red Worms – 19 July 2001](https://catalog.caida.org/dataset/telescope_codered_worm). The software uses the [MIT License](LICENSE); the source data have a separate [CAIDA agreement](https://www.caida.org/about/legal/aua/public_aua/), which requires users to cite the dataset and report publications using it to CAIDA.

## Files

- [model.py](slbs/model.py) defines the dynamics, threshold and cost (§§3–5); [projection.py](slbs/projection.py) and [solver.py](slbs/solver.py) compute the sampled-data schedule (§§5–6).
- [strategies.py](slbs/strategies.py) implements the baselines (§6.2). [stochastic.py](slbs/stochastic.py) simulates individual host transitions (Appendix A); [learning.py](slbs/learning.py) fits and applies the surrogate policies (Appendix B).
- [settings.json](settings.json) specifies the scenarios and fits. `learning_assets.joblib` contains three predictors fitted to simulated SLBS scenarios, their training samples and the PMP trajectories.
- In [results/](results/), `main_grid.csv` and `practical_strategies.csv` give the §7.1 comparisons. The outbreak and fitting CSVs give the §7.2 results; `supporting_results.json` contains the sensitivity, observation-error, finite-population, and fitted-update comparisons.
