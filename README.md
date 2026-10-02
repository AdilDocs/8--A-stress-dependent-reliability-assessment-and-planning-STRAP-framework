# STRAP — Source Code

Code, data and results for the paper on STRAP (STress-dependent Reliability Assessment and Planning) for composite power systems.

## Requirements

Python 3.11 with:

```
numpy>=2.0  scipy>=1.12  pandas>=2.0  matplotlib>=3.8  highspy>=1.7  numba>=0.59
```

```bash
pip install numpy scipy pandas matplotlib highspy numba
export PYTHONPATH=$PWD        # run all scripts from this folder
```

## Folder layout

| File | Content |
|---|---|
| `config.py` | Parameters of Table `tab:params` (reliability data, stress-factor ranges, κ, γ, a, ε, storage and cost data, seeds) |
| `data/*.m` | PGLib-OPF cases (IEEE 39, 57, 118 bus) |
| `data_sources.py` | RTS-79 hourly load model, unit data, PGLib parser |
| `model.py` | Stress index, hazard, β calibration, peak calibration, sequential simulation, DC load-shedding LP, adequacy indices |
| `methods.py` | Conventional and comparison methods (COPT, MCS, Weibull, hour-of-day, seasonal, weather, two-level, load-dependent), maximum-likelihood estimation of κ and γ |
| `planning.py` | Storage-planning objective J |
| `optimizers.py` | jDE and the comparison optimizers |
| `run_*.py` | Experiments (write CSV files to `results/`) |
| `make_*.py` | Tables and figures from the CSV files |
| `results/` | All results used in the paper |

## Reproducing the results

The `results/` folder already holds every output, so the tables and figures can be rebuilt directly (step 2). Step 1 repeats the experiments.

**Step 1 — experiments** (approximate run time on two cores)

```bash
python run_main.py 1000            # 4 model variants x 3 systems, 1000 years      (several hours)
python run_planning.py             # storage planning and optimizer runs             (several hours)
python run_sensitivity.py 1500     # sensitivity to kappa, gamma, a                  (~2 h)
python run_benchmark_hl1.py 20     # known-truth benchmark, HL-I, 20 histories       (~1 h)
python run_benchmark_hl2.py        # known-truth benchmark, HL-II, 5 histories       (~2 h)
python run_benchmark_extra.py      # Weibull, hour-of-day, weather, load-dependent   (~1.5 h)
```

`run_main.py` also accepts a system list, e.g. `python run_main.py 1000 IEEE39,IEEE57`.

**Step 2 — tables and figures**

```bash
python make_tables.py
python make_results_tables.py
python make_comparison_table.py
python make_figures.py
python make_framework_fig.py
```

## Paper items and their sources

| Paper item | Script | Output |
|---|---|---|
| Fig. `fig:method` (framework) | `make_framework_fig.py` | `results/figures/method_strap.png/.pdf` |
| Table `tab:params` | `make_results_tables.py` | `results/tables/T01_parameters.*` |
| Table `tab:systems` | `make_tables.py` | `results/tables/table1_systems.*` |
| Table `tab:main` | `make_tables.py` | `results/tables/table3_main.*` |
| Table `tab:effects` | `make_tables.py` | `results/tables/table4_effects.*` |
| Fig. `fig:effects` | `make_figures.py` | `results/figures/fig_effect_sizes.*` |
| Fig. `fig:ccdf` | `make_figures.py` | `results/figures/fig_eens_ccdf.*` |
| Table `tab:components` | `make_results_tables.py` | `results/tables/T09_components.*` |
| Table `tab:loadpoints` | `make_results_tables.py` | `results/tables/T10_loadpoints.*` |
| Table `tab:optset` | `make_results_tables.py` | `results/tables/T16_opt_settings.tex` |
| Table `tab:optimizers` | `make_tables.py` | `results/tables/table5_optimizers.*` |
| Fig. `fig:optimizers` | `make_figures.py` | `results/figures/fig_optimizers.*` |
| Table `tab:sensitivity` | `make_results_tables.py` | `results/tables/T14_sensitivity.*` |
| Fig. `fig:sensitivity` | `make_figures.py` | `results/figures/fig_sensitivity.*` |
| Table `tab:compare` | `make_comparison_table.py` | `results/tables/tab_compare.csv` |

The scripts also write some supplementary outputs that the paper does not use: `T07`, `T08`, `T11`, `T12`, `T13`, `T15`, `T17`, `table6_planning` and `fig_benchmark`.

## Reproducibility notes

- Component stress factors are drawn once with a fixed seed (`config.py`), so all experiments use the same component population.
- Model variants share common random numbers (failure thresholds and repair draws) in each simulated year, so their differences are paired.
- CPU times depend on the machine. The values in the paper were measured on two cores of an Intel Xeon at 2.10 GHz.
