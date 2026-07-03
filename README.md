# ENAS-Time-Series

Educational Evolutionary Neural Architecture Search for Time Series Prediction.

This repository contains MATLAB research code associated with:

Escalona-Llaguno, M. I., & Sarmiento-Rosales, S. M. (2024). Educational Evolutionary Neural Architecture Search for Time Series Prediction. In IJCCI (pp. 234--241).

The project is intended as an accessible example of how evolutionary search can be used to explore neural-network architectures for univariate time-series forecasting. It is research and teaching code, not a production AutoML framework.

## Background

Neural Architecture Search (NAS) is a method for automatically exploring different neural-network designs instead of choosing every design detail by hand. In this repository, the searched design choices include the number of hidden neurons and the number of time lags used by a nonlinear autoregressive neural network.

Evolutionary NAS uses ideas from evolutionary algorithms. A population of candidate neural-network architectures is evaluated, the better candidates are selected, and new candidates are created through operations such as crossover and mutation.

Time-series prediction means using past values of a sequence to estimate future or held-out values. The script expects a univariate time series stored in a CSV file where the first column is a date/time column and the second column is the numeric variable to predict.

The educational aspect of this project is that the workflow is contained in one interactive MATLAB script. Users can see the main stages of the search process: data loading, population initialization, model training, evaluation, selection, crossover, mutation, and plotting.

## Main Features

- Interactive MATLAB script for evolutionary NAS on time-series data.
- Candidate architectures encoded with Gray-code binary strings.
- Search over hidden neurons and lag values.
- Default configuration for a small educational run.
- Optional custom configuration for population size, generations, elitism, crossover probability, and mutation probability.
- Plots showing the initial/final population and evaluation behavior across generations.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `NAS_FC.m` | Main MATLAB script. Loads data, runs the evolutionary search, trains/evaluates candidate NAR networks, and plots results. |
| `README.md` | Project overview, setup, usage, reproducibility notes, and citation. |
| `CITATION.cff` | Machine-readable citation metadata for GitHub and citation tools. |
| `.gitignore` | Local data, output, editor, and temporary-file ignore rules. |

## Requirements

This is a MATLAB project. It does not require Python dependencies.

Required software:

- MATLAB, preferably a current desktop or MATLAB Online version.
- MATLAB Deep Learning Toolbox, because the script uses functions such as `narnet`, `preparets`, `train`, `mapminmax`, and `mse`.

No GPU is explicitly required by the script. Runtime depends on the dataset length, population size, number of generations, and MATLAB/toolbox version.

## Data

Datasets are not included in this repository. The original README pointed users to hourly energy-consumption datasets available from:

- [Kaggle: Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption/data)
- [panambY/Hourly_Energy_Consumption data mirror](https://github.com/panambY/Hourly_Energy_Consumption/tree/master/data)

The script menu includes these expected CSV names:

- `PJM_Load_hourly.csv`
- `PJME_hourly.csv`
- `PJMW_hourly.csv`
- `NI_hourly.csv`
- `FE_hourly.csv`
- `EKPC_hourly.csv`
- `DUQ_hourly.csv`
- `DOM_hourly.csv`
- `DEOK_hourly.csv`
- `DAYTON_hourly.csv`
- `COMED_hourly.csv`
- `AEP_hourly.csv`

You may also provide a custom CSV file. The expected format is:

1. First column: date or timestamp.
2. Second column: numeric time-series value to predict.
3. File format: `.csv`.

Place the CSV file in the MATLAB current folder before running the script, or provide a path that MATLAB can resolve when prompted.

## Installation

Clone the repository:

```bash
git clone https://github.com/SergioSarmientoRosales/ENAS-Time-Series.git
cd ENAS-Time-Series
```

Then download the dataset files you want to use and place them in the repository folder, or keep them in another folder and provide the path when using the custom dataset option.

In MATLAB, set the current folder to the repository directory:

```matlab
cd path/to/ENAS-Time-Series
```

## Quick Start

In MATLAB, run:

```matlab
NAS_FC
```

Recommended first run:

1. Select one of the dataset names from the menu, or select `Custom Dataset`.
2. Choose `Use default values`.
3. Select either `Random Population` or `Uniform Population`.
4. Wait for the script to train and evaluate the candidate networks.
5. Use the final plotting menu to inspect the search behavior.

The default settings are:

| Parameter | Default |
| --- | ---: |
| Initial population | 20 |
| Generations | 5 |
| Elitism rate | 0.2 |
| Crossover probability | 0.9 |
| Mutation probability | 0.1 |
| Hidden-neuron search range | 1 to 20 |
| Lag search range | 1 to 20 |

## Running Experiments

The script is interactive. It prompts for:

- Dataset selection.
- Default or custom evolutionary parameters.
- Population initialization method.
- Final plot selection.

During execution, the command window reports each generation, each trained individual, its test correlation coefficient `R`, hidden neurons, and lag value. The script also computes test mean squared error internally for each individual.

The script does not save result files by default. Outputs are shown in the MATLAB command window and figures. If you need persistent experiment records, save the command-window output and figures manually, or extend the script with explicit result-saving code.

## Expected Outputs

Typical outputs include:

- Command-window progress logs for each generation and individual.
- A reported best individual with generation, `R`, hidden neurons, and lags.
- Scatter plots for initial and final populations.
- Histograms of hidden-neuron and lag frequencies.
- Optional plots of `R` values across generations.

No paper-level numerical results are reproduced automatically from a clean clone because the datasets are external and the repository does not include a scripted reproduction pipeline.

## Reproducibility Notes

- The script sets `rng(1)` near the beginning to make MATLAB random-number generation more reproducible.
- The neural network uses block-based train/validation/test splitting through `divideblock`.
- Results can still vary across MATLAB releases, toolbox versions, hardware, and training settings.
- Record the dataset file, dataset source/version, MATLAB version, toolbox version, selected menu options, and all custom parameters when reporting results.
- Use the default values for a first run before changing population size, generation count, or probabilities.
- The current code is an interactive research script. It is useful for education and exploration, but it is not organized as a fully automated benchmark suite.

## Troubleshooting

`Unable to find or open ...csv`

MATLAB cannot find the selected dataset. Place the CSV file in the current folder, or use the custom dataset option and provide a resolvable path.

`Undefined function 'narnet'`, `preparets`, or `mapminmax`

Install or enable MATLAB Deep Learning Toolbox. These functions are not part of base MATLAB.

Errors converting the second column to numeric values

The script uses the second CSV column as the numeric target series. Check that the second column contains numeric values and that missing or text values have been cleaned.

Very long runtime

Reduce the number of individuals or generations. Each candidate architecture trains a neural network, so runtime grows quickly.

Indexing errors during crossover or mutation

This may indicate that the evolutionary update loop needs review for the selected population settings. This documentation update does not change algorithm logic; review the population-size and indexing assumptions in `NAS_FC.m` before using modified settings or publication-grade runs.

## Citation

If you use this repository or the associated work, please cite:

Escalona-Llaguno, M. I., & Sarmiento-Rosales, S. M. (2024). Educational Evolutionary Neural Architecture Search for Time Series Prediction. In IJCCI (pp. 234--241).

```bibtex
@inproceedings{escalona2024educational,
  title={Educational Evolutionary Neural Architecture Search for Time Series Prediction},
  author={Escalona-Llaguno, M. I. and Sarmiento-Rosales, S. M.},
  booktitle={IJCCI},
  pages={234--241},
  year={2024}
}
```

## Contact

For repository questions, open an issue on GitHub:

https://github.com/SergioSarmientoRosales/ENAS-Time-Series/issues
