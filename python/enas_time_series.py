#!/usr/bin/env python3
"""Educational evolutionary NAS for univariate time-series prediction.

This script is a Python companion to the MATLAB implementation in NAS_FC.m.
It keeps the same teaching objective: use an evolutionary search to explore
simple neural-network architectures for time-series forecasting.

The Python version searches over two architecture choices:

* number of lagged time steps used as inputs
* number of hidden neurons in a one-hidden-layer MLP regressor

It is not intended to reproduce paper-level numerical results. It is a small,
inspectable baseline for learning, experimentation, and smoke testing.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor


BIT_WIDTH = 5


@dataclass(frozen=True)
class SearchConfig:
    population_size: int = 20
    generations: int = 5
    elitism_rate: float = 0.2
    crossover_probability: float = 0.9
    mutation_probability: float = 0.1
    min_lags: int = 1
    max_lags: int = 20
    min_neurons: int = 1
    max_neurons: int = 20
    max_iter: int = 300
    seed: int = 1
    population_init: str = "random"


@dataclass(frozen=True)
class Individual:
    genotype: tuple[int, ...]

    def decode(self, config: SearchConfig) -> tuple[int, int]:
        """Decode a 10-bit Gray-code genotype into neurons and lags."""
        if len(self.genotype) != 2 * BIT_WIDTH:
            raise ValueError("A genotype must contain 10 bits.")

        neurons = gray_bits_to_int(self.genotype[:BIT_WIDTH])
        lags = gray_bits_to_int(self.genotype[BIT_WIDTH:])

        neurons = int(np.clip(neurons, config.min_neurons, config.max_neurons))
        lags = int(np.clip(lags, config.min_lags, config.max_lags))
        return neurons, lags


@dataclass(frozen=True)
class Evaluation:
    generation: int
    member: int
    individual: Individual
    neurons: int
    lags: int
    fitness: float
    validation_r: float
    validation_mse: float
    test_r: float
    test_mse: float


def int_to_gray_bits(value: int, width: int = BIT_WIDTH) -> tuple[int, ...]:
    """Encode a non-negative integer as a fixed-width Gray-code bit tuple."""
    if value < 0 or value >= 2**width:
        raise ValueError(f"value must be in [0, {2**width - 1}]")

    gray = value ^ (value >> 1)
    return tuple((gray >> shift) & 1 for shift in range(width - 1, -1, -1))


def gray_bits_to_int(bits: Iterable[int]) -> int:
    """Decode a Gray-code bit sequence into an integer."""
    gray_bits = [int(bit) for bit in bits]
    if not gray_bits:
        raise ValueError("Cannot decode an empty Gray-code sequence.")

    binary_bits = [gray_bits[0]]
    for bit in gray_bits[1:]:
        binary_bits.append(binary_bits[-1] ^ bit)

    value = 0
    for bit in binary_bits:
        value = (value << 1) | bit
    return value


def make_individual(neurons: int, lags: int) -> Individual:
    """Create an encoded individual from decoded architecture values."""
    genotype = int_to_gray_bits(neurons) + int_to_gray_bits(lags)
    return Individual(genotype=genotype)


def load_series(csv_path: Path, value_column: Optional[str]) -> np.ndarray:
    """Load the target series from a CSV file.

    By default, the second column is used to match the MATLAB script. The
    optional value_column argument can be either a column name or a zero-based
    column index represented as text.
    """
    data = pd.read_csv(csv_path)
    if data.shape[1] < 2 and value_column is None:
        raise ValueError("The CSV must contain at least two columns.")

    if value_column is None:
        raw_values = data.iloc[:, 1]
    else:
        try:
            column_index = int(value_column)
        except ValueError:
            if value_column not in data.columns:
                raise ValueError(
                    f"Column '{value_column}' was not found in {csv_path}."
                ) from None
            raw_values = data[value_column]
        else:
            raw_values = data.iloc[:, column_index]

    values = pd.to_numeric(raw_values, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size < 50:
        raise ValueError("The selected series is too short after removing missing values.")
    return values


def make_demo_series(length: int, seed: int) -> np.ndarray:
    """Create a small synthetic series for smoke testing only."""
    rng = np.random.default_rng(seed)
    time = np.arange(length, dtype=float)
    seasonal = np.sin(2.0 * np.pi * time / 24.0)
    slower_cycle = 0.4 * np.sin(2.0 * np.pi * time / 168.0)
    trend = 0.001 * time
    noise = rng.normal(0.0, 0.08, size=length)
    return seasonal + slower_cycle + trend + noise


def make_lagged_samples(series: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Transform a univariate series into supervised lagged samples."""
    if lags < 1:
        raise ValueError("lags must be at least 1.")
    if len(series) <= lags + 10:
        raise ValueError("The series is too short for the requested lag count.")

    x_columns = [series[lags - offset : len(series) - offset] for offset in range(1, lags + 1)]
    x = np.column_stack(x_columns)
    y = series[lags:]
    return x, y


def temporal_train_validation_test_split(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split samples in chronological order."""
    if len(y) < 20:
        raise ValueError("At least 20 supervised samples are required.")

    train_end = int(len(y) * train_ratio)
    validation_end = int(len(y) * (train_ratio + validation_ratio))

    if train_end < 2 or validation_end <= train_end or validation_end >= len(y):
        raise ValueError("The series is too short for a 70/15/15 temporal split.")

    return (
        x[:train_end],
        y[:train_end],
        x[train_end:validation_end],
        y[train_end:validation_end],
        x[validation_end:],
        y[validation_end:],
    )


def standardize_from_train(
    train: np.ndarray,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    """Standardize arrays using statistics from the training split."""
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    scaled_train = (train - mean) / std
    scaled_arrays = [(array - mean) / std for array in arrays]
    return scaled_train, scaled_arrays, mean, std


def correlation_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return Pearson correlation, or 0.0 when it is undefined."""
    if actual.size < 2 or predicted.size < 2:
        return 0.0
    if np.std(actual) == 0.0 or np.std(predicted) == 0.0:
        return 0.0
    score = float(np.corrcoef(actual, predicted)[0, 1])
    if math.isnan(score):
        return 0.0
    return score


def evaluate_individual(
    series: np.ndarray,
    individual: Individual,
    config: SearchConfig,
    generation: int,
    member: int,
) -> Evaluation:
    """Train and evaluate one candidate architecture."""
    neurons, lags = individual.decode(config)
    x, y = make_lagged_samples(series, lags)
    x_train, y_train, x_validation, y_validation, x_test, y_test = (
        temporal_train_validation_test_split(x, y)
    )

    x_train_scaled, scaled_x, _, _ = standardize_from_train(
        x_train, x_validation, x_test
    )
    x_validation_scaled, x_test_scaled = scaled_x

    y_mean = float(y_train.mean())
    y_std = float(y_train.std()) or 1.0
    y_train_scaled = (y_train - y_mean) / y_std

    random_state = config.seed + generation * 1000 + member
    model = MLPRegressor(
        hidden_layer_sizes=(neurons,),
        activation="relu",
        solver="adam",
        max_iter=config.max_iter,
        random_state=random_state,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(x_train_scaled, y_train_scaled)

    validation_predictions = model.predict(x_validation_scaled) * y_std + y_mean
    test_predictions = model.predict(x_test_scaled) * y_std + y_mean

    validation_r = correlation_score(y_validation, validation_predictions)
    test_r = correlation_score(y_test, test_predictions)
    validation_mse = float(mean_squared_error(y_validation, validation_predictions))
    test_mse = float(mean_squared_error(y_test, test_predictions))

    return Evaluation(
        generation=generation,
        member=member,
        individual=individual,
        neurons=neurons,
        lags=lags,
        fitness=validation_r,
        validation_r=validation_r,
        validation_mse=validation_mse,
        test_r=test_r,
        test_mse=test_mse,
    )


def initialize_population(config: SearchConfig, rng: np.random.Generator) -> list[Individual]:
    """Create the initial population."""
    if config.population_init == "uniform":
        neurons_values = np.linspace(
            config.min_neurons, config.max_neurons, config.population_size
        )
        lag_values = np.linspace(config.min_lags, config.max_lags, config.population_size)
        return [
            make_individual(int(round(neurons)), int(round(lags)))
            for neurons, lags in zip(neurons_values, lag_values)
        ]

    return [
        make_individual(
            int(rng.integers(config.min_neurons, config.max_neurons + 1)),
            int(rng.integers(config.min_lags, config.max_lags + 1)),
        )
        for _ in range(config.population_size)
    ]


def mutate_bits(
    genotype: tuple[int, ...],
    mutation_probability: float,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Flip individual genotype bits with the configured probability."""
    return tuple(
        1 - bit if rng.random() < mutation_probability else bit for bit in genotype
    )


def crossover(
    parent_a: Individual,
    parent_b: Individual,
    rng: np.random.Generator,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Two-point crossover for equal-length binary genotypes."""
    length = len(parent_a.genotype)
    if length != len(parent_b.genotype):
        raise ValueError("Parents must have equal-length genotypes.")

    point_a, point_b = sorted(rng.choice(np.arange(1, length), size=2, replace=False))
    genotype_a = parent_a.genotype
    genotype_b = parent_b.genotype

    child_a = genotype_a[:point_a] + genotype_b[point_a:point_b] + genotype_a[point_b:]
    child_b = genotype_b[:point_a] + genotype_a[point_a:point_b] + genotype_b[point_b:]
    return child_a, child_b


def evolve_population(
    evaluations: list[Evaluation],
    config: SearchConfig,
    rng: np.random.Generator,
) -> list[Individual]:
    """Create the next generation from ranked evaluations."""
    ranked = sorted(evaluations, key=lambda result: result.fitness, reverse=True)
    elite_count = max(1, int(round(config.elitism_rate * config.population_size)))
    elite_count = min(elite_count, config.population_size)
    elites = [result.individual for result in ranked[:elite_count]]

    next_population = list(elites)
    while len(next_population) < config.population_size:
        parent_a = elites[int(rng.integers(0, len(elites)))]
        parent_b = elites[int(rng.integers(0, len(elites)))]

        if rng.random() < config.crossover_probability:
            child_a_bits, child_b_bits = crossover(parent_a, parent_b, rng)
        else:
            child_a_bits, child_b_bits = parent_a.genotype, parent_b.genotype

        child_a_bits = mutate_bits(child_a_bits, config.mutation_probability, rng)
        child_b_bits = mutate_bits(child_b_bits, config.mutation_probability, rng)

        next_population.append(Individual(child_a_bits))
        if len(next_population) < config.population_size:
            next_population.append(Individual(child_b_bits))

    return next_population


def run_search(series: np.ndarray, config: SearchConfig) -> list[Evaluation]:
    """Run the full evolutionary architecture search."""
    rng = np.random.default_rng(config.seed)
    population = initialize_population(config, rng)
    all_evaluations: list[Evaluation] = []

    for generation in range(1, config.generations + 1):
        evaluations = [
            evaluate_individual(series, individual, config, generation, member)
            for member, individual in enumerate(population, start=1)
        ]
        all_evaluations.extend(evaluations)

        ranked = sorted(evaluations, key=lambda result: result.fitness, reverse=True)
        best = ranked[0]
        mean_fitness = float(np.mean([result.fitness for result in evaluations]))
        print(
            "Generation "
            f"{generation}: best validation R={best.validation_r:.4f}, "
            f"test R={best.test_r:.4f}, neurons={best.neurons}, "
            f"lags={best.lags}, mean validation R={mean_fitness:.4f}"
        )

        if generation < config.generations:
            population = evolve_population(evaluations, config, rng)

    return all_evaluations


def evaluations_to_frame(evaluations: list[Evaluation]) -> pd.DataFrame:
    """Convert evaluations to a tabular result frame."""
    return pd.DataFrame(
        [
            {
                "generation": result.generation,
                "member": result.member,
                "neurons": result.neurons,
                "lags": result.lags,
                "fitness_validation_r": result.fitness,
                "validation_r": result.validation_r,
                "validation_mse": result.validation_mse,
                "test_r": result.test_r,
                "test_mse": result.test_mse,
                "genotype": "".join(str(bit) for bit in result.individual.genotype),
            }
            for result in evaluations
        ]
    )


def write_outputs(
    evaluations: list[Evaluation],
    output_dir: Path,
    write_plots: bool,
) -> None:
    """Save CSV/JSON outputs and optional plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = evaluations_to_frame(evaluations)
    frame.to_csv(output_dir / "generation_results.csv", index=False)

    best = max(evaluations, key=lambda result: result.fitness)
    best_summary = {
        "generation": best.generation,
        "member": best.member,
        "neurons": best.neurons,
        "lags": best.lags,
        "validation_r": best.validation_r,
        "validation_mse": best.validation_mse,
        "test_r": best.test_r,
        "test_mse": best.test_mse,
        "genotype": "".join(str(bit) for bit in best.individual.genotype),
    }
    (output_dir / "best_individual.json").write_text(
        json.dumps(best_summary, indent=2), encoding="utf-8"
    )

    if write_plots:
        write_result_plots(frame, output_dir)


def write_result_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    """Write simple diagnostic plots for the evolutionary run."""
    import matplotlib.pyplot as plt

    grouped = frame.groupby("generation")
    best_by_generation = grouped["validation_r"].max()
    mean_by_generation = grouped["validation_r"].mean()

    plt.figure(figsize=(8, 4))
    plt.plot(best_by_generation.index, best_by_generation.values, marker="o", label="Best")
    plt.plot(mean_by_generation.index, mean_by_generation.values, marker="s", label="Mean")
    plt.xlabel("Generation")
    plt.ylabel("Validation R")
    plt.title("Evolutionary search fitness")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fitness_by_generation.png", dpi=150)
    plt.close()

    last_generation = int(frame["generation"].max())
    final_population = frame[frame["generation"] == last_generation]
    plt.figure(figsize=(5, 5))
    plt.scatter(final_population["neurons"], final_population["lags"])
    plt.xlabel("Hidden neurons")
    plt.ylabel("Lags")
    plt.title("Final generation architectures")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "final_population.png", dpi=150)
    plt.close()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Educational evolutionary NAS for univariate time-series prediction."
    )
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--csv", type=Path, help="Path to a CSV time-series file.")
    data_group.add_argument(
        "--demo",
        action="store_true",
        help="Run a small synthetic demo series for smoke testing.",
    )
    parser.add_argument(
        "--value-column",
        help="Target column name or zero-based index. Defaults to the second column.",
    )
    parser.add_argument("--population-size", type=positive_int, default=20)
    parser.add_argument("--generations", type=positive_int, default=5)
    parser.add_argument("--elitism-rate", type=probability, default=0.2)
    parser.add_argument("--crossover-probability", type=probability, default=0.9)
    parser.add_argument("--mutation-probability", type=probability, default=0.1)
    parser.add_argument("--min-lags", type=positive_int, default=1)
    parser.add_argument("--max-lags", type=positive_int, default=20)
    parser.add_argument("--min-neurons", type=positive_int, default=1)
    parser.add_argument("--max-neurons", type=positive_int, default=20)
    parser.add_argument("--max-iter", type=positive_int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--population-init",
        choices=("random", "uniform"),
        default="random",
        help="Initial population strategy.",
    )
    parser.add_argument(
        "--demo-length",
        type=positive_int,
        default=500,
        help="Length of the synthetic demo series.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for CSV/JSON outputs and optional plots.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save diagnostic plots. Requires --output-dir.",
    )
    return parser.parse_args()


def validate_config(config: SearchConfig) -> None:
    if config.min_lags > config.max_lags:
        raise ValueError("--min-lags cannot be greater than --max-lags.")
    if config.min_neurons > config.max_neurons:
        raise ValueError("--min-neurons cannot be greater than --max-neurons.")
    if config.max_lags >= 2**BIT_WIDTH or config.max_neurons >= 2**BIT_WIDTH:
        raise ValueError("The 5-bit Gray-code encoding supports values up to 31.")


def main() -> None:
    args = parse_args()
    if args.plot and args.output_dir is None:
        raise SystemExit("--plot requires --output-dir so plots have a destination.")

    config = SearchConfig(
        population_size=args.population_size,
        generations=args.generations,
        elitism_rate=args.elitism_rate,
        crossover_probability=args.crossover_probability,
        mutation_probability=args.mutation_probability,
        min_lags=args.min_lags,
        max_lags=args.max_lags,
        min_neurons=args.min_neurons,
        max_neurons=args.max_neurons,
        max_iter=args.max_iter,
        seed=args.seed,
        population_init=args.population_init,
    )
    validate_config(config)

    if args.demo:
        series = make_demo_series(args.demo_length, config.seed)
        print("Running with a synthetic demo series. Do not report demo metrics as paper results.")
    else:
        series = load_series(args.csv, args.value_column)
        print(f"Loaded {len(series)} numeric observations from {args.csv}.")

    evaluations = run_search(series, config)
    best = max(evaluations, key=lambda result: result.fitness)
    print(
        "Best individual by validation R: "
        f"generation={best.generation}, member={best.member}, "
        f"validation R={best.validation_r:.4f}, test R={best.test_r:.4f}, "
        f"neurons={best.neurons}, lags={best.lags}, test MSE={best.test_mse:.6f}"
    )

    if args.output_dir is not None:
        write_outputs(evaluations, args.output_dir, args.plot)
        print(f"Wrote outputs to {args.output_dir}.")


if __name__ == "__main__":
    main()
