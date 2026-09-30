from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from gsi.llm.client import ModelSpec
from gsi.prompts.modes import LIBRARIES, libraries_for

DEFAULT_CANONICAL = {
    "graphqa": {"structure": "edge_list", "syntax": "graphqa_nl"},
    "erdos": {"structure": "edge_list", "syntax": "erdos_nl"},
}


@dataclass
class ExperimentConfig:
    name: str
    root: Path
    datasets: dict[str, dict]
    variants: dict
    modes: list[str]
    models: list[str]
    libraries: list[str] = field(default_factory=lambda: list(LIBRARIES))
    library_tasks: dict[str, list[str]] = field(default_factory=dict)
    exec: dict = field(default_factory=dict)
    paths: dict = field(default_factory=dict)
    inject_graph_type: bool = False   # state "undirected" explicitly in M3 prompts (see prompts.modes)

    @property
    def data_dir(self) -> Path:
        return self.root / self.paths.get("data", "data/processed") / self.name

    @property
    def results_dir(self) -> Path:
        return self.root / self.paths.get("results", "results") / self.name

    @property
    def cache_dir(self) -> Path:
        return self.root / self.paths.get("results", "results") / "cache"

    @property
    def models_file(self) -> Path:
        return self.root / self.paths.get("models", "configs/models.yaml")

    def canonical_for(self, dataset: str) -> dict:
        return {**DEFAULT_CANONICAL.get(dataset, {}), **self.datasets[dataset].get("canonical", {})}

    def variants_for(self, dataset: str) -> dict:
        return self.datasets[dataset].get("variants", self.variants)

    def arms_for(self, mode: str, task: str) -> list[str | None]:
        """The (mode, library) arms to run for a task. `direct` has no library arm; a
        library listed in `library_tasks` runs only on the tasks named there, which is how
        the native arm can be kept off tasks where hand-written code has no chance."""
        arms = libraries_for(mode, self.libraries)
        return [lib for lib in arms
                if lib is None or lib not in self.library_tasks or task in self.library_tasks[lib]]


CONFIG_KEYS = {"name", "root", "datasets", "variants", "modes", "models", "libraries", "library_tasks", "exec",
               "paths", "inject_graph_type"}


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).resolve()
    with path.open() as f:
        raw = yaml.safe_load(f)
    unknown = sorted(set(raw) - CONFIG_KEYS)
    if unknown:  # a misspelt key would otherwise leave its setting at the default without a word
        raise ValueError(f"{path}: unknown config key(s) {unknown}; known keys are {sorted(CONFIG_KEYS)}")
    root = Path(raw.get("root", path.parent.parent)).resolve()
    return ExperimentConfig(
        name=raw["name"], root=root, datasets=raw["datasets"], variants=raw.get("variants", {}),
        modes=list(raw["modes"]), models=list(raw["models"]),
        libraries=list(raw.get("libraries", LIBRARIES)),
        library_tasks={k: list(v) for k, v in (raw.get("library_tasks") or {}).items()},
        exec=raw.get("exec", {}), paths=raw.get("paths", {}),
        inject_graph_type=bool(raw.get("inject_graph_type", False)),
    )


def load_models(path: str | Path) -> dict[str, ModelSpec]:
    with Path(path).open() as f:
        raw = yaml.safe_load(f)
    return {name: ModelSpec.from_dict(name, d or {}) for name, d in raw.items()}
