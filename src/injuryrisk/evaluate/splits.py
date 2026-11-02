"""Forward-chaining splits by season (spec 6.2).

For test season S: validation = the last usable season before S, train = every
usable season before that. Hyperparameters are chosen on validation, then the
model is refit on train + validation and scored once on S. Never a random
split, never a split date inside a season.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Split:
    train: tuple[int, ...]
    val: int
    test: int

    @property
    def fit(self) -> tuple[int, ...]:
        """Seasons for the final refit: train + validation."""
        return self.train + (self.val,)


def usable_seasons(cfg: dict, available: list[int]) -> list[int]:
    """Labelled seasons, minus COVID seasons when ``covid_mode: exclude``."""
    last = int(cfg["label_last_season"])
    covid = set(cfg.get("covid_seasons", []))
    out = [s for s in sorted(available) if s <= last]
    if cfg.get("covid_mode", "exclude") == "exclude":
        out = [s for s in out if s not in covid]
    return out


def forward_chain(cfg: dict, available: list[int]) -> list[Split]:
    seasons = usable_seasons(cfg, available)
    covid = set(cfg.get("covid_seasons", []))
    min_train = int(cfg["split"]["min_train_seasons"])
    splits = []
    for test in cfg["split"]["test_seasons"]:
        test = int(test)
        if test in covid:
            raise ValueError(f"{test} is a COVID season; it cannot be a test season")
        if test not in seasons:
            raise ValueError(f"test season {test} has no labels or is excluded")
        prior = [s for s in seasons if s < test]
        if len(prior) < min_train + 1:
            print(f"  skip test {test}: only {len(prior)} prior seasons")
            continue
        splits.append(Split(train=tuple(prior[:-1]), val=prior[-1], test=test))
    return splits
