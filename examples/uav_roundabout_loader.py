"""Utilities for loading UAV roundabout car-count tasks."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class UavCarCountTask:
    """Simple record describing a single UAV car-count question."""

    image_name: str
    image_path: str
    num_cars: int
    question: str


_REQUIRED_COLUMNS = {"image_name", "num_cars", "image_path"}


def load_uav_roundabout_tasks(
    csv_path: str = "uav_roundabout_tasks.csv",
) -> List[UavCarCountTask]:
    """Load the roundabout tasks CSV into structured task records."""

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"UAV task file not found: {path}")

    tasks: List[UavCarCountTask] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("UAV task CSV missing header row")
        missing = _REQUIRED_COLUMNS.difference({name.strip() for name in reader.fieldnames})
        if missing:
            raise ValueError(f"UAV task CSV missing required columns: {sorted(missing)}")
        for row in reader:
            try:
                image_name = row["image_name"].strip()
                image_path = row["image_path"].strip()
                num_cars = int(row["num_cars"].strip())
            except (KeyError, ValueError, AttributeError) as exc:
                raise ValueError(f"Invalid UAV task row: {row}") from exc
            question = (
                f"In the image at path '{image_path}', how many cars are present in the scene?"
            )
            tasks.append(
                UavCarCountTask(
                    image_name=image_name,
                    image_path=image_path,
                    num_cars=num_cars,
                    question=question,
                )
            )
    return tasks


if __name__ == "__main__":
    all_tasks = load_uav_roundabout_tasks()
    print(f"Loaded {len(all_tasks)} UAV car-count tasks.")
    preview = all_tasks[:5]
    for task in preview:
        print(f"- {task.image_name}: {task.num_cars} cars @ {task.image_path}")
