"""Loading and validation helpers for the frozen ten-task benchmark."""

import json
from typing import Any, Dict, List


EXPECTED_TASK_IDS = (
    "factory_01",
    "factory_02",
    "observer_01",
    "observer_02",
    "strategy_01",
    "strategy_02",
    "singleton_01",
    "singleton_02",
    "adapter_01",
    "adapter_02",
)


def validate_contract(task: Dict[str, Any]) -> None:
    """Raise a clear error when a benchmark task has no usable contract."""
    task_id = task.get("task_id", "<unknown>")
    contract = task.get("contract")
    if not isinstance(contract, dict):
        raise ValueError(f"{task_id}: contract must be a dictionary")

    for class_info in contract.get("classes", []):
        if not isinstance(class_info, dict) or not class_info.get("name"):
            raise ValueError(f"{task_id}: every class needs a name")

    class_names = {item["name"] for item in contract.get("classes", [])}
    for method in contract.get("methods", []):
        if not isinstance(method, dict):
            raise ValueError(f"{task_id}: every method must be a dictionary")
        owner = method.get("class")
        if owner not in class_names:
            raise ValueError(f"{task_id}: method owner {owner!r} is not declared")
        if not method.get("name"):
            raise ValueError(f"{task_id}: every method needs a name")
        for parameter in method.get("parameters", []):
            if not parameter.get("name"):
                raise ValueError(f"{task_id}: every parameter needs a name")


def load_tasks(dataset_path: str) -> List[Dict[str, Any]]:
    """Load and validate the complete benchmark without contacting an API."""
    with open(dataset_path, "r", encoding="utf-8") as handle:
        tasks = json.load(handle)

    if not isinstance(tasks, list):
        raise ValueError("benchmark dataset must contain a list of tasks")
    task_ids = tuple(task.get("task_id") for task in tasks)
    if task_ids != EXPECTED_TASK_IDS:
        raise ValueError(
            f"benchmark must contain the ten frozen task IDs in order; got {task_ids}"
        )
    for task in tasks:
        if not task.get("description") or not task.get("pattern"):
            raise ValueError(f"{task.get('task_id', '<unknown>')}: missing task metadata")
        validate_contract(task)
        task["contract"]["pattern"] = task["pattern"]
    return tasks


def partial_skeleton(contract: Dict[str, Any]) -> str:
    """Return the same deliberately incomplete class skeleton for both prompts."""
    lines = [
        "PARTIAL IMPLEMENTATION SKELETON",
        "Complete the following class declarations; do not treat this as a full solution:",
    ]
    for class_info in contract.get("classes", []):
        lines.extend((f"class {class_info['name']}:", "    pass", ""))
    return "\n".join(lines).rstrip()
