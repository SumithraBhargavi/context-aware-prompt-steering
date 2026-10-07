"""CAPS implementation demo.

This file is a FACULTY DEMONSTRATION, not experimental data.
It shows the complete evaluation path using two deterministic example
outputs for Factory task 01.

The example baseline is intentionally structurally non-conforming but
behaviorally usable; the example CAPS output follows the contract exactly.
Do not include these examples as LLM generations in the research dataset.
"""

import json
from pathlib import Path

from src.context_engine import ContextEngine
from src.evaluator import CodeEvaluator


BASE_DIR = Path(__file__).resolve().parent
TASKS = json.loads(
    (BASE_DIR / "dataset" / "test_tasks.json").read_text(encoding="utf-8")
)
TASK = next(t for t in TASKS if t["task_id"] == "factory_01")


BASELINE_EXAMPLE = """from abc import ABC, abstractmethod

class Notification(ABC):
    @abstractmethod
    def send(self, message: str = "") -> None:
        pass

class EmailNotification(Notification):
    def send(self, message: str = "") -> None:
        print("Sending email")

class SMSNotification(Notification):
    def send(self, message: str = "") -> None:
        print("Sending SMS")

class NotificationFactory:
    @staticmethod
    def create_notification(notification_type: str) -> Notification:
        if notification_type.lower() == "email":
            return EmailNotification()
        if notification_type.lower() == "sms":
            return SMSNotification()
        raise ValueError("Unsupported notification type")
"""


CAPS_EXAMPLE = """from abc import ABC, abstractmethod

class Notification(ABC):
    @abstractmethod
    def send(self) -> None:
        pass

class EmailNotification(Notification):
    def send(self) -> None:
        print("Sending email")

class SMSNotification(Notification):
    def send(self) -> None:
        print("Sending SMS")

class NotificationFactory:
    @staticmethod
    def create_notification(notification_type: str) -> Notification:
        if notification_type.lower() == "email":
            return EmailNotification()
        if notification_type.lower() == "sms":
            return SMSNotification()
        raise ValueError("Unsupported notification type")
"""


def evaluate(label: str, code: str) -> None:
    result = CodeEvaluator(TASK["contract"]).evaluate_full(code, TASK["task_id"])
    structural = result["structural"]
    functional = result["functional"]

    print(f"\n--- {label} ---")
    print(f"AST valid:              {structural['ast_valid']}")
    print(f"Interface conformance:  {structural['interface_conformance']:.1%}")
    print(f"Structural violations:  {structural['violation_count']}")
    print(f"Functional score:       {functional['functional_score']:.1%}")
    print(f"Functional valid:       {functional['functional_valid']}")

    if structural["violations"]:
        print("Structural issue:")
        print("  - " + structural["violations"][0])


def main() -> None:
    contract = ContextEngine(TASK["contract"]).generate_context()

    print("=" * 68)
    print("CONTEXT-AWARE PROMPT STEERING (CAPS) — FACULTY DEMO")
    print("=" * 68)
    print("\n1. Context Engine extracted this structural contract:")
    print(contract)

    print("\n2. Evaluation uses the SAME evaluator for both conditions.")
    evaluate("Baseline example", BASELINE_EXAMPLE)
    evaluate("CAPS example", CAPS_EXAMPLE)

    print("\nDemo complete.")
    print("These two code samples are controlled examples, not LLM experiment runs.")


if __name__ == "__main__":
    main()
