import json

from src.evaluator import CodeEvaluator


# ---------------------------------------------------------
# Load one task from our benchmark
# ---------------------------------------------------------

with open("dataset/test_tasks.json", "r", encoding="utf-8") as f:
    tasks = json.load(f)

task = tasks[0]

contract = task["contract"]

evaluator = CodeEvaluator(contract)


# ---------------------------------------------------------
# TEST 1: Correct code
# ---------------------------------------------------------

correct_code = """
from abc import ABC, abstractmethod


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
    def create_notification(
        notification_type: str
    ) -> Notification:

        if notification_type == "email":
            return EmailNotification()

        return SMSNotification()
"""


print("=" * 60)
print("TEST 1: CORRECT CODE")
print("=" * 60)

result = evaluator.evaluate(correct_code)

print("AST Valid:", result["ast_valid"])
print(
    "Interface Conformance:",
    f"{result['interface_conformance'] * 100:.2f}%"
)
print("Violations:")

for violation in result["violations"]:
    print("-", violation)


# ---------------------------------------------------------
# TEST 2: Incorrect structural code
# ---------------------------------------------------------

incorrect_code = """
class Notification:
    def send(self):
        pass


class EmailNotification:

    def send(self):
        pass


class SMSNotification(Notification):

    def send(self):
        pass


class NotificationFactory:

    def create_notification(self, notification_type):
        return EmailNotification()
"""


print("\n" + "=" * 60)
print("TEST 2: INCORRECT CODE")
print("=" * 60)

result = evaluator.evaluate(incorrect_code)

print("AST Valid:", result["ast_valid"])
print(
    "Interface Conformance:",
    f"{result['interface_conformance'] * 100:.2f}%"
)
print("Violations:")

for violation in result["violations"]:
    print("-", violation)


# ---------------------------------------------------------
# TEST 3: Invalid Python
# ---------------------------------------------------------

invalid_code = """
class Notification(
    def send(self):
        pass
"""


print("\n" + "=" * 60)
print("TEST 3: SYNTAX ERROR")
print("=" * 60)

result = evaluator.evaluate(invalid_code)

print("AST Valid:", result["ast_valid"])
print(
    "Interface Conformance:",
    f"{result['interface_conformance'] * 100:.2f}%"
)
print("Violations:")

for violation in result["violations"]:
    print("-", violation)