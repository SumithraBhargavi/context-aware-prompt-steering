"""
test_repair_engine.py — Unit tests for RepairEngine (CAPS Phase 2).
"""

import os
import shutil
import tempfile
import unittest

from src.repair_engine import RepairEngine


class TestRepairEngine(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="caps_test_repair_")
        self.engine = RepairEngine(base_dir=self.tmpdir)
        self.task_id = "factory_01"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_extract_violations(self):
        struct_res = {
            "violations": ["Missing class: NotificationFactory", "NotificationFactory.create_notification should be concrete"]
        }
        extracted = RepairEngine.extract_violations(struct_res)
        self.assertEqual(len(extracted), 2)
        self.assertIn("Missing class: NotificationFactory", extracted)

    def test_construct_repair_prompt(self):
        code = "class NotificationFactory:\n    pass"
        violations = ["Missing class: EmailNotification"]
        prompt = self.engine.construct_repair_prompt(self.task_id, code, violations)
        self.assertIn("Structural Contract", prompt)
        self.assertIn("Missing class: EmailNotification", prompt)
        self.assertIn("Original Code to Repair", prompt)

    def test_evaluate_code(self):
        code = """from abc import ABC, abstractmethod

class Notification(ABC):
    @abstractmethod
    def send(self) -> None:
        pass

class EmailNotification(Notification):
    def send(self) -> None:
        pass

class SMSNotification(Notification):
    def send(self) -> None:
        pass

class NotificationFactory:
    @staticmethod
    def create_notification(notification_type: str) -> Notification:
        if notification_type == "email":
            return EmailNotification()
        return SMSNotification()
"""
        res = self.engine.evaluate_code(self.task_id, code, run_functional=True)
        self.assertTrue(res["structural"]["ast_valid"])
        self.assertEqual(res["structural"]["interface_conformance"], 1.0)
        self.assertEqual(res["structural"]["violation_count"], 0)
        self.assertTrue(res["functional"]["functional_valid"])

    def test_run_repair_loop_already_valid(self):
        valid_code = """from abc import ABC, abstractmethod

class Notification(ABC):
    @abstractmethod
    def send(self) -> None:
        pass

class EmailNotification(Notification):
    def send(self) -> None:
        pass

class SMSNotification(Notification):
    def send(self) -> None:
        pass

class NotificationFactory:
    @staticmethod
    def create_notification(notification_type: str) -> Notification:
        if notification_type == "email":
            return EmailNotification()
        return SMSNotification()
"""
        res = self.engine.run_repair_loop(
            task_id=self.task_id,
            model_id="test_model",
            source_condition="baseline",
            source_run=1,
            original_code=valid_code,
            repair_llm_fn=lambda p: valid_code,
            max_iterations=3,
            run_functional=True,
            save_results=True,
        )

        self.assertTrue(res["repair_success"])
        self.assertEqual(res["iterations_completed"], 0)
        self.assertEqual(res["post_repair"]["interface_conformance"], 1.0)

    def test_run_repair_loop_successful_repair(self):
        bad_code = """class NotificationFactory:
    def create_notification(self, notification_type):
        pass
"""
        good_code = """from abc import ABC, abstractmethod

class Notification(ABC):
    @abstractmethod
    def send(self) -> None:
        pass

class EmailNotification(Notification):
    def send(self) -> None:
        pass

class SMSNotification(Notification):
    def send(self) -> None:
        pass

class NotificationFactory:
    @staticmethod
    def create_notification(notification_type: str) -> Notification:
        if notification_type == "email":
            return EmailNotification()
        return SMSNotification()
"""
        def mock_llm(prompt):
            return good_code

        res = self.engine.run_repair_loop(
            task_id=self.task_id,
            model_id="test_model",
            source_condition="baseline",
            source_run=1,
            original_code=bad_code,
            repair_llm_fn=mock_llm,
            max_iterations=3,
            run_functional=True,
            save_results=True,
        )

        self.assertTrue(res["repair_success"])
        self.assertGreater(res["post_repair"]["interface_conformance"], res["pre_repair"]["interface_conformance"])
        self.assertEqual(res["post_repair"]["violation_count"], 0)

    def test_safety_check_rejects_degraded_repair(self):
        bad_code = "class NotificationFactory:\n    pass"
        worse_code = "INVALID PYTHON SYNTAX ::::"

        res = self.engine.run_repair_loop(
            task_id=self.task_id,
            model_id="test_model",
            source_condition="baseline",
            source_run=1,
            original_code=bad_code,
            repair_llm_fn=lambda p: worse_code,
            max_iterations=2,
            run_functional=False,
            save_results=False,
        )

        self.assertFalse(res["repair_success"])
        self.assertEqual(res["final_code"], bad_code)


if __name__ == "__main__":
    unittest.main()
