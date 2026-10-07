import os
import unittest

from src.llm_runner import LLMRunner
from src.task_contracts import EXPECTED_TASK_IDS, load_tasks


DATASET = os.path.join(os.path.dirname(__file__), "dataset", "test_tasks.json")


class TestTaskContractsAndPromptVariants(unittest.TestCase):
    def test_all_ten_contracts_load_and_factory_methods_are_static(self):
        tasks = load_tasks(DATASET)
        self.assertEqual(tuple(task["task_id"] for task in tasks), EXPECTED_TASK_IDS)
        for task in tasks:
            self.assertTrue(task["contract"]["classes"])
            self.assertTrue(task["contract"]["methods"])
        factory_methods = {
            method["name"]
            for task in tasks
            if task["pattern"] == "Factory"
            for method in task["contract"]["methods"]
            if method["class"].endswith("Factory")
        }
        self.assertEqual(factory_methods, {"create_notification", "create_document"})
        for task in tasks[:2]:
            factory_method = next(
                method for method in task["contract"]["methods"]
                if method["class"].endswith("Factory")
            )
            self.assertTrue(factory_method["static"])

    def test_baseline_and_caps_share_only_the_partial_skeleton(self):
        task = load_tasks(DATASET)[0]
        runner = LLMRunner(task, model="test-model")
        baseline = runner.build_baseline_prompt()
        caps = runner.build_caps_prompt()
        self.assertIn("PARTIAL IMPLEMENTATION SKELETON", baseline)
        self.assertIn("PARTIAL IMPLEMENTATION SKELETON", caps)
        self.assertNotIn("STRUCTURAL CONTRACT", baseline)
        self.assertIn("STRUCTURAL CONTRACT", caps)
        self.assertEqual(
            baseline.split("Requirements:", 1)[0],
            caps.split("Requirements:", 1)[0],
        )


if __name__ == "__main__":
    unittest.main()
