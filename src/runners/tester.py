from src.model.model_training import *


class Tester():
    def __init__(self, args, kg, model) -> None:
        self.args = args
        self.kg = kg
        self.model = model
        self.test_processor = DBatching(args, kg)

    def test(self):
        self.args.valid = False
        return self.test_processor.process_epoch(self.model)


class UnlearningTester():
    def __init__(self, args, kg, model) -> None:
        self.args = args
        self.kg = kg
        self.model = model
        self.forget_test_processor = ForgetDBatching(args, kg)
        self.retain_test_processor = RetainDBatching(args, kg)

    def test(self):
        self.args.valid = False
        forget_results = self.forget_test_processor.process_epoch(self.model)
        retain_results = self.retain_test_processor.process_epoch(self.model)
        return forget_results, retain_results

    def test_with_report(self):
        forget_results, retain_results = self.test()

        print("\n===== UNLEARNING TEST REPORT =====")
        print(f"FORGOTTEN TRIPLES METRICS:")
        print(f"  MRR: {forget_results['mrr']:.4f} (lower is better)")
        print(f"  Hits@1: {forget_results['hits1']:.4f}")
        print(f"  Hits@10: {forget_results['hits10']:.4f}")

        print(f"\nRETAINED TRIPLES METRICS:")
        print(f"  MRR: {retain_results['mrr']:.4f} (higher is better)")
        print(f"  Hits@1: {retain_results['hits1']:.4f}")
        print(f"  Hits@10: {retain_results['hits10']:.4f}")

        # Calculate combined score (example formula)
        forget_quality = 1 - forget_results['mrr']  # Lower MRR is better for forgetting
        retain_quality = retain_results['mrr']  # Higher MRR is better for retaining
        combined_score = (forget_quality * retain_quality)

        print(f"\nCOMBINED METRICS:")
        print(f"  Forget Quality: {forget_quality:.4f}")
        print(f"  Retain Quality: {retain_quality:.4f}")
        print(f"  Combined Score: {combined_score:.4f}")
        print("================================\n")

        return forget_results, retain_results
