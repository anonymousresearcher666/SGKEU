from src.model.model_training import *


class Trainer():
    def __init__(self, args, kg, model, optimizer) -> None:
        self.args = args
        self.kg = kg
        self.model = model
        self.logger = args.logger
        self.train_processor = UnlTrainBatch(args, kg)
        self.valid_processor = DBatching(args, kg)
        self.optimizer = optimizer


    def run_epoch(self):
        self.args.valid = True
        loss = self.train_processor.process_epoch(self.model, self.optimizer)
        res = self.valid_processor.process_epoch(self.model)
        self.args.valid = False
        return loss, res


class UnlearningTrainer():
    def __init__(self, args, kg, model, optimizer) -> None:
        self.args = args
        self.kg = kg
        self.model = model
        self.optimizer = optimizer
        self.logger = args.logger
        if self.args.unlearning_method in ["pretrain", "finetune"]:
            self.train_processor = UnlTrainBatch(args, kg)
        elif self.args.unlearning_method == "SGKU":


            self.train_processor = SGKUBatching(args, kg)
        else:
            raise NotImplementedError
        #
        ##Test how well the model has **forgotten** knowledge
        self.run_forget_valid_processor = ForgetDBatching(args, kg)
        ##Test how well the model has **preserved** knowledge
        self.retain_valid_processor = RetainDBatching(args, kg)

        # print("forget_valid_processor ", len(self.run_forget_valid_processor.data_loader)) ###
        # print("reserve_valid_processor ", len(self.reserve_valid_processor.data_loader)) ##

    def evaluate_model(self, model, forget_processor, reserve_processor):
        """Evaluate model on both forget and reserve datasets."""
        # Run evaluations separately
        forget_results = forget_processor.process_epoch(model)
        retain_results = reserve_processor.process_epoch(model)

        # Calculate unlearning metrics
        unlearning_metrics = calculate_unlearning_metrics(
            forget_processor=forget_processor,
            reserve_processor=reserve_processor
        )

        # You can save or return the metrics as needed
        return {
            'forget_results': forget_results,
            'retain_results': retain_results,
            'unlearning_metrics': unlearning_metrics
        }

    def run_epoch(self):
        self.args.valid = True
        loss = self.train_processor.process_epoch(self.model, self.optimizer)
        if self.args.epoch % self.args.valid_gap == 0:
            results = self.evaluate_model(self.model, self.run_forget_valid_processor, self.retain_valid_processor)
            self.args.valid = False
            return loss, results['forget_results'], results['retain_results']
        else:
            return loss, None, None
