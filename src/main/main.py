import argparse
import logging
import shutil
import sys
from itertools import product
import yaml
from prettytable import PrettyTable
from torch.optim import Adam

from src.loading.KG import KGUnlearningData
from src.model.Retrain import Retrain
from src.model.SGKU import SGKU
from src.runners.tester import *
from src.runners.trainer import *

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR)

# Get the project root directory (assuming the script is in a subdirectory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))


class Runner():
    def __init__(self, args) -> None:
        """ 1. Set parameters, seeds, logger, paths and device """
        """ Set parameters """
        self.args = args

        """ Important: Set unlearning parameters """
        self.args.begin_pretrain = False
        self.args.begin_unleanring = True
        self.args.unlearning_save_path = os.path.join(PROJECT_ROOT, "checkpoint_unlearning", args.unlearning_method)

        """ Set seeds """
        set_seeds(self.args.seed)
        """ Set logger """
        logger = logging.getLogger()
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
        console_formatter = logging.Formatter('%(asctime)-8s: %(message)s')
        now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        self.args.log_path = f"logs/{now_time}_{self.args.unlearning_method}"  # modified

        logging_file_name = f'{self.args.log_path}.log'  # modified
        file_handler = logging.FileHandler(logging_file_name)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.formatter = console_formatter
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        self.args.logger = logger

        """ Set paths """
        self.args.unlearning_save_path = self.args.unlearning_save_path + self.args.data_name
        if os.path.exists(self.args.unlearning_save_path):
            shutil.rmtree(self.args.unlearning_save_path, True)
        if not os.path.exists(self.args.unlearning_save_path):
            os.mkdir(self.args.unlearning_save_path)

        """ Set device """
        device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
        self.args.device = device

        """ 2. Define data """
        self.kg = KGUnlearningData(self.args)

        """ 3. Define model """
        self.model, self.optimizer = self.create_model()

    def create_model(self):
        if self.args.unlearning_method in ["pretrain", "finetune"]:
            model = Retrain(self.args, self.kg)
        elif self.args.unlearning_method == "SGKU":
            if self.args.kge.lower() == "transe":
                model_class = TransE_module.TransE
            elif self.args.kge.lower() == "rotate":
                model_class = RotatE_module.RotatE
            elif self.args.kge.lower() == "distmult":
                model_class = DistMult_module.DistMult
            elif self.args.kge.lower() == "complexe":
                model_class = ComplexE_module.ComplexE
            else:
                raise ValueError(f"Unsupported KGE model: {self.args.kge_model}")

            model = SGKU(args=self.args, kg=self.kg, kge_model_class=model_class, schema_store=self.kg.schema_store)

        else:
            model = Retrain(self.args, self.kg)
        model.to(self.args.device)
        optimizer = Adam(model.parameters(), lr=self.args.lr)
        return model, optimizer

    def reset_model(self):
        model = Retrain(self.args, self.kg)
        model.to(self.args.device)
        optimizer = Adam(model.parameters(), lr=self.args.lr)
        return model, optimizer

    def unlearning(self):
        report_results = PrettyTable()
        report_results.field_names = ['Timestep', 'Time', 'MRR+', 'Hits@1+', 'Hits@10+', 'MRR-', 'Hits@1-', 'Hits@10-']
        training_times = []
        for ss_id in range(int(self.args.timesteps_num)):
            self.args.timestep = ss_id
            self.args.timestep_test = ss_id
            self.args.timestep_validation = ss_id

            if self.args.unlearning_method == "pretrain":
                self.model, self.optimizer = self.reset_model()
            if (self.args.unlearning_method == "SGKU"):
                self.model.save_embeddings()
            """ training """

            print(f"Starting training for timestep {ss_id}...")

            training_time = self.train()

            """ save and load model """
            best_checkpoint = os.path.join(
                self.args.unlearning_save_path, f'{str(ss_id)}model_best.tar'
            )
            self.load_checkpoint(best_checkpoint)

            """ predict """
            forget_results, retain_results = self.test()

            training_times.append(training_time)

            """ output """
            report_results.add_row([
                self.args.timestep,
                training_time,
                retain_results["mrr"],
                retain_results["hits1"],
                retain_results["hits10"],
                forget_results["mrr"],
                forget_results["hits1"],
                forget_results["hits10"]
            ])
            self.args.logger.info(f"\n{report_results}")
            training_times.append(training_time)
        return report_results, training_times

    def get_report_results(self, results):
        mrrs, hits1s, hits3s, hits10s, num_test = [], [], [], [], []
        for idx, result in enumerate(results):
            mrrs.append(result['mrr'])
            hits1s.append(result['hits1'])
            hits3s.append(result['hits3'])
            hits10s.append(result['hits10'])
            num_test.append(len(self.kg.train_data))
        whole_mrr = sum(
            mrr * num_test[i] for i, mrr in enumerate(mrrs)
        ) / sum(num_test)
        whole_hits1 = sum(
            hits1 * num_test[i] for i, hits1 in enumerate(hits1s)
        ) / sum(num_test)
        whole_hits3 = sum(
            hits3 * num_test[i] for i, hits3 in enumerate(hits3s)
        ) / sum(num_test)
        whole_hits10 = sum(
            hits10 * num_test[i] for i, hits10 in enumerate(hits10s)
        ) / sum(num_test)
        return round(whole_mrr, 3), round(whole_hits1, 3), round(whole_hits3, 3), round(whole_hits10, 3)

    def train(self):
        start_time = time.time()
        self.best_valid = 0.0
        self.stop_epoch = 0


        trainer = UnlearningTrainer(self.args, self.kg, self.model, self.optimizer)

        print("\n========================================")
        print(f"STARTING TRAINING: {self.args.epoch_num} epochs")
        print(f"Model: {self.args.unlearning_method}")
        print(f"Device: {self.args.device}")
        print(f"Batch Size: {self.args.batch_size}")
        print("========================================")

        for epoch in range(int(self.args.epoch_num)):
            epoch_start_time = time.time()
            print(f"=== EPOCH {epoch + 1}/{int(self.args.epoch_num)} STARTED ===")
            if self.args.debug and epoch > 0:
                print("DEBUG MODE: Stopping after first epoch")
                break

            self.args.epoch = epoch
            if self.args.unlearning_method in ["pretrain", "finetune", "SGKU"]:
                valid_res = dict()
                # Print message before training
                #print(f"Training epoch {epoch + 1}...")
                to = time.time()
                # Run the epoch
                loss, forget_results, retain_results = trainer.run_epoch()
                elapsed = time.time() - to
                print(f"Epoch ({epoch}/{int(self.args.epoch_num)}) completed in {elapsed:.2f} secs")

                # Validation phase
                if self.args.epoch % self.args.valid_gap == 0:
                    print(f"\n--- VALIDATION PHASE FOR EPOCH {epoch + 1} ---")
                    print(f"Forget results: {forget_results}")
                    print(f"Retain results: {retain_results}")
                    # Calculate combined metrics
                    valid_res[self.args.valid_metrics] = 0.5 * (1 - forget_results[self.args.valid_metrics]) + 0.5 * \
                                                         retain_results[self.args.valid_metrics]
                    valid_res['hits3'] = 0.5 * (1 - forget_results['hits3']) + 0.5 * retain_results['hits3']
                    valid_res['hits10'] = 0.5 * (1 - forget_results['hits10']) + 0.5 * retain_results['hits10']
                    print(f"Combined validation results: {valid_res}")
                    print(f"--- VALIDATION PHASE COMPLETE ---\n")
            else:
                raise NotImplementedError

            if self.args.epoch % self.args.valid_gap != 0:
                print(f"Skipping validation (epoch {epoch + 1} % {self.args.valid_gap} != 0)")
                continue

            # Model saving logic
            if valid_res[self.args.valid_metrics] > self.best_valid:
                prev_best = self.best_valid
                self.best_valid = valid_res[self.args.valid_metrics]
                self.stop_epoch = 0
                print(f"NEW BEST MODEL! Metrics improved from {prev_best:.4f} to {self.best_valid:.4f}")
                self.save_model(is_best=True)
            else:
                self.stop_epoch += 1
                print(f"No improvement. Current: {valid_res[self.args.valid_metrics]:.4f}, Best: {self.best_valid:.4f}")
                self.save_model()
                if self.stop_epoch >= self.args.patience:
                    print(f"\n*** EARLY STOPPING! Epoch: {epoch + 1}, No improvement for {self.stop_epoch} epochs ***")
                    self.args.logger.info(
                        f'Early Stopping! Epoch: {epoch + 1} Best Results: {round(self.best_valid * 100, 3)}'
                    )
                    break

            # Log results
            if epoch % 1 == 0:
                log_message = (
                    f"Epoch:{epoch + 1}\tLoss:{round(loss, 3)}\tMRR-Avg:{round(valid_res['mrr'] * 100, 3)}\t"
                    f"Hits@10:{round(valid_res['hits10'] * 100, 3)}\tBest:{round(self.best_valid * 100, 3)}"
                )
                print(log_message)
                self.args.logger.info(log_message)

            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            print(f"--- EPOCH {epoch} COMPLETED in {epoch_time:.2f} seconds ---\n")

        total_time = time.time() - start_time
        print("\n========================================")
        print(f"TRAINING COMPLETED: {epoch} epochs")
        print(f"Best validation score: {round(self.best_valid * 100, 3)}")
        print(f"Total training time: {total_time:.2f} seconds")
        print("========================================\n")
        return total_time

    def test(self):
        print("========================= Start testing =========================")
        tester = UnlearningTester(self.args, self.kg, self.model)
        return tester.test_with_report()

    def save_model(self, is_best=False):
        checkpoint_dict = {'state_dict': self.model.state_dict()}
        checkpoint_dict['epoch_id'] = self.args.epoch
        out_tar = os.path.join(
            self.args.unlearning_save_path,
            f'{str(self.args.timestep)}checkpoint-{self.args.epoch}.tar',
        )
        torch.save(checkpoint_dict, out_tar)
        if is_best:
            best_path = os.path.join(
                self.args.unlearning_save_path, f'{str(self.args.timestep)}model_best.tar'
            )
            shutil.copyfile(out_tar, best_path)

    def load_checkpoint(self, input_file):
        if os.path.isfile(input_file):
            logging.info(f"=> loading checkpoint \'{input_file}\'")
            checkpoint = torch.load(input_file, map_location=f"mps:0", weights_only=True)
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            logging.info(f'=> no checking found at \'{input_file}\'')


def run_experiments(config_file="hyperparameters.yaml"):

    config_file = os.path.join(BASE_DIR, config_file)

    """
    Reads hyperparameters from a YAML file and runs experiments.

    Args:
        config_file (str): Path to the YAML configuration file.
    """
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        return

    args = argparse.Namespace()

    # Load default settings from YAML and update args
    for key, value in config["defaults"].items():
        setattr(args, key, value)

    for dataset_config in config["datasets"]:
        dataset = dataset_config["name"]
        args.data_name = dataset

        # Save checkpoint in project root
        args.unlearning_save_path = os.path.join(PROJECT_ROOT, "checkpoint_unlearning", args.unlearning_method, dataset)

        for experiment in dataset_config["experiments"]:
            experiment_name = experiment["name"]
            print(f"\n--- *************** Running Experiment: {experiment_name} ***************** ---")
            # Collect all hyperparameter values for this experiment
            # Use empty list as default if parameter is not specified
            param_values = {}
            for param_name, param_value in experiment.items():
                if param_name != "name":  # Skip the experiment name
                    param_values[param_name] = param_value

            # Get all parameter combinations using product
            param_names = list(param_values.keys())
            param_combinations = list(product(*[param_values[name] for name in param_names]))

            for combination in param_combinations:
                # Set the parameters for this run
                for i, param_name in enumerate(param_names):
                    setattr(args, param_name, combination[i])
                # Print current parameter configuration
                param_str = ", ".join([f"{name}={getattr(args, name)}" for name in param_names])
                print(f"------ ***************  Running with {param_str} ------ *************** ")
                # Run the experiment with these parameters
                ins = Runner(args)
                ins.unlearning()
                print("------ ***************  Finished parameter combination........ *****************-------")
                print()


run_experiments()
