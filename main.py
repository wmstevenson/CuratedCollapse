from config import Config
from runner import run_experiment


def main():
    audit_levels = [0.4, 0.6, 0.8, None]

    for audit_alpha in audit_levels:
        print(f"audit_alpha = {audit_alpha}")

        for seed in range(10):
            print(f"  seed = {seed}")

            cfg = Config(
                train_dataset_path="diabetes/real/train.csv",
                test_dataset_path="diabetes/real/test.csv",
                target_name = "Outcome",
                audit_alpha=audit_alpha,
                results_path="diabetes/results",
                output_path = "diabetes/synth",
                plugin_name="ctgan",
                generations=10,
                seed=seed,
            )

            run_experiment(cfg)


if __name__ == "__main__":
    main()
