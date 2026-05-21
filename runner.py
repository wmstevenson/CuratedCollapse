import numpy as np
import pandas as pd
import json
from pathlib import Path

from synthcity.plugins import Plugins
from config import Config
from metrics import alpha_beta_metrics, audit_alpha_precision, train_synth_test_real


def generate_synth_df(
    plugin,
    count: int,
):
    synth_df = plugin.generate(
        count=count,
    ).dataframe().reset_index(drop=True)

    return synth_df


def generate_audited_synth_df(
    plugin,
    real_df: pd.DataFrame,
    target_n: int,
    audit_alpha: float,
):
    accepted_parts = []
    accepted_count = 0
    total_generated = 0

    while accepted_count < target_n:
        remaining = target_n - accepted_count

        current_synth_df = generate_synth_df(
            plugin=plugin,
            count=remaining,
        )

        precision_audit_mask = audit_alpha_precision(
            real_df,
            current_synth_df,
            audit_alpha
        )


        accepted_batch = current_synth_df.loc[precision_audit_mask].reset_index(drop=True)

        total_generated += len(current_synth_df)
        accepted_count += len(accepted_batch)

        if not accepted_batch.empty:
            accepted_parts.append(accepted_batch)

    audited_synth_df = pd.concat(
        accepted_parts,
        ignore_index=True,
    ).reset_index(drop=True)

    audit_info = {
        "retention_rate": target_n / total_generated,
        "generated_samples": total_generated,
    }

    return audited_synth_df, audit_info





def run_experiment(cfg: Config):
    real_train_df = pd.read_csv(cfg.train_dataset_path)
    real_test_df = pd.read_csv(cfg.test_dataset_path)
    target_n = len(real_train_df)

    previous_synth = None
    results = []

    results_file_path = (
        Path(cfg.results_path)
        / cfg.plugin_name
        / f"audit_{cfg.audit_alpha}"
        / f"seed_{cfg.seed}.json"
    )
    results_file_path.parent.mkdir(parents=True, exist_ok=True)

    for gen in range(cfg.generations):
        train_df = real_train_df if gen == 0 else previous_synth

        plugin = Plugins().get(cfg.plugin_name, random_state=cfg.seed)
        plugin.fit(train_df)

        if cfg.audit_alpha is not None:
            previous_synth, audit_info = generate_audited_synth_df(
                plugin=plugin,
                real_df=real_train_df,
                target_n=target_n,
                audit_alpha=cfg.audit_alpha,
            )
        else:
            previous_synth = generate_synth_df(
                plugin=plugin,
                count=target_n,
            )

            audit_info = {
                "retention_rate": 1.0,
                "generated_samples": target_n,
            }

            
        fidelity_metrics = alpha_beta_metrics(real_train_df, previous_synth)

        utility_metrics = train_synth_test_real(
            train_df=previous_synth,
            test_df=real_test_df,
            target_col=cfg.target_name,
        )

        row = {
            "generation": gen,
            "seed": cfg.seed,
            **audit_info,
            **fidelity_metrics,
            **utility_metrics,
        }

        results.append(row)



        if cfg.seed == 0:
            output_dir = (
                Path(cfg.output_path)
                / cfg.plugin_name
                / f"audit_{cfg.audit_alpha}"
                / f"seed_{cfg.seed}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file_path = output_dir / f"gen_{gen}.csv"
            previous_synth.to_csv(output_file_path, index=False)

        print(
            f"Generation {gen}:\n"
            f"    alpha_prec={row['delta_precision_alpha']}\n"
            f"    beta_cov={row['delta_coverage_beta']}\n"
            f"    tstr_acc={row['tstr_accuracy']}\n"
            f"    tstr_bal_acc={row['tstr_balanced_accuracy']}\n"
            f"    tstr_macro_f1={row['tstr_macro_f1']}\n"
            f"    retention={row['retention_rate']}\n"
            f"    generated={row['generated_samples']}"
        )

    with open(results_file_path, "w") as f:
        json.dump(
            {"config": cfg.__dict__, "results": results},
            f,
            indent=2,
        )