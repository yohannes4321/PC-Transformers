def per_layer_T_objective(trial, device=None, flash=False, enable_batch_logging=False):

import optuna
import argparse
import os
import sys
from tuning.trial_objective import objective
from utils.device_utils import setup_device
from utils.model_utils import set_seed
from data_preparation.config import vocab_size

def per_layer_T_objective(trial, device=None, flash=False, enable_batch_logging=False):
    # Only tune per-layer T values, use fixed config for all else
    embed_T = trial.suggest_int('embed_T', 1, 10)
    attn_T = trial.suggest_int('attn_T', 1, 10)
    linear_attn_T = trial.suggest_int('linear_attn_T', 1, 10)
    fc1_T = trial.suggest_int('fc1_T', 1, 10)
    fc2_T = trial.suggest_int('fc2_T', 1, 10)
    linear_output_T = trial.suggest_int('linear_output_T', 1, 10)

    # Fixed config values (set as needed for your model)
    from predictive_coding.config import GPTConfig
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=128,
        peak_learning_rate=1e-3,
        warmup_steps=100,
        n_embed=128,
        dropout=0.1,
        lr=1e-4,
        T=1,  # Not used, just for compatibility
        num_heads=4,
        n_blocks=2,
        batch_size=8,
        num_epochs=2,
        update_bias=True,
        internal_energy_fn_name="pc_e",
        output_energy_fn_name="pc_e",
        combined_internal_weight=0.5,
        combined_output_weight=0.5,
        use_flash_attention=flash,
        alpha=0.5
    )
    config.embed_T = embed_T
    config.attn_T = attn_T
    config.linear_attn_T = linear_attn_T
    config.fc1_T = fc1_T
    config.fc2_T = fc2_T
    config.linear_output_T = linear_output_T

    def wrapped_objective(trial_, device_=device, flash_=flash, enable_batch_logging_=enable_batch_logging):
        trial_.set_user_attr('embed_T', embed_T)
        trial_.set_user_attr('attn_T', attn_T)
        trial_.set_user_attr('linear_attn_T', linear_attn_T)
        trial_.set_user_attr('fc1_T', fc1_T)
        trial_.set_user_attr('fc2_T', fc2_T)
        trial_.set_user_attr('linear_output_T', linear_output_T)
        trial_.set_user_attr('config', config.__dict__)
        # Patch config into trial for downstream use
        # Call the original objective, but force config
        import types
        old_get_dynamic_model_config = None
        try:
            import tuning.config as config_mod
            old_get_dynamic_model_config = getattr(config_mod, 'get_dynamic_model_config', None)
            config_mod.get_dynamic_model_config = lambda *_a, **_k: config
            result = objective(trial_, device_, flash_, enable_batch_logging_)
            # Print EFE, CE, and PPL for this trial
            efe = trial_.user_attrs.get('energy', None)
            ce = trial_.user_attrs.get('ce_loss', None)
            ppl = trial_.user_attrs.get('perplexity', None)
            print(f"[Trial {trial_.number}] embed_T={embed_T}, attn_T={attn_T}, linear_attn_T={linear_attn_T}, fc1_T={fc1_T}, fc2_T={fc2_T}, linear_output_T={linear_output_T} | EFE={efe} | CE={ce} | PPL={ppl}")
            return result
        finally:
            if old_get_dynamic_model_config:
                config_mod.get_dynamic_model_config = old_get_dynamic_model_config

    return wrapped_objective(trial, device, flash, enable_batch_logging)


def main():
    parser = argparse.ArgumentParser(description="Optuna per-layer T hyperparameter tuning")
    parser.add_argument('--n_trials', type=int, default=30)
    parser.add_argument('--study_name', type=str, default="per_layer_T_tuning")
    parser.add_argument('--flash', action='store_true')
    parser.add_argument('--log_batches', action='store_true')
    args = parser.parse_args()

    set_seed(42)
    local_rank, device, use_ddp = setup_device()

    storage_url = f"sqlite:///tuning/{args.study_name}.db"
    if local_rank == 0:
        try:
            _ = optuna.create_study(
                direction='minimize',
                study_name=args.study_name,
                storage=storage_url,
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=5,
                    n_warmup_steps=3,
                    interval_steps=1
                )
            )
        except Exception as e:
            print(f"Study creation skipped: {e}")
    import torch.distributed as dist
    if use_ddp and not dist.is_initialized():
        import torch
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    if use_ddp:
        dist.barrier()
    study = optuna.load_study(
        study_name=args.study_name,
        storage=storage_url
    )
    def callback(study, trial):
        if local_rank == 0:
            print(f"Best trial so far: {study.best_trial.number} | Value: {study.best_trial.value}")
    study.optimize(lambda trial: per_layer_T_objective(trial, device, args.flash, enable_batch_logging=args.log_batches),
                   n_trials=args.n_trials, callbacks=[callback], show_progress_bar=(local_rank == 0))
    if use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
