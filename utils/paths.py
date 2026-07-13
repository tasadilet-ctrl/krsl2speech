"""
Per-machine path overrides via environment variables.

Problem: one config.yaml is rsynced to several HPC boxes, but each box
stores the datasets in a different place (new box: /data/shared/asan-dataset,
old box: /raid/shared/dataset/...). Hardcoding one path in the synced config
breaks the other box on every sync.

Solution: keep box-agnostic defaults in config.yaml, and let each box export
env vars once (e.g. in ~/.bashrc or the tmux session). These override the
config after it's loaded, so syncing never clobbers a box's local paths.

Recognized env vars:
    ASAN_ROOT          -> paths.asan.root
    ASAN_PROSODY_ROOT  -> paths.asan.prosody_root
    KRSL_OUTPUT        -> paths.output

Usage (right after yaml.safe_load):
    from utils.paths import apply_env_overrides
    cfg = apply_env_overrides(cfg)
"""
import os


def apply_env_overrides(cfg):
    """Mutate and return cfg, applying any *_ROOT env overrides that are set."""
    paths = cfg.get('paths', {}) if isinstance(cfg, dict) else {}

    asan = paths.get('asan')
    if isinstance(asan, dict):
        if os.environ.get('ASAN_ROOT'):
            asan['root'] = os.environ['ASAN_ROOT']
        if os.environ.get('ASAN_PROSODY_ROOT'):
            asan['prosody_root'] = os.environ['ASAN_PROSODY_ROOT']

    if os.environ.get('KRSL_OUTPUT'):
        paths['output'] = os.environ['KRSL_OUTPUT']

    # Announce overrides once so runs are self-documenting in the logs.
    active = {k: os.environ[k] for k in
              ('ASAN_ROOT', 'ASAN_PROSODY_ROOT', 'KRSL_OUTPUT')
              if os.environ.get(k)}
    if active:
        print(f"[paths] env overrides applied: {active}")

    return cfg
