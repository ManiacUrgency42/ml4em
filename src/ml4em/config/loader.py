"""
Configuration loading and secret injection.

Rules
-----
1. Config files hold structure, not secrets.
   API tokens are never written to config.yaml.  They live in environment
   variables so they cannot be accidentally committed.

2. .env files are supported for local development.
   Place a .env file in the project root (gitignored):

       ML4EM_ZTF_TOKEN=your_kowalski_token
       ML4EM_RUBIN_TOKEN=your_rubin_rsp_token

   load_config() loads this file automatically before reading config.yaml.

3. Config is validated at load time.
   Pydantic raises immediately on type errors or invalid values — not
   silently mid-pipeline.

4. PipelineConfig() works with no file.
   load_default_config() returns a fully valid config from schema defaults,
   useful in tests and notebooks.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import yaml

from ml4em.config.schema import PipelineConfig

try:
    from dotenv import load_dotenv as _load_dotenv
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False


def _try_load_dotenv(dotenv_path: Path) -> None:
    if _DOTENV_AVAILABLE and dotenv_path.exists():
        _load_dotenv(dotenv_path, override=False)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(
    path: Union[str, Path] = "config.yaml",
    *,
    dotenv_path: Union[str, Path, None] = None,
) -> PipelineConfig:
    """Load and validate a ml4em config file.

    Reads the YAML file at ``path``, merges with schema defaults, and
    validates every field.  Raises ``FileNotFoundError`` if the file is
    absent, and ``pydantic.ValidationError`` if any field is invalid.

    A .env file is loaded before the YAML so that environment variables
    are available if the loader needs them.

    Parameters
    ----------
    path:
        Path to config.yaml.  Defaults to ``config.yaml`` in the current
        working directory.
    dotenv_path:
        Path to a .env file.  Defaults to ``.env`` in the same directory
        as the config file.

    Returns
    -------
    PipelineConfig
        Fully validated configuration object.

    Examples
    --------
    >>> cfg = load_config("config.yaml")
    >>> cfg.sources.ztf.collection_sources
    'ZTF_sources_20240515'
    >>> cfg.features.period.algorithms
    ['CE', 'AOV', 'LS', 'BLS']
    >>> cfg.storage.features_dir
    'features'
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path.resolve()}\n"
            "Copy config.example.yaml to config.yaml and edit as needed."
        )

    env_path = Path(dotenv_path) if dotenv_path else config_path.parent / ".env"
    _try_load_dotenv(env_path)

    with config_path.open() as fh:
        raw = yaml.safe_load(fh) or {}

    return PipelineConfig(**raw)


def load_default_config() -> PipelineConfig:
    """Return a PipelineConfig built entirely from schema defaults.

    No file I/O.  Useful in unit tests and exploratory notebooks.
    """
    return PipelineConfig()


# ---------------------------------------------------------------------------
# Secret accessors
# ---------------------------------------------------------------------------

def get_ztf_token() -> str:
    """Return the Kowalski API token for ZTF access.

    Reads ``ML4EM_ZTF_TOKEN`` from the environment.  Set this in your shell
    or in a .env file — never in config.yaml.

    Raises
    ------
    EnvironmentError
        If ``ML4EM_ZTF_TOKEN`` is not set or is empty.
    """
    token = os.environ.get("ML4EM_ZTF_TOKEN", "").strip()
    if not token:
        raise EnvironmentError(
            "ML4EM_ZTF_TOKEN is not set.\n"
            "Add it to your .env file:\n"
            "    ML4EM_ZTF_TOKEN=your_kowalski_token"
        )
    return token


def get_rubin_token() -> str:
    """Return the RSP API token for Rubin DP1 TAP access.

    Reads ``ML4EM_RUBIN_TOKEN`` from the environment.

    Raises
    ------
    EnvironmentError
        If ``ML4EM_RUBIN_TOKEN`` is not set or is empty.
    """
    token = os.environ.get("ML4EM_RUBIN_TOKEN", "").strip()
    if not token:
        raise EnvironmentError(
            "ML4EM_RUBIN_TOKEN is not set.\n"
            "Add it to your .env file:\n"
            "    ML4EM_RUBIN_TOKEN=your_rubin_rsp_token"
        )
    return token
