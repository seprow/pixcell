from __future__ import annotations
import yaml
import json
import pickle
import torch
from typing import Any, TypeVar
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from loguru import logger


T = TypeVar("T")
class PickleIO:
    """Utility for saving and loading Python objects with pickle."""

    @staticmethod
    def save(
        obj: T,
        path: Path,
    ) -> None:

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as f:
            pickle.dump(
                obj,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @staticmethod
    def load(
        path: Path,
    ) -> T:

        with path.open("rb") as f:
            return pickle.load(f)
        
class NumpyIO:

    @staticmethod
    def save(image: sitk.Image, path: Path):

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            path,
            sitk.GetArrayFromImage(image),
        )

    @staticmethod
    def load(path: Path):

        return np.load(path)

class NiftiIO:

    @staticmethod
    def save(image: sitk.Image, path: Path):

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        sitk.WriteImage(
            image,
            str(path),
        )
        

def load_yaml(path):
    with open(path) as f:
        loaded_yaml = yaml.safe_load(f)

    return loaded_yaml


def read_json_file(
    file_path: str | Path,
) -> dict[str, Any]:
    """
    Read and parse a JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Parsed JSON content.
    """

    file_path = Path(file_path)

    try:
        with file_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    except FileNotFoundError:
        logger.exception("JSON file not found: %s", file_path)
        raise

    except json.JSONDecodeError:
        logger.exception("Invalid JSON format: %s", file_path)
        raise

    except Exception:
        logger.exception("Failed to read JSON file: %s", file_path)
        raise


def load_checkpoint(path: str, device="cpu"):
    """
    Safe torch.load wrapper.
    """
    logger.info(f"Loading checkpoint: {path}")
    return torch.load(path, device=device)
