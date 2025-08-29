import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple
from visualization.config import Config
from visualization.exceptions import VisualizationError
from visualization.models import Model
from visualization.utils import load_data, plot_convergence, plot_flow_theory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Visualization:
    def __init__(self, config: Config):
        self.config = config
        self.model = Model(config)

    def visualize_convergence(self, data: pd.DataFrame) -> None:
        """
        Visualize the convergence of the optimization algorithm.

        Args:
            data (pd.DataFrame): Data containing the convergence information.
        """
        try:
            plot_convergence(data, self.config.plot_settings)
        except Exception as e:
            logger.error(f"Failed to visualize convergence: {e}")
            raise VisualizationError("Failed to visualize convergence")

    def visualize_flow_theory(self, data: pd.DataFrame) -> None:
        """
        Visualize the flow theory of the optimization algorithm.

        Args:
            data (pd.DataFrame): Data containing the flow theory information.
        """
        try:
            plot_flow_theory(data, self.config.plot_settings)
        except Exception as e:
            logger.error(f"Failed to visualize flow theory: {e}")
            raise VisualizationError("Failed to visualize flow theory")

    def visualize_results(self, data: pd.DataFrame) -> None:
        """
        Visualize the results of the optimization algorithm.

        Args:
            data (pd.DataFrame): Data containing the results information.
        """
        try:
            self.visualize_convergence(data)
            self.visualize_flow_theory(data)
        except Exception as e:
            logger.error(f"Failed to visualize results: {e}")
            raise VisualizationError("Failed to visualize results")

class Model:
    def __init__(self, config: Config):
        self.config = config

    def load_data(self) -> pd.DataFrame:
        """
        Load the data for the optimization algorithm.

        Returns:
            pd.DataFrame: Data containing the optimization information.
        """
        try:
            return load_data(self.config.data_settings)
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise VisualizationError("Failed to load data")

class Config:
    def __init__(self):
        self.plot_settings = {
            "title": "Optimization Convergence",
            "xlabel": "Iteration",
            "ylabel": "Value",
            "legend": True,
        }
        self.data_settings = {
            "file_path": "data.csv",
            "columns": ["iteration", "value"],
        }

class VisualizationError(Exception):
    pass

def load_data(settings: Dict) -> pd.DataFrame:
    """
    Load the data for the optimization algorithm.

    Args:
        settings (Dict): Settings for loading the data.

    Returns:
        pd.DataFrame: Data containing the optimization information.
    """
    try:
        file_path = settings["file_path"]
        columns = settings["columns"]
        data = pd.read_csv(file_path, usecols=columns)
        return data
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise VisualizationError("Failed to load data")

def plot_convergence(data: pd.DataFrame, settings: Dict) -> None:
    """
    Plot the convergence of the optimization algorithm.

    Args:
        data (pd.DataFrame): Data containing the convergence information.
        settings (Dict): Settings for plotting the convergence.
    """
    try:
        plt.figure(figsize=(8, 6))
        plt.plot(data["iteration"], data["value"])
        plt.title(settings["title"])
        plt.xlabel(settings["xlabel"])
        plt.ylabel(settings["ylabel"])
        if settings["legend"]:
            plt.legend()
        plt.show()
    except Exception as e:
        logger.error(f"Failed to plot convergence: {e}")
        raise VisualizationError("Failed to plot convergence")

def plot_flow_theory(data: pd.DataFrame, settings: Dict) -> None:
    """
    Plot the flow theory of the optimization algorithm.

    Args:
        data (pd.DataFrame): Data containing the flow theory information.
        settings (Dict): Settings for plotting the flow theory.
    """
    try:
        plt.figure(figsize=(8, 6))
        plt.plot(data["iteration"], data["value"])
        plt.title(settings["title"])
        plt.xlabel(settings["xlabel"])
        plt.ylabel(settings["ylabel"])
        if settings["legend"]:
            plt.legend()
        plt.show()
    except Exception as e:
        logger.error(f"Failed to plot flow theory: {e}")
        raise VisualizationError("Failed to plot flow theory")