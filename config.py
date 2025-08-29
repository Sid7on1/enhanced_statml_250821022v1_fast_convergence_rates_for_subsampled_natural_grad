import logging
import os
import yaml
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
from contextlib import contextmanager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
CONFIG_FILE = 'config.yaml'
DEFAULT_CONFIG = {
    'optimization': {
        'algorithm': 'SNGD',
        'learning_rate': 0.01,
        'batch_size': 32,
        'num_iterations': 1000
    },
    'flow_theory': {
        'velocity_threshold': 0.1,
        'flow_rate': 0.5
    }
}

# Define enums
class Algorithm(str, Enum):
    SNGD = 'SNGD'
    SPRING = 'SPRING'

class OptimizationMode(str, Enum):
    TRAIN = 'train'
    VALIDATE = 'validate'

# Define dataclasses
@dataclass
class OptimizationConfig:
    algorithm: Algorithm
    learning_rate: float
    batch_size: int
    num_iterations: int

@dataclass
class FlowTheoryConfig:
    velocity_threshold: float
    flow_rate: float

@dataclass
class Config:
    optimization: OptimizationConfig
    flow_theory: FlowTheoryConfig

# Define functions
def load_config(file_path: str = CONFIG_FILE) -> Config:
    """Load configuration from file."""
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
            return Config(
                optimization=OptimizationConfig(
                    algorithm=Algorithm(config['optimization']['algorithm']),
                    learning_rate=config['optimization']['learning_rate'],
                    batch_size=config['optimization']['batch_size'],
                    num_iterations=config['optimization']['num_iterations']
                ),
                flow_theory=FlowTheoryConfig(
                    velocity_threshold=config['flow_theory']['velocity_threshold'],
                    flow_rate=config['flow_theory']['flow_rate']
                )
            )
    except FileNotFoundError:
        logger.warning(f"Config file not found: {file_path}")
        return Config(
            optimization=OptimizationConfig(
                algorithm=Algorithm.SNGD,
                learning_rate=DEFAULT_CONFIG['optimization']['learning_rate'],
                batch_size=DEFAULT_CONFIG['optimization']['batch_size'],
                num_iterations=DEFAULT_CONFIG['optimization']['num_iterations']
            ),
            flow_theory=FlowTheoryConfig(
                velocity_threshold=DEFAULT_CONFIG['flow_theory']['velocity_threshold'],
                flow_rate=DEFAULT_CONFIG['flow_theory']['flow_rate']
            )
        )
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        raise

def save_config(config: Config, file_path: str = CONFIG_FILE) -> None:
    """Save configuration to file."""
    with open(file_path, 'w') as f:
        yaml.dump({
            'optimization': {
                'algorithm': config.optimization.algorithm.value,
                'learning_rate': config.optimization.learning_rate,
                'batch_size': config.optimization.batch_size,
                'num_iterations': config.optimization.num_iterations
            },
            'flow_theory': {
                'velocity_threshold': config.flow_theory.velocity_threshold,
                'flow_rate': config.flow_theory.flow_rate
            }
        }, f, default_flow_style=False)

def get_config() -> Config:
    """Get the current configuration."""
    return load_config()

def update_config(config: Config) -> None:
    """Update the current configuration."""
    save_config(config)

# Define context manager
@contextmanager
def config_context(config: Config):
    """Context manager for configuration."""
    try:
        yield config
    finally:
        update_config(config)

# Define integration interfaces
class ConfigInterface:
    def get_config(self) -> Config:
        raise NotImplementedError

    def update_config(self, config: Config) -> None:
        raise NotImplementedError

class FileConfigInterface(ConfigInterface):
    def __init__(self, file_path: str = CONFIG_FILE):
        self.file_path = file_path

    def get_config(self) -> Config:
        return load_config(self.file_path)

    def update_config(self, config: Config) -> None:
        save_config(config, self.file_path)

# Define unit tests
import unittest

class TestConfig(unittest.TestCase):
    def test_load_config(self):
        config = load_config()
        self.assertIsInstance(config, Config)
        self.assertEqual(config.optimization.algorithm, Algorithm.SNGD)
        self.assertEqual(config.flow_theory.velocity_threshold, DEFAULT_CONFIG['flow_theory']['velocity_threshold'])

    def test_save_config(self):
        config = Config(
            optimization=OptimizationConfig(
                algorithm=Algorithm.SPRING,
                learning_rate=0.1,
                batch_size=64,
                num_iterations=500
            ),
            flow_theory=FlowTheoryConfig(
                velocity_threshold=0.2,
                flow_rate=0.6
            )
        )
        save_config(config)
        loaded_config = load_config()
        self.assertEqual(loaded_config.optimization.algorithm, Algorithm.SPRING)
        self.assertEqual(loaded_config.flow_theory.velocity_threshold, 0.2)

if __name__ == '__main__':
    unittest.main()