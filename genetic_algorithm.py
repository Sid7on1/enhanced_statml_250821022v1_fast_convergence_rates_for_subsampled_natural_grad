import logging
import numpy as np
import torch
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants
POPULATION_SIZE = 100
GENERATIONS = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.5
SELECTION_RATE = 0.5

# Define data structures
@dataclass
class Individual:
    """Represents an individual in the population."""
    id: int
    fitness: float
    genome: List[float]

@dataclass
class Population:
    """Represents the population of individuals."""
    individuals: List[Individual]

# Define exception classes
class GeneticAlgorithmException(Exception):
    """Base exception class for genetic algorithm exceptions."""
    pass

class InvalidPopulationSizeException(GeneticAlgorithmException):
    """Raised when the population size is invalid."""
    pass

class InvalidGenerationCountException(GeneticAlgorithmException):
    """Raised when the generation count is invalid."""
    pass

# Define the genetic algorithm class
class GeneticAlgorithm:
    """Implements the genetic algorithm."""
    def __init__(self, population_size: int = POPULATION_SIZE, generations: int = GENERATIONS, 
                 mutation_rate: float = MUTATION_RATE, crossover_rate: float = CROSSOVER_RATE, 
                 selection_rate: float = SELECTION_RATE):
        """
        Initializes the genetic algorithm.

        Args:
        - population_size (int): The size of the population.
        - generations (int): The number of generations.
        - mutation_rate (float): The mutation rate.
        - crossover_rate (float): The crossover rate.
        - selection_rate (float): The selection rate.
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_rate = selection_rate
        self.population = self.initialize_population()

    def initialize_population(self) -> Population:
        """
        Initializes the population with random individuals.

        Returns:
        - population (Population): The initialized population.
        """
        individuals = []
        for i in range(self.population_size):
            genome = np.random.rand(10).tolist()  # Random genome
            fitness = self.calculate_fitness(genome)
            individual = Individual(i, fitness, genome)
            individuals.append(individual)
        return Population(individuals)

    def calculate_fitness(self, genome: List[float]) -> float:
        """
        Calculates the fitness of an individual.

        Args:
        - genome (List[float]): The genome of the individual.

        Returns:
        - fitness (float): The fitness of the individual.
        """
        # Calculate fitness using the formula from the paper
        fitness = sum([x**2 for x in genome])
        return fitness

    def selection(self) -> List[Individual]:
        """
        Selects individuals for the next generation.

        Returns:
        - selected_individuals (List[Individual]): The selected individuals.
        """
        selected_individuals = []
        for _ in range(int(self.population_size * self.selection_rate)):
            individual = np.random.choice(self.population.individuals)
            selected_individuals.append(individual)
        return selected_individuals

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """
        Performs crossover between two parents.

        Args:
        - parent1 (Individual): The first parent.
        - parent2 (Individual): The second parent.

        Returns:
        - child (Individual): The child individual.
        """
        child_genome = []
        for i in range(len(parent1.genome)):
            if np.random.rand() < self.crossover_rate:
                child_genome.append(parent1.genome[i])
            else:
                child_genome.append(parent2.genome[i])
        child_fitness = self.calculate_fitness(child_genome)
        child = Individual(len(self.population.individuals), child_fitness, child_genome)
        return child

    def mutation(self, individual: Individual) -> Individual:
        """
        Performs mutation on an individual.

        Args:
        - individual (Individual): The individual to mutate.

        Returns:
        - mutated_individual (Individual): The mutated individual.
        """
        mutated_genome = individual.genome.copy()
        for i in range(len(mutated_genome)):
            if np.random.rand() < self.mutation_rate:
                mutated_genome[i] += np.random.randn()
        mutated_fitness = self.calculate_fitness(mutated_genome)
        mutated_individual = Individual(individual.id, mutated_fitness, mutated_genome)
        return mutated_individual

    def next_generation(self) -> None:
        """
        Generates the next generation of individuals.
        """
        selected_individuals = self.selection()
        children = []
        for _ in range(self.population_size - len(selected_individuals)):
            parent1, parent2 = np.random.choice(selected_individuals, 2, replace=False)
            child = self.crossover(parent1, parent2)
            children.append(child)
        mutated_individuals = [self.mutation(individual) for individual in selected_individuals]
        self.population.individuals = selected_individuals + children + mutated_individuals

    def run(self) -> None:
        """
        Runs the genetic algorithm.
        """
        for generation in range(self.generations):
            logger.info(f"Generation {generation+1}")
            self.next_generation()
            best_individual = max(self.population.individuals, key=lambda x: x.fitness)
            logger.info(f"Best individual: {best_individual.id}, Fitness: {best_individual.fitness}")

# Define a dataset class for the genetic algorithm
class GeneticAlgorithmDataset(Dataset):
    """A dataset class for the genetic algorithm."""
    def __init__(self, population: Population):
        self.population = population

    def __len__(self) -> int:
        return len(self.population.individuals)

    def __getitem__(self, index: int) -> Tuple[Individual, int]:
        individual = self.population.individuals[index]
        return individual, index

# Define a data loader class for the genetic algorithm
class GeneticAlgorithmDataLoader(DataLoader):
    """A data loader class for the genetic algorithm."""
    def __init__(self, dataset: GeneticAlgorithmDataset, batch_size: int = 32, shuffle: bool = True):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)

# Define a main function to run the genetic algorithm
def main() -> None:
    genetic_algorithm = GeneticAlgorithm()
    genetic_algorithm.run()

if __name__ == "__main__":
    main()