import torch
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm
import random
from scipy.stats import entropy
from copy import deepcopy

from src.genetics.dna_sequence_perturber import *
from src.genetics.dna_sequence_perturber_mutators import *

class GeneticAlgorithmDNA:
    """Genetic Algorithm to find optimal N positions for maximizing prediction entropy"""
    
    def __init__(
        self,
        classifier,
        ranks_for_entropy,
        
        population_size: int = 50,
        
        max_n_count: int = 50,
        max_sequence_lengths: int = 900,
        
        entropy_weight: float = 100,
        n_penalty_weight: float = 0.01,
        discontinuity_penalty_weight: float = 0.05,
        
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
        tournament_size: int = 5,
        elitism_count: int = 5
    ):
        """
        Args:
            model: Trained DNA classifier model
            population_size: Number of individuals in population
            max_n_count: Maximum number of N positions allowed
            n_penalty_weight: Weight for penalizing number of Ns
            continuity_penalty_weight: Weight for penalizing discontinuous segments
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            tournament_size: Size of tournament for selection
            elitism_count: Number of best individuals to preserve
        """
        self.classifier = classifier
        self.ranks_for_entropy = ranks_for_entropy
        self.population_size = population_size
        self.max_n_count = max_n_count
        self.max_sequence_lengths = max_sequence_lengths
        self.entropy_weight = entropy_weight
        self.n_penalty_weight = n_penalty_weight
        self.discontinuity_penalty_weight = discontinuity_penalty_weight
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        
        # Cache for fitness evaluations
        self.fitness_cache = {}
    
    def _get_cache_key(self, individual: DNASequencePerturber) -> str:
        """Generate a cache key from N positions"""
        return ','.join(map(str, individual.n_positions))

    def calculate_prediction_entropy_batch(self, sequences: List[str]) -> List[float]:
            
        predictions = self.classifier(sequences)

        sequence_entropies = []
        # Get batch size (number of sequences)
        batch_size = predictions[0].shape[0]

        # For each sequence in the batch
        for seq_idx in range(batch_size):
            rank_entropies = []
            
            # For each rank we care about
            for rank_idx in self.ranks_for_entropy:
                # Get probabilities for this sequence at this rank
                rank_probs = predictions[rank_idx][seq_idx]
                rank_entropy = entropy(rank_probs)
                rank_entropies.append(rank_entropy)
            
            # Average entropy across ranks for this sequence
            avg_entropy_for_seq = np.mean(rank_entropies)
            sequence_entropies.append(avg_entropy_for_seq)

        return sequence_entropies
    
    def evaluate_fitness(self, individual: DNASequencePerturber, original_sequences: List[str]) -> float:
        
        cache_key = self._get_cache_key(individual)
        if cache_key in self.fitness_cache:
            cached = self.fitness_cache[cache_key]
            individual.fitness = cached['fitness']
            individual.entropy_score = cached['entropy']
            individual.n_count_penalty = cached['n_penalty']
            individual.continuity_penalty = cached['cont_penalty']
            return individual.fitness

        perturbed_batch = [individual.apply_to_sequence(seq) for seq in original_sequences]
        entropies = self.calculate_prediction_entropy_batch(perturbed_batch)

        relative_entropy = entropies/(self.current_entropy + 0.001)

        mean_entropy = np.mean(relative_entropy)
        entropy_score = mean_entropy * self.entropy_weight

        # Penalize number of Ns
        ns_score = len(individual.n_positions) / self.max_n_count
        ns_penalty = self.n_penalty_weight * ns_score
        
        # Penalize discontinuity (more segments = higher penalty)
        continuity_score = individual.calculate_continuity_score()
        discontinuity_penalty = self.discontinuity_penalty_weight * (1 - continuity_score)
        
        # Fitness is entropy minus penalties
        fitness = entropy_score - ns_penalty - discontinuity_penalty
        
        # Store components for analysis
        individual.fitness = fitness
        individual.entropy_score = entropy_score
        individual.n_count_penalty = ns_penalty
        individual.continuity_penalty = discontinuity_penalty

        self.fitness_cache[cache_key] = {
            'fitness': fitness,
            'entropy': mean_entropy,
            'n_penalty': ns_penalty,
            'cont_penalty': discontinuity_penalty
        }
        
        return fitness
    
    def initialize_population(self) -> List[DNASequencePerturber]:
        """Initialize random population with preference for continuous segments"""
        population = []
        
        for i in range(self.population_size):
            
            if i < self.population_size // 2:
                n_positions = generate_random_continuous_positions(self.max_sequence_lengths//10, self.max_sequence_lengths)
            
            else:
                n_positions = generate_few_segments(self.max_n_count, self.max_sequence_lengths, 40)
            
            individual = DNASequencePerturber(self.max_sequence_lengths, n_positions)
            population.append(individual)
        
        return population
    
    def tournament_selection(self, population: List[DNASequencePerturber]) -> DNASequencePerturber:
        """Select individual using tournament selection"""
        tournament = random.sample(population, self.tournament_size)
        winner = max(tournament, key=lambda ind: ind.fitness)
        return winner.copy(True)
    
    def crossover(
        self, 
        parent1: DNASequencePerturber, 
        parent2: DNASequencePerturber
    ) -> Tuple[DNASequencePerturber, DNASequencePerturber]:
        
        chance = random.random() 
        """Perform crossover between two parents - segment-aware"""
        if chance > self.crossover_rate:
            return parent1.copy(True), parent2.copy(True)
        
        # Get segments from both parents
        segments1 = parent1.get_continuous_segments()
        segments2 = parent2.get_continuous_segments()
        
        if not segments1 and not segments2:
            return parent1.copy(True), parent2.copy(True)
        
        # Randomly select segments from each parent
        child1_positions = []
        child2_positions = []
        
        all_segments = segments1 + segments2
        random.shuffle(all_segments)
        
        # Split segments between children
        for i, (start, length) in enumerate(all_segments):
            segment_positions = generate_continues_positions(start, length)
            
            if i % 2 == 0:
                child1_positions.extend(segment_positions)
            else:
                child2_positions.extend(segment_positions)
        
        child1_positions = remove_duplicates_and_fix_size(child1_positions, self.max_n_count)
        child2_positions = remove_duplicates_and_fix_size(child2_positions, self.max_n_count)
        
        child1 = DNASequencePerturber(parent1.sequence_length, child1_positions)
        child2 = DNASequencePerturber(parent2.sequence_length, child2_positions)
        
        return child1, child2
    
    def mutate(self, individual: DNASequencePerturber) -> DNASequencePerturber:
        
        """Mutate an individual - continuity-aware mutations"""
        if random.random() > self.mutation_rate:
            return individual.copy(True)
        
        mutated = individual.copy(False)
        # Choose mutation type with bias toward continuity-preserving mutations
        mutation_types = ['add_continuous','remove_segment', 'extend_segment','merge_segments','split_segment','shift_segment']

        mutation_type = random.choice(mutation_types)
        
        if mutation_type == 'add_continuous':
            add_continuous(mutated, self.max_n_count, 10)

        elif mutation_type == 'remove_segment':
            remove_segment(mutated)

        elif mutation_type == 'extend_segment':
            extend_segment(mutated, self.max_n_count, 10)

        elif mutation_type == 'merge_segments':
            merge_segments(mutated, self.max_n_count, 15)
        
        elif mutation_type == 'split_segment':
            split_segment(mutated, 40, 20)
        
        elif mutation_type == 'shift_segment':
            shift_segment(mutated, self.max_n_count, 10)

        else:
            assert False, "Hubo un problemas con las mutaciones"
        
        return mutated

    def evolve_generation(
        self, 
        population: List[DNASequencePerturber],
        original_sequences: List[str]
    ) -> List[DNASequencePerturber]:
        """Evolve one generation"""
        
        # Evaluate fitness for all individuals
        for ind in population:
            self.evaluate_fitness(ind, original_sequences)
        
        # Sort by fitness
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        
        best_ind = population[0]
        avg_fitness = np.mean([ind.fitness for ind in population])

        # Elitism: keep best individuals
        new_population = [ind.copy(True) for ind in population[:self.elitism_count]]
        
        # Generate rest of population through selection, crossover, and mutation
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
            
            # Crossover
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutation
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            # Add to new population
            new_population.extend([child1, child2])
        
        # Trim to population size
        return new_population[:self.population_size], best_ind, avg_fitness
    
    def run(
        self, 
        dataloader, 
        epochs: int = 10,
        patience = 150,
        verbose: bool = True
    ) -> Tuple[DNASequencePerturber, Dict]:
        """
        Run the genetic algorithm
        
        Returns:
            best_individual: Best solution found
            history: Dictionary with evolution history
        """
        
        # Initialize population
        population = self.initialize_population()
        
        # History tracking
        history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_entropy': [],
            'best_n_count': [],
            'best_n_count_penalty': [],
            'best_continuity_penalty': []
        }
        
        best_ind = None 
        best_fitness = 0
        avg_fitness = 0
        equal_count = 0
        drawback = 50

        for epoch in range(epochs):

            pbar = tqdm(dataloader, desc="Evolution")
            for sequences, _ in pbar:

                self.current_entropy = self.calculate_prediction_entropy_batch(sequences)
                population, best_ind, avg_fitness = self.evolve_generation(population, sequences)

                new_best_fitness = best_ind.fitness
                if new_best_fitness > best_fitness:
                    best_fitness = new_best_fitness
                    equal_count = max(0, equal_count - 75)
                else:
                    equal_count += 1
                
                if(equal_count >= patience):
                    equal_count -= drawback
                    drawback = drawback//2
                    break

                pbar.set_postfix({
                    'fitness': f'{best_ind.fitness:.4f}',
                    'entropy': f'{best_ind.entropy_score:.4f}',
                    'n_count': len(best_ind.n_positions),
                    'patience': (patience - equal_count)
                })

            print(f"\nEpoch [{epoch+1}/{epochs}]")
            print("\nGenetic Algorithm Status")
            print(f"  Best Fitness:              {best_ind.fitness:.4f}")
            print(f"  Average Fitness:           {avg_fitness:.4f}")
            print(f"  Best Entropy Score:        {best_ind.entropy_score:.4f}")
            print(f"  Best N Count:              {len(best_ind.n_positions)}")
            print(f"  Best N Count Penalty:      {best_ind.n_count_penalty:.4f}")
            print(f"  Best Continuity Penalty:   {best_ind.continuity_penalty:.4f}")
            print(f"\n{vars(best_ind)}")

            history['best_fitness'].append(best_ind.fitness)
            history['avg_fitness'].append(avg_fitness)
            history['best_entropy'].append(best_ind.entropy_score)
            history['best_n_count'].append(len(best_ind.n_positions))
            history['best_n_count_penalty'].append(best_ind.n_count_penalty)
            history['best_continuity_penalty'].append(best_ind.continuity_penalty)
            
        return best_ind, history, population
