from typing import List, Tuple, Dict


class DNASequencePerturber:
    """Represents a solution: positions where to place 'N' in a DNA sequence"""
    
    def __init__(self, sequence_length: int, n_positions: List[int] = None):
        self.sequence_length = sequence_length
        self.n_positions = n_positions if n_positions is not None else []
        self.fitness = None
        self.entropy_score = None
        self.n_count_penalty = None
        self.continuity_penalty = None
    
    def apply_to_sequence(self, original_sequence: str) -> str:
        """Apply the N positions to create a perturbed sequence"""
        seq_list = list(original_sequence)
        for pos in self.n_positions:
            if pos < len(seq_list):
                seq_list[pos] = 'N'
        return ''.join(seq_list)
    
    def get_continuous_segments(self) -> List[Tuple[int, int]]:
        """
        Get continuous segments of N positions
        Returns list of (start, length) tuples
        """
        if not self.n_positions:
            return []
        
        segments = []
        current_start = self.n_positions[0]
        current_length = 1
        
        for i in range(1, len(self.n_positions)):
            if self.n_positions[i] == self.n_positions[i-1] + 1:
                # Continuous
                current_length += 1
            else:
                # Break in continuity
                segments.append((current_start, current_length))
                current_start = self.n_positions[i]
                current_length = 1
        
        # Don't forget last segment
        segments.append((current_start, current_length))
        
        return segments

    def calculate_continuity_score(self) -> float:
        """
        Calculate continuity score - rewards continuous segments
        Returns a value between 0 (all isolated) and 1 (all continuous)
        """
        if len(self.n_positions) <= 1:
            return 1.0
        
        segments = self.get_continuous_segments()
        
        # Ideal case: all Ns in one segment
        # Worst case: all Ns isolated (n_positions segments)
        num_segments = len(segments)
        max_segments = len(self.n_positions)  # Worst case
        min_segments = 1  # Best case
        
        # Normalize: 0 = worst (many segments), 1 = best (few segments)
        if max_segments == min_segments:
            return 1.0
        
        continuity = 1.0 - (num_segments - min_segments) / (max_segments - min_segments)
        return continuity
    
    def copy(self, save_others):
        """Create a deep copy of the individual"""
        new_ind = DNASequencePerturber(self.sequence_length, self.n_positions.copy())
        
        if save_others:
            new_ind.fitness = self.fitness
            new_ind.entropy_score = self.entropy_score
            new_ind.n_count_penalty = self.n_count_penalty
            new_ind.continuity_penalty = self.continuity_penalty
        
        return new_ind
