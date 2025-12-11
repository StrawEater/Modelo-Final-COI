import random 
from typing import List, Tuple, Dict

def remove_duplicates_and_fix_size(n_positions, max_count_n):
    return sorted(list(set(n_positions)))[:max_count_n]

def generate_continues_positions(start_pos, length):
    return list(range(start_pos, start_pos + length))

def generate_random_continuous_positions(max_length, right_edge) -> List[int]:
    """Generate a single continuous segment of Ns"""

    segment_length = random.randint(1, max_length)
    start_pos = random.randint(0, right_edge - segment_length)
    
    return generate_continues_positions(start_pos, segment_length)
    
def generate_few_segments(max_count_n, right_edge, max_new_segment_length) -> List[int]:
    """Generate 2-5 continuous segments"""
    
    num_segments = random.randint(2, 5)
    positions = []
    
    remaining_n = max_count_n
    
    for _ in range(num_segments):
        
        if remaining_n <= 0:
            break
        
        max_length = min(remaining_n, max_new_segment_length)
        segment = generate_random_continuous_positions(max_length, right_edge)

        positions.extend(segment)

        remaining_n -= len(segment)

    positions = remove_duplicates_and_fix_size(positions, max_count_n)
    return positions

def add_continues_segment_to_mutated(mutated, new_segment, max_count_n):
    mutated.n_positions.extend(new_segment)
    mutated.n_positions = remove_duplicates_and_fix_size(mutated.n_positions, max_count_n)

def remove_continues_segment_from_mutated(mutated, start, length):
    positions_to_remove = generate_continues_positions(start, length)
    mutated.n_positions = [p for p in mutated.n_positions if p not in positions_to_remove]

def get_correct_max_length_for_mutated(mutated, max_count_n, max_new_segment_length):
    ammount_of_ns_in_mutate = len(mutated.n_positions)
    remaining_n = max_count_n - ammount_of_ns_in_mutate
    return min(remaining_n, max_new_segment_length)

def extend_mutated_segment_right(mutated, segment_start, segment_length, right_edge, new_segment_length, max_count_n):
    
    segment_right_edge = segment_start + segment_length

    if (segment_right_edge == right_edge):
        return False

    if (segment_right_edge + new_segment_length) >= right_edge:
            new_segment_length =  right_edge - segment_right_edge - 1

    new_segment = generate_continues_positions(segment_right_edge, new_segment_length)
    add_continues_segment_to_mutated(mutated, new_segment, max_count_n)
    
    return True

def extend_mutated_segment_left(mutated, segment_start, new_segment_length, max_count_n):
    
    if (segment_start == 0):
        return False

    if segment_start - new_segment_length < 0:
        new_segment_length = segment_start

    new_segment_start = segment_start - new_segment_length

    new_segment = generate_continues_positions(new_segment_start, new_segment_length)
    add_continues_segment_to_mutated(mutated, new_segment, max_count_n)
    
    return True

################### MUTACIONES ###########################3

def add_continuous(mutated, max_count_n, max_new_segment_length):
    
    if (len(mutated.n_positions) == max_count_n):
        return

    max_length = get_correct_max_length_for_mutated(mutated, max_count_n, max_new_segment_length)
    right_edge = mutated.sequence_length
    
    new_segment = generate_random_continuous_positions(max_length, right_edge)
    add_continues_segment_to_mutated(mutated, new_segment, max_count_n)

def remove_segment(mutated):
    segments = mutated.get_continuous_segments()
    
    if not segments:
        return
    
    segment_to_remove = random.choice(segments)
    
    start, length = segment_to_remove
    remove_continues_segment_from_mutated(mutated, start, length)

def extend_segment(mutated, max_count_n, max_new_segment_length):
    
    segments = mutated.get_continuous_segments()
    
    if not segments:
        return
    
    segment_to_expand = random.choice(segments)

    start, length = segment_to_expand
    right_edge = mutated.sequence_length

    max_length = get_correct_max_length_for_mutated(mutated, max_count_n, max_new_segment_length)
    new_segment_length = random.randint(1, max_length)

    coin = (random.random() < 0.5)

    if coin:
        extended = extend_mutated_segment_right(mutated, start, length, right_edge, new_segment_length, max_count_n)
        if not extended:
            extend_mutated_segment_left(mutated, start, new_segment_length, max_count_n)

    else:
        extended = extend_mutated_segment_left(mutated, start, new_segment_length, max_count_n)
        if not extended:
            extend_mutated_segment_right(mutated, start, length, right_edge, new_segment_length, max_count_n)
        
def merge_segments(mutated, max_count_n, max_gap_size):
    
    segments = mutated.get_continuous_segments()
    
    if not segments:
        return
    
    segments_sorted = sorted(segments, key=lambda x: x[0])

    for i in range(len(segments_sorted) - 1):
        start1, len1 = segments_sorted[i]
        start2, len2 = segments_sorted[i + 1]

        gap = start2 - (start1 + len1)
        
        if gap > max_gap_size:
            continue

        if len(mutated.n_positions) + gap > max_count_n:
            continue
        

        new_segment = generate_continues_positions(start1 + len1, gap)
        add_continues_segment_to_mutated(mutated, new_segment, max_count_n)
        
        break

def split_segment(mutated, min_size_long_segment, max_length_cut):
    
    segments = mutated.get_continuous_segments()
    
    if not segments:
        return

    long_segments = [s for s in segments if s[1] > min_size_long_segment]

    if long_segments:
        
        segment = random.choice(long_segments)
        start, length = segment
        
        # Make sure we have room to cut
        # We need at least 1bp before cut, 1bp to cut, and 1bp after cut
        if length < 3:
            return
        
        # Cut can start anywhere except the first and last position
        cut_start = random.randint(start + 1, start + length - 2)
        
        # Calculate max possible cut length from this position
        max_possible_cut = (start + length) - cut_start - 1
        
        # Actual cut length (at least 1, at most max_length_cut or what's possible)
        max_cut = min(max_length_cut, max_possible_cut)
        
        if max_cut < 1:
            return
        
        cut_length = random.randint(1, max_cut)
        
        remove_continues_segment_from_mutated(mutated, cut_start, cut_length)


def shift_segment(mutated, max_count_n, shift_ammount):
    
    right_edge = mutated.sequence_length

    segments = mutated.get_continuous_segments()
    
    if not segments:
        return
    
    segment = random.choice(segments)
    
    start, length = segment
            
    possible_shifts = list(range(-shift_ammount, shift_ammount + 1))
    possible_shifts.remove(0)

    shift = random.choice(possible_shifts)

    new_start = max(0, min(right_edge - length, start + shift))
            
    if new_start != start:

        remove_continues_segment_from_mutated(mutated, start, length)
        
        new_segment = generate_continues_positions(new_start, length)
        add_continues_segment_to_mutated(mutated, new_segment, max_count_n)