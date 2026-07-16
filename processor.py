import json
import numpy as np
from typing import Any, List, Tuple
from config import MAX_SEQUENCE_LENGTH, VOCABULARY_CAPACITY

class PolymorphicStreamProcessor:
    """Transforms arbitrary structures (dicts, matrices, strings, nested lists) directly into tensor spaces."""
    def __init__(self):
        self.vocab = {"<PAD>": 0, "<OOV>": 1}
        self.vocab_counter = 2

    def flatten_recursive(self, element: Any) -> List[Any]:
        """Recursively unwinds arbitrary deep structures (lists, dicts, primitives) down to a linear vector."""
        flat = []
        if isinstance(element, list):
            for item in element:
                flat.extend(self.flatten_recursive(item))
        elif isinstance(element, dict):
            for k, v in element.items():
                flat.extend(self.flatten_recursive(k))
                flat.extend(self.flatten_recursive(v))
        else:
            flat.append(element)
        return flat

    def map_to_numeric(self, token: Any) -> int:
        """Converts strings and numbers to unique token identifiers with strict OOV protections."""
        if isinstance(token, str):
            if token in self.vocab:
                return self.vocab[token]
            if self.vocab_counter >= VOCABULARY_CAPACITY:
                return self.vocab["<OOV>"]
            self.vocab[token] = self.vocab_counter
            self.vocab_counter += 1
            return self.vocab[token]
        else:
            try:
                # Retain numerical characteristics without structural mapping overlap collisions
                return int(float(token)) + 50000
            except (ValueError, TypeError):
                return self.vocab["<OOV>"]

    def vectorize_payload(self, raw_data_payload: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms arbitrary structural configurations directly into normalized, fixed-length matrices."""
        # 1. Unroll data objects
        flat_list = self.flatten_recursive(raw_data_payload)
        
        # 2. Translate multi-type elements to unique integer hashes
        numeric_sequence = [self.map_to_numeric(item) for item in flat_list]
        
        # 3. Instantiate padded arrays and boolean masks
        padded_array = np.zeros(MAX_SEQUENCE_LENGTH, dtype=np.int64)
        padding_mask = np.ones(MAX_SEQUENCE_LENGTH, dtype=bool) # True means mask out
        
        effective_len = min(len(numeric_sequence), MAX_SEQUENCE_LENGTH)
        padded_array[:effective_len] = numeric_sequence[:effective_len]
        padding_mask[:effective_len] = False # False means legitimate data
        
        return padded_array, padding_mask
