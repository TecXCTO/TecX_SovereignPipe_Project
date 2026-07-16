import torch

# Global infrastructural operating boundary constraints
MAX_SEQUENCE_LENGTH = 50       # Rigid outer array padding layout limit
VOCABULARY_CAPACITY = 25000    # Hard RAM ceiling for dynamic categorical hashing
BATCH_SIZE = 32                # System memory aggregation boundary
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8042

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
