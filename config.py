import torch

# Global infrastructural operating boundary constraints
MAX_SEQUENCE_LENGTH = 50       # Rigid outer array padding layout limit
VOCABULARY_CAPACITY = 25000    # Hard RAM ceiling for dynamic categorical hashing
BATCH_SIZE = 64                # Increased batch size to saturate multiple GPU cores
SERVER_HOST = "0.0.0.0"        # Exposed to match container networks
SERVER_PORT = 8042

# Multi-GPU Detection Architecture
AVAILABLE_GPUS = torch.cuda.device_count()
DEVICE = torch.device("cuda:0" if AVAILABLE_GPUS > 0 else "cpu")

print(f"🖥️  [Hardware Init] Active Compute Device: {DEVICE} | Total Sub-GPUs Detected: {AVAILABLE_GPUS}")
