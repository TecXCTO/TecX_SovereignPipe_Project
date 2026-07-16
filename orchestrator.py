import sys
import time
import traceback
import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import BATCH_SIZE, DEVICE, AVAILABLE_GPUS
from security import CompileTimeSandbox
from processor import PolymorphicStreamProcessor
from models import PolymorphicTransformerEngine

# Import Prometheus instrumentation client
from prometheus_client import Counter, Gauge, Histogram

# =====================================================================
# PROMETHEUS METRICS INSTRUMENTATION DEFINITIONS
# =====================================================================
METRIC_TOTAL_INGESTED = Counter("sovereignpipe_ingested_payloads_total", "Total count of unstructured objects fed into gateway.")
METRIC_TOTAL_PROCESSED = Counter("sovereignpipe_processed_batches_total", "Total number of data batches successfully optimized.")
METRIC_PIPELINE_CRASHES = Counter("sovereignpipe_runtime_failures_total", "Total runtime exceptions caught inside AI sandbox closures.", ["stage"])
METRIC_QUEUE_DEPTH = Gauge("sovereignpipe_broker_queue_depth", "Current number of payloads waiting in memory queue.")
METRIC_BATCH_LOSS = Gauge("sovereignpipe_training_loss", "Instantaneous loss of the transformer network.")
METRIC_VOCAB_SIZE = Gauge("sovereignpipe_vocabulary_unique_tokens", "Total unique strings hashed inside RAM arrays.")
METRIC_LATENCY = Histogram("sovereignpipe_batch_execution_seconds", "Time distribution spent transforming and training a data batch.")

class SovereignMetaOrchestrator:
    """Manages raw code orchestration queues and guides internal multi-GPU AI network updates."""
    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=50000)
        self.processor = PolymorphicStreamProcessor()
        
        # 1. Base Model instantiation
        base_model = PolymorphicTransformerEngine()
        
        # 2. Multi-GPU Scalability Multiwrap Hook
        if AVAILABLE_GPUS > 1:
            print(f"🚀 [Parallelism Core] Wrapping neural layers inside torch.nn.DataParallel across {AVAILABLE_GPUS} GPUs.")
            self.model = nn.DataParallel(base_model)
        else:
            self.model = base_model
            
        self.model.to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.BCELoss()

    async def stage_transaction(self, raw_data_mixed: Any, custom_code_str: str, label: float):
        """Stages transactional inputs directly onto the local processing line track."""
        await self.queue.put((raw_data_mixed, custom_code_str, label))
        METRIC_TOTAL_INGESTED.inc()
        METRIC_QUEUE_DEPTH.set(self.queue.qsize())

    async def execute_continuous_worker_loop(self):
        """Continuous pipeline loop. Pulls variable packages, runs custom code, and updates weights."""
        print("⚙️  [Orchestrator] Multi-GPU & Telemetry-Aware Worker Active.")
        b_X, b_masks, b_y = [], [], []
        
        while True:
            raw_data, code_str, label = await self.queue.get()
            METRIC_QUEUE_DEPTH.set(self.queue.qsize())
            
            start_time = time.time()
            try:
                # Phase 1: Isolated Code Injection Execution
                sandbox = CompileTimeSandbox.get_secure_env()
                try:
                    bytecode = compile(code_str, filename="<runtime_dynamic_pipe>", mode="exec")
                    exec(bytecode, sandbox)
                except Exception as compile_err:
                    METRIC_PIPELINE_CRASHES.labels(stage="compilation").inc()
                    raise compile_err
                
                # Check for standard pipeline transformation hook override interface
                if "custom_pipeline_transform" in sandbox:
                    transform_hook = sandbox["custom_pipeline_transform"]
                    try:
                        processed_payload = transform_hook(raw_data)
                    except Exception as tx_err:
                        METRIC_PIPELINE_CRASHES.labels(stage="transformation").inc()
                        raise tx_err
                else:
                    processed_payload = raw_data
                
                # Phase 2: Vectorization layer conversion
                x_pad, x_mask = self.processor.vectorize_payload(processed_payload)
                METRIC_VOCAB_SIZE.set(self.processor.vocab_counter)
                
                b_X.append(x_pad)
                b_masks.append(x_mask)
                b_y.append(label)
                
                # Phase 3: Multi-GPU Parallelized Aggregation Optimization Step
                if len(b_X) == BATCH_SIZE:
                    X_tensor = torch.tensor(np.array(b_X), dtype=torch.long).to(DEVICE)
                    mask_tensor = torch.tensor(np.array(b_masks), dtype=torch.bool).to(DEVICE)
                    y_tensor = torch.tensor(np.array(b_y), dtype=torch.float32).view(-1, 1).to(DEVICE)
                    
                    self.model.train()
                    self.optimizer.zero_grad()
                    
                    # DataParallel splits this batch evenly across all GPUs automatically
                    predictions = self.model(X_tensor, mask_tensor)
                    loss = self.loss_fn(predictions, y_tensor)
                    
                    loss.backward()
                    self.optimizer.step()
                    
                    # Update Prometheus Gauges and Metrics
                    METRIC_BATCH_LOSS.set(float(loss.item()))
                    METRIC_TOTAL_PROCESSED.inc()
                    METRIC_LATENCY.observe(time.time() - start_time)
                    
                    print(f"📈 [Parallel Batch Complete] Loss: {loss.item():.5f} | Queue Depth: {self.queue.qsize()}")
                    b_X, b_masks, b_y = [], [], []
                    
            except Exception as pipeline_fault:
                METRIC_PIPELINE_CRASHES.labels(stage="pipeline_runtime").inc()
                print(f"❌ [Pipeline Intercept]: {pipeline_fault}")
            finally:
                self.queue.task_done()
