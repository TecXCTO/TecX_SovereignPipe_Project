import sys
import traceback
import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import BATCH_SIZE, DEVICE
from security import CompileTimeSandbox
from processor import PolymorphicStreamProcessor
from models import PolymorphicTransformerEngine

class SovereignMetaOrchestrator:
    """Manages raw code orchestration queues and guides internal AI network updates."""
    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self.processor = PolymorphicStreamProcessor()
        self.model = PolymorphicTransformerEngine().to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.BCELoss()

    async def stage_transaction(self, raw_data_mixed: Any, custom_code_str: str, label: float):
        """Stages transactional inputs directly onto the local processing line track."""
        await self.queue.put((raw_data_mixed, custom_code_str, label))

    async def execute_continuous_worker_loop(self):
        """Continuous pipeline loop. Pulls variable packages, runs custom code, and updates weights."""
        print("⚙️  [Orchestrator] Dynamic Multi-Type Processing Worker Active.")
        b_X, b_masks, b_y = [], [], []
        
        while True:
            raw_data, code_str, label = await self.queue.get()
            try:
                # Phase 1: Isolated Code Injection Execution
                sandbox = CompileTimeSandbox.get_secure_env()
                bytecode = compile(code_str, filename="<runtime_dynamic_pipe>", mode="exec")
                exec(bytecode, sandbox)
                
                # Check for standard pipeline transformation hook override interface
                if "custom_pipeline_transform" in sandbox:
                    transform_hook = sandbox["custom_pipeline_transform"]
                    processed_payload = transform_hook(raw_data)
                else:
                    processed_payload = raw_data # Pass-through strategy if no interface hook exists
                
                # Phase 2: Vectorization layer conversion
                x_pad, x_mask = self.processor.vectorize_payload(processed_payload)
                
                b_X.append(x_pad)
                b_masks.append(x_mask)
                b_y.append(label)
                
                # Phase 3: Aggregation optimization step
                if len(b_X) == BATCH_SIZE:
                    X_tensor = torch.tensor(np.array(b_X), dtype=torch.long).to(DEVICE)
                    mask_tensor = torch.tensor(np.array(b_masks), dtype=torch.bool).to(DEVICE)
                    y_tensor = torch.tensor(np.array(b_y), dtype=torch.float32).view(-1, 1).to(DEVICE)
                    
                    self.model.train()
                    self.optimizer.zero_grad()
                    loss = self.loss_fn(self.model(X_tensor, mask_tensor), y_tensor)
                    loss.backward()
                    self.optimizer.step()
                    
                    print(f"📈 [Pipeline Progress] Batch optimized inside RAM. Batch Loss: {loss.item():.5f} | Registered Unique Tokens: {self.processor.vocab_counter}")
                    b_X, b_masks, b_y = [], [], []
                    
            except Exception as pipeline_fault:
                print(f"❌ [Pipeline Runtime Exception Intercepted]: {pipeline_fault}")
                traceback.print_exc()
            finally:
                self.queue.task_done()
