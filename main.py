import asyncio
from fastapi import FastAPI, HTTPException
import uvicorn
from typing import Dict, Any
from config import SERVER_HOST, SERVER_PORT
from orchestrator import SovereignMetaOrchestrator
from simulator import AutonomousDataScriptSimulator

app = FastAPI(title="SovereignPipe_Core_Engine")
orchestrator = SovereignMetaOrchestrator()

@app.on_event("startup")
async def startup_pipeline_infra():
    # Initialize the decoupled background consumer worker routine instantly in the event loop
    asyncio.create_task(orchestrator.execute_continuous_worker_loop())
    # Fire up our parallel simulated injection process thread framework
    asyncio.create_task(run_continuous_input_ingestion_simulation())

@app.post("/gateway/push")
async def ingest_heterogeneous_packet(payload: Dict[str, Any]):
    """Unified entry point for completely unstructured data payloads and transformation code scripts."""
    data_packet = payload.get("data")
    code_packet = payload.get("code")
    label_value = payload.get("label", 0.0)
    
    if data_packet is None or code_packet is None:
        raise HTTPException(status_code=400, detail="Missing baseline structure payload specifications.")
        
    await orchestrator.stage_transaction(data_packet, code_packet, float(label_value))
    return {"status": "ENQUEUED", "active_queue_depth": orchestrator.queue.qsize()}

async def run_continuous_input_ingestion_simulation():
    """Simulates thousands of concurrent updates from an AI system or client data lines."""
    await asyncio.sleep(2) # Give server loops clear runway to finalize startup tasks
    print("🚀 [Simulation System] Launching polymorphic automated data generation feed loops...")
    
    # Push 10,000 mixed-structure samples through the pipeline to demonstrate multi-format compatibility
    for step in range(10000):
        mixed_data, ai_code, target_label = AutonomousDataScriptSimulator.output_next_transaction(step)
        await orchestrator.stage_transaction(mixed_data, ai_code, target_label)
        
        if step % 200 == 0:
            await asyncio.sleep(0.01) # Yield tiny time blocks to ensure execution loops stay fluid

if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
