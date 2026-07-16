import asyncio
from fastapi import FastAPI, HTTPException, Response
import uvicorn
from typing import Dict, Any

from config import SERVER_HOST, SERVER_PORT
from orchestrator import SovereignMetaOrchestrator
from simulator import AutonomousDataScriptSimulator

# Import Prometheus metric generator exporter
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(title="SovereignPipe_Enterprise_Engine")
orchestrator = SovereignMetaOrchestrator()

@app.on_event("startup")
async def startup_pipeline_infra():
    # Initialize the decoupled background consumer worker routine instantly in the event loop
    asyncio.create_task(orchestrator.execute_continuous_worker_loop())
    # Fire up our parallel simulated injection process thread framework
    asyncio.create_task(run_continuous_input_ingestion_simulation())

@app.get("/metrics")
async def metrics_endpoint():
    """
    Prometheus Scrape Endpoint. Returns the raw structural string 
    telemetry data to your cluster scraping containers.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

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
    """Simulates thousands of high-speed multi-type payloads hitting the system pipeline."""
    await asyncio.sleep(2) # Give server loops clear runway to finalize startup tasks
    print("🚀 [Simulation System] Launching parallel telemetry-aware simulation streams...")
    
    for step in range(50000):
        mixed_data, ai_code, target_label = AutonomousDataScriptSimulator.output_next_transaction(step)
        await orchestrator.stage_transaction(mixed_data, ai_code, target_label)
        
        if step % 500 == 0:
            # Yield minimal execution breathing window to maintain async parity
            await asyncio.sleep(0.001)

if __name__ == "__main__":
    # Start the integrated production system
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="warning")
