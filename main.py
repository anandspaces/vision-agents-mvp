"""
FastAPI that runs the CLI as subprocess and captures the demo URL
"""

import asyncio
import logging
import re
import sys
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="Vision Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active processes
active_processes = {}

# ============================================
# CLI SCRIPT TEMPLATE
# ============================================

CLI_SCRIPT = '''
import logging
from dotenv import load_dotenv
from vision_agents.core import User, Agent, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import getstream, gemini

logger = logging.getLogger(__name__)
load_dotenv()

async def create_agent(**kwargs) -> Agent:
    llm = gemini.Realtime()
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="My happy AI friend", id="agent"),
        instructions="You're a voice AI assistant. Keep responses short and conversational. Don't use special characters or formatting. Be friendly and helpful.",
        llm=llm,
    )
    return agent

async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    await agent.create_user()
    call = await agent.create_call(call_type, call_id)
    logger.info("🤖 Starting Gemini Realtime Agent...")
    
    with await agent.join(call):
        logger.info("Joining call")
        await agent.edge.open_demo(call)
        logger.info("LLM ready")
        await agent.llm.simple_response("chat with the user about them.")
        await agent.finish()

if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
'''

# ============================================
# HELPER FUNCTIONS
# ============================================

def clean_url(url: str) -> str:
    """
    Remove ANSI escape codes and other unwanted characters from URL
    """
    # Remove ANSI escape codes
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    url = ansi_escape.sub('', url)
    
    # Remove any remaining escape sequences
    url = re.sub(r'\\u001b\[[0-9;]*m', '', url)
    
    # Remove any whitespace
    url = url.strip()
    
    return url

# ============================================
# API ENDPOINT
# ============================================

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "Vision Agent API",
        "endpoint": "/start-agent",
        "active_agents": len(active_processes)
    }

@app.get("/stop-agent/{call_id}")
async def stop_agent(call_id: str):
    """
    Stop a running agent by call_id
    
    Usage:
    ```bash
    curl http://localhost:8000/stop-agent/YOUR_CALL_ID
    ```
    """
    if call_id not in active_processes:
        return {
            "status": "error",
            "message": f"No active agent found for call_id: {call_id}"
        }
    
    try:
        process_info = active_processes[call_id]
        process = process_info['process']
        
        # Kill process
        process.kill()
        await process.wait()
        
        # Clean up temp file
        try:
            import os
            os.unlink(process_info['script_path'])
        except:
            pass
        
        # Remove from active
        del active_processes[call_id]
        
        logger.info(f"🛑 Stopped agent for call: {call_id}")
        
        return {
            "status": "success",
            "message": f"Agent stopped for call_id: {call_id}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/list-agents")
async def list_agents():
    """List all active agents"""
    return {
        "status": "success",
        "active_agents": len(active_processes),
        "call_ids": list(active_processes.keys())
    }

@app.get("/start-agent")
async def start_agent():
    """
    Start the agent by running CLI as subprocess and capture demo URL
    
    Returns:
    ```json
    {
        "status": "success",
        "demo_url": "https://getstream.io/video/demos/join/...",
        "call_id": "..."
    }
    ```
    """
    try:
        call_id = str(uuid.uuid4())
        logger.info(f"🚀 Starting agent subprocess for call: {call_id}")
        
        # Write temporary script file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(CLI_SCRIPT)
            script_path = f.name
        
        try:
            # Run the CLI script as subprocess
            process = await asyncio.create_subprocess_exec(
                'python', script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=os.environ.copy()
            )
            
            # Read output line by line
            demo_url = None
            timeout = 30  # 30 seconds timeout
            start_time = asyncio.get_event_loop().time()
            
            # Wait for URL and keep logging
            while True:
                # Check timeout
                if asyncio.get_event_loop().time() - start_time > timeout:
                    logger.error("Timeout waiting for URL")
                    process.kill()
                    break
                
                try:
                    line = await asyncio.wait_for(
                        process.stdout.readline(),
                        timeout=1.0
                    )
                    
                    if not line:
                        break
                    
                    line = line.decode('utf-8').strip()
                    logger.info(f"[subprocess] {line}")
                    
                    # Look for the demo URL
                    if "Opening browser to:" in line or "🌐 Opening browser to:" in line:
                        url_match = re.search(r'(https://getstream\.io/video/demos/join/[^\s]+)', line)
                        if url_match:
                            demo_url = clean_url(url_match.group(1))
                            logger.info(f"✅ Captured URL: {demo_url[:100]}...")
                            # URL found - break to return it
                            break
                    
                    # Alternative: look for xdg-open error
                    if "xdg-open: no method available for opening" in line:
                        url_match = re.search(r"opening '(https://[^']+)'", line)
                        if url_match:
                            demo_url = clean_url(url_match.group(1))
                            logger.info(f"✅ Captured URL from xdg-open: {demo_url[:100]}...")
                            break
                    
                except asyncio.TimeoutError:
                    continue
            
            # After URL is captured, continue monitoring in background
            async def continue_monitoring():
                """Keep logging subprocess output"""
                try:
                    while True:
                        line = await process.stdout.readline()
                        if not line:
                            break
                        line = line.decode('utf-8').strip()
                        if line:
                            logger.info(f"[subprocess] {line}")
                except Exception as e:
                    logger.error(f"Monitoring stopped: {e}")
            
            # Return result
            if demo_url:
                # Extract call_id from URL
                call_id_match = re.search(r'/join/([a-f0-9-]+)', demo_url)
                actual_call_id = call_id_match.group(1) if call_id_match else call_id
                
                # Start background monitoring
                monitor_task = asyncio.create_task(continue_monitoring())
                
                # Store process so it keeps running
                active_processes[actual_call_id] = {
                    'process': process,
                    'script_path': script_path,
                    'monitor_task': monitor_task
                }
                
                logger.info(f"✅ Agent running in background for call: {actual_call_id}")
                
                return {
                    "status": "success",
                    "demo_url": demo_url,
                    "call_id": actual_call_id,
                    "message": "Agent started successfully. Open the demo_url to join the call.",
                    "note": "Agent process running in background"
                }
            else:
                process.kill()
                return {
                    "status": "error",
                    "message": "Could not capture demo URL from subprocess",
                    "call_id": call_id
                }
                
        finally:
            # DON'T clean up temp file or kill process - keep it running
            pass
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e)
        }

# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Vision Agent API Server...")
    logger.info("📡 API: http://localhost:8000/start-agent")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )