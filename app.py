import uuid
import logging
import asyncio
from typing import Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from vision_agents.core import User, Agent
from vision_agents.plugins import getstream, gemini

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# In-memory session store
ACTIVE_SESSIONS: Dict[str, Agent] = {}
BACKGROUND_TASKS: Dict[str, asyncio.Task] = {}

# ==================== Lifespan Manager ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan with proper startup and shutdown."""
    # Startup
    logger.info("🚀 Voice Agent Server starting up...")
    logger.info("📡 WebSocket server ready")
    logger.info("⚡ Background task workers initialized")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down server...")
    
    # Cancel all background tasks
    for session_id, task in list(BACKGROUND_TASKS.items()):
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"🚫 Cancelled background task: {session_id[:8]}")
    
    # Cleanup all active sessions
    session_ids = list(ACTIVE_SESSIONS.keys())
    for session_id in session_ids:
        agent = ACTIVE_SESSIONS.pop(session_id, None)
        if agent:
            try:
                await agent.finish()
                logger.info(f"🧹 Cleaned up session: {session_id[:8]}")
            except Exception as e:
                logger.error(f"❌ Error cleaning up session {session_id[:8]}: {e}")
    
    logger.info("✅ All sessions and tasks cleaned up")

app = FastAPI(
    title="Voice Agent WebSocket Server",
    description="Multi-user voice agent server - WebSocket-first approach",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Pydantic Models ====================

class SessionInfo(BaseModel):
    session_id: str
    active: bool
    agent_name: str
    task_running: bool

class HealthResponse(BaseModel):
    status: str
    active_sessions: int
    background_tasks: int

# ==================== Agent Factory ====================

async def create_agent(session_id: str) -> Agent:
    """Create a new agent instance for a session."""
    llm = gemini.Realtime()

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(
            id=f"agent-{session_id}",
            name=f"AI Friend {session_id[:8]}"
        ),
        instructions=(
            "You're a voice AI assistant. "
            "Keep responses short and conversational. "
            "Don't use special characters or formatting. "
            "Be friendly and helpful."
        ),
        llm=llm,
    )

    await agent.create_user()
    logger.info(f"✅ Agent created for session: {session_id[:8]}")
    return agent

# ==================== Background Worker ====================

async def agent_worker(session_id: str, call_type: str = "default"):
    """
    Background worker that runs the agent lifecycle.
    This runs independently per session and handles the call.
    """
    logger.info(f"🔧 Worker started for session: {session_id[:8]}")
    
    agent = ACTIVE_SESSIONS.get(session_id)
    if not agent:
        logger.error(f"❌ No agent found for session: {session_id[:8]}")
        return
    
    try:
        # Create the call
        call_id = session_id
        call = await agent.create_call(call_type, call_id)
        logger.info(f"📞 Call created in worker: {session_id[:8]}")
        
        # Join the call and start the agent
        async with await agent.join(call):
            logger.info(f"🎙️  Agent joined call in worker: {session_id[:8]}")
            
            # Open demo interface
            await agent.edge.open_demo(call)
            logger.info(f"✨ Demo interface opened: {session_id[:8]}")
            
            # Start the conversation
            await agent.llm.simple_response("chat with the user about them.")
            
            # Keep the agent running until the call ends
            await agent.finish()
            logger.info(f"🏁 Agent finished naturally: {session_id[:8]}")
    
    except asyncio.CancelledError:
        logger.info(f"🚫 Worker cancelled for session: {session_id[:8]}")
        raise
    
    except Exception as e:
        logger.error(f"❌ Worker error for session {session_id[:8]}: {e}")
    
    finally:
        # Cleanup
        try:
            await agent.finish()
        except:
            pass
        
        ACTIVE_SESSIONS.pop(session_id, None)
        BACKGROUND_TASKS.pop(session_id, None)
        logger.info(f"🧹 Worker cleanup complete: {session_id[:8]}")

# ==================== REST API Endpoints (Read-only) ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_sessions": len(ACTIVE_SESSIONS),
        "background_tasks": len(BACKGROUND_TASKS)
    }

@app.get("/sessions")
async def list_sessions():
    """List all active sessions and their worker status."""
    sessions = []
    for sid in ACTIVE_SESSIONS.keys():
        task = BACKGROUND_TASKS.get(sid)
        agent = ACTIVE_SESSIONS.get(sid)
        sessions.append({
            "session_id": sid[:8] + "...",
            "full_session_id": sid,
            "agent_name": agent.agent_user.name if agent else "Unknown",
            "worker_running": task is not None and not task.done()
        })
    
    return {
        "active_sessions": len(ACTIVE_SESSIONS),
        "background_tasks": len(BACKGROUND_TASKS),
        "sessions": sessions
    }

@app.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Get information about a specific session."""
    agent = ACTIVE_SESSIONS.get(session_id)
    
    if not agent:
        raise HTTPException(status_code=404, detail="Session not found")
    
    task = BACKGROUND_TASKS.get(session_id)
    task_running = task is not None and not task.done()
    
    return {
        "session_id": session_id,
        "active": True,
        "agent_name": agent.agent_user.name,
        "task_running": task_running
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Manually terminate a session and its background worker."""
    agent = ACTIVE_SESSIONS.get(session_id)
    task = BACKGROUND_TASKS.get(session_id)
    
    if not agent and not task:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Cancel background task
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Cleanup agent
        if agent:
            await agent.finish()
            ACTIVE_SESSIONS.pop(session_id, None)
        
        BACKGROUND_TASKS.pop(session_id, None)
        
        logger.info(f"🗑️  Session terminated: {session_id[:8]}")
        return {"status": "terminated", "session_id": session_id}
    except Exception as e:
        logger.error(f"❌ Error terminating session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to terminate session: {str(e)}")

# ==================== Main WebSocket Endpoint ====================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint - handles session creation and agent lifecycle.
    
    Client connects here directly, session is created automatically.
    No REST API needed for session creation!
    """
    session_id = str(uuid.uuid4())
    await websocket.accept()
    
    logger.info(f"🔌 New WebSocket connection: {session_id[:8]}")
    
    try:
        # Send session info to client
        await websocket.send_json({
            "type": "session_created",
            "session_id": session_id,
            "status": "initializing"
        })
        
        # Create agent
        agent = await create_agent(session_id)
        ACTIVE_SESSIONS[session_id] = agent
        
        # Start background worker
        task = asyncio.create_task(agent_worker(session_id))
        BACKGROUND_TASKS[session_id] = task
        
        await websocket.send_json({
            "type": "agent_ready",
            "session_id": session_id,
            "agent_name": agent.agent_user.name,
            "status": "ready"
        })
        
        logger.info(f"✅ Session fully initialized: {session_id[:8]}")
        
        # Main message loop
        while True:
            try:
                # Receive messages from client
                data = await websocket.receive_text()
                logger.debug(f"📨 Received from {session_id[:8]}: {data}")
                
                # Handle commands
                if data == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": asyncio.get_event_loop().time()
                    })
                
                elif data == "status":
                    worker_status = "running" if task and not task.done() else "stopped"
                    await websocket.send_json({
                        "type": "status",
                        "session_id": session_id,
                        "worker_status": worker_status,
                        "agent_name": agent.agent_user.name
                    })
                
                elif data == "disconnect":
                    logger.info(f"👋 Client requested disconnect: {session_id[:8]}")
                    await websocket.send_json({
                        "type": "disconnecting",
                        "message": "Goodbye!"
                    })
                    break
                
                else:
                    # Echo back or handle custom messages
                    await websocket.send_json({
                        "type": "message_received",
                        "your_message": data
                    })
                
                # Check if worker is still running
                if task.done():
                    logger.info(f"⚠️  Worker completed for {session_id[:8]}")
                    await websocket.send_json({
                        "type": "worker_completed",
                        "message": "Agent has finished"
                    })
                    break
            
            except WebSocketDisconnect:
                logger.info(f"🔌 Client disconnected: {session_id[:8]}")
                break
            except Exception as e:
                logger.error(f"❌ Error in message loop: {e}")
                await websocket.send_json({
                    "type": "error",
                    "error": str(e)
                })
                break
    
    except Exception as e:
        logger.error(f"❌ WebSocket error for {session_id[:8]}: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
            await websocket.close(code=1011, reason=str(e))
        except:
            pass
    
    finally:
        # Cleanup
        logger.info(f"🧹 Cleaning up session: {session_id[:8]}")
        
        # Cancel background task
        task = BACKGROUND_TASKS.get(session_id)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Remove from tracking
        ACTIVE_SESSIONS.pop(session_id, None)
        BACKGROUND_TASKS.pop(session_id, None)
        
        logger.info(f"✅ Session cleanup complete: {session_id[:8]} (Active: {len(ACTIVE_SESSIONS)})")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=9020, reload=True)