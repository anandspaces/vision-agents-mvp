import logging
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Optional, List
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from golf_coach_service import create_golf_agent, join_golf_call
from general_coach_service import create_general_agent, join_general_call
from fitness_coach_service import create_fitness_agent, join_fitness_call
from yoga_coach_service import create_yoga_agent, join_yoga_call

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Store active sessions
active_sessions: Dict[str, Dict] = {}


# ==================== Pydantic Models ====================

class CallRequest(BaseModel):
    """Request model for creating a new agent session."""
    call_type: Optional[str] = Field(
        default="default",
        description="Type of call to create (e.g., 'default', 'video', 'audio')",
        example="default"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "call_type": "default"
            }
        }


class CallResponse(BaseModel):
    """Response model for successful session creation."""
    call_id: str = Field(..., description="Unique identifier for the call session")
    call_type: str = Field(..., description="Type of call created")
    call_url: str = Field(..., description="URL to join the call/session")
    agent_type: str = Field(..., description="Type of agent (golf, general, fitness, yoga)")
    message: str = Field(..., description="Success message")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp of session creation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "call_id": "550e8400-e29b-41d4-a716-446655440000",
                "call_type": "default",
                "call_url": "https://stream.example.com/call/550e8400",
                "agent_type": "golf",
                "message": "Golf coaching session created successfully",
                "created_at": "2024-01-03T10:30:00"
            }
        }


class SessionInfo(BaseModel):
    """Model for session information."""
    call_id: str = Field(..., description="Unique identifier for the call session")
    call_type: str = Field(..., description="Type of call")
    agent_type: str = Field(..., description="Type of agent")
    status: str = Field(default="active", description="Current status of the session")
    
    class Config:
        json_schema_extra = {
            "example": {
                "call_id": "550e8400-e29b-41d4-a716-446655440000",
                "call_type": "default",
                "agent_type": "golf",
                "status": "active"
            }
        }


class SessionListItem(BaseModel):
    """Model for session list item."""
    call_id: str = Field(..., description="Unique identifier for the call session")
    agent_type: str = Field(..., description="Type of agent")
    call_type: str = Field(..., description="Type of call")


class SessionListResponse(BaseModel):
    """Response model for listing all sessions."""
    active_sessions: int = Field(..., description="Number of active sessions")
    sessions: List[SessionListItem] = Field(..., description="List of active sessions")
    
    class Config:
        json_schema_extra = {
            "example": {
                "active_sessions": 2,
                "sessions": [
                    {
                        "call_id": "550e8400-e29b-41d4-a716-446655440000",
                        "agent_type": "golf",
                        "call_type": "default"
                    },
                    {
                        "call_id": "660e8400-e29b-41d4-a716-446655440001",
                        "agent_type": "fitness",
                        "call_type": "default"
                    }
                ]
            }
        }


class DeleteSessionResponse(BaseModel):
    """Response model for session deletion."""
    message: str = Field(..., description="Confirmation message")
    call_id: str = Field(..., description="ID of the deleted session")
    agent_type: str = Field(..., description="Type of agent that was deleted")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Session ended successfully",
                "call_id": "550e8400-e29b-41d4-a716-446655440000",
                "agent_type": "golf"
            }
        }


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="API health status")
    message: str = Field(..., description="Status message")
    active_sessions: int = Field(..., description="Number of currently active sessions")
    endpoints: Dict[str, str] = Field(..., description="Available API endpoints")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "message": "Vision Agents API is running",
                "active_sessions": 3,
                "endpoints": {
                    "golf": "/golf",
                    "general": "/general",
                    "fitness": "/fitness",
                    "yoga": "/yoga"
                }
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response model."""
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Specific error code")
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Session not found",
                "error_code": "SESSION_NOT_FOUND"
            }
        }


# ==================== Lifespan Management ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown events."""
    # Startup
    logger.info("🚀 Starting Vision Agents API Server...")
    logger.info("✅ Server ready to accept requests")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Vision Agents API Server...")
    # Clean up any active sessions
    for session_id in list(active_sessions.keys()):
        logger.info(f"Cleaning up session: {session_id}")
        active_sessions.pop(session_id, None)
    logger.info("✅ Cleanup complete")


# ==================== FastAPI App Configuration ====================

app = FastAPI(
    title="Vision Agents API",
    description="API for creating AI agents with video capabilities for coaching and training",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS - allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


# ==================== API Endpoints ====================

@app.get(
    "/",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Check if the API is running and get basic information"
)
async def root():
    """Health check endpoint."""
    return HealthCheckResponse(
        status="healthy",
        message="Vision Agents API is running",
        active_sessions=len(active_sessions),
        endpoints={
            "golf": "/golf",
            "general": "/general",
            "fitness": "/fitness",
            "yoga": "/yoga"
        }
    )


@app.post(
    "/golf",
    response_model=CallResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Golf Coaching Session",
    description="Create a new golf coaching session with pose detection for swing analysis",
    responses={
        201: {"description": "Session created successfully"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def create_golf_session(request: CallRequest = CallRequest()):
    """Create a golf coaching session with pose detection."""
    try:
        call_id = str(uuid.uuid4())
        call_type = request.call_type
        
        logger.info(f"Creating golf session: {call_id}")
        
        # Create agent
        agent = await create_golf_agent()
        
        # Get call URL
        await agent.create_user()
        call = await agent.create_call(call_type, call_id)
        call_url = await agent.edge.get_demo_url(call)
        
        # Store session info
        active_sessions[call_id] = {
            "agent": agent,
            "call_type": call_type,
            "agent_type": "golf",
            "created_at": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Golf session created: {call_url}")
        
        return CallResponse(
            call_id=call_id,
            call_type=call_type,
            call_url=call_url,
            agent_type="golf",
            message="Golf coaching session created successfully"
        )
        
    except Exception as e:
        logger.error(f"Error creating golf session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post(
    "/general",
    response_model=CallResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create General Chat Session",
    description="Create a new general conversational AI session without pose detection",
    responses={
        201: {"description": "Session created successfully"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def create_general_session(request: CallRequest = CallRequest()):
    """Create a general conversational AI session."""
    try:
        call_id = str(uuid.uuid4())
        call_type = request.call_type
        
        logger.info(f"Creating general session: {call_id}")
        
        # Create agent
        agent = await create_general_agent()
        
        # Get call URL
        await agent.create_user()
        call = await agent.create_call(call_type, call_id)
        call_url = await agent.edge.get_demo_url(call)
        
        # Store session info
        active_sessions[call_id] = {
            "agent": agent,
            "call_type": call_type,
            "agent_type": "general",
            "created_at": datetime.now().isoformat()
        }
        
        logger.info(f"✅ General session created: {call_url}")
        
        return CallResponse(
            call_id=call_id,
            call_type=call_type,
            call_url=call_url,
            agent_type="general",
            message="General chat session created successfully"
        )
        
    except Exception as e:
        logger.error(f"Error creating general session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post(
    "/fitness",
    response_model=CallResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Fitness Training Session",
    description="Create a new fitness training session with pose detection for form analysis",
    responses={
        201: {"description": "Session created successfully"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def create_fitness_session(request: CallRequest = CallRequest()):
    """Create a fitness training session with pose detection."""
    try:
        call_id = str(uuid.uuid4())
        call_type = request.call_type
        
        logger.info(f"Creating fitness session: {call_id}")
        
        # Create agent
        agent = await create_fitness_agent()
        
        # Get call URL
        await agent.create_user()
        call = await agent.create_call(call_type, call_id)
        call_url = await agent.edge.get_demo_url(call)
        
        # Store session info
        active_sessions[call_id] = {
            "agent": agent,
            "call_type": call_type,
            "agent_type": "fitness",
            "created_at": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Fitness session created: {call_url}")
        
        return CallResponse(
            call_id=call_id,
            call_type=call_type,
            call_url=call_url,
            agent_type="fitness",
            message="Fitness training session created successfully"
        )
        
    except Exception as e:
        logger.error(f"Error creating fitness session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post(
    "/yoga",
    response_model=CallResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Yoga Instruction Session",
    description="Create a new yoga instruction session with pose detection for alignment analysis",
    responses={
        201: {"description": "Session created successfully"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def create_yoga_session(request: CallRequest = CallRequest()):
    """Create a yoga instruction session with pose detection."""
    try:
        call_id = str(uuid.uuid4())
        call_type = request.call_type
        
        logger.info(f"Creating yoga session: {call_id}")
        
        # Create agent
        agent = await create_yoga_agent()
        
        # Get call URL
        await agent.create_user()
        call = await agent.create_call(call_type, call_id)
        call_url = await agent.edge.get_demo_url(call)
        
        # Store session info
        active_sessions[call_id] = {
            "agent": agent,
            "call_type": call_type,
            "agent_type": "yoga",
            "created_at": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Yoga session created: {call_url}")
        
        return CallResponse(
            call_id=call_id,
            call_type=call_type,
            call_url=call_url,
            agent_type="yoga",
            message="Yoga instruction session created successfully"
        )
        
    except Exception as e:
        logger.error(f"Error creating yoga session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get(
    "/session/{call_id}",
    response_model=SessionInfo,
    summary="Get Session Information",
    description="Retrieve information about a specific active session",
    responses={
        200: {"description": "Session information retrieved successfully"},
        404: {"model": ErrorResponse, "description": "Session not found"}
    }
)
async def get_session_info(call_id: str):
    """Get information about an active session."""
    if call_id not in active_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    session = active_sessions[call_id]
    return SessionInfo(
        call_id=call_id,
        call_type=session["call_type"],
        agent_type=session["agent_type"],
        status="active"
    )


@app.get(
    "/sessions",
    response_model=SessionListResponse,
    summary="List All Sessions",
    description="Get a list of all currently active sessions"
)
async def list_sessions():
    """List all active sessions."""
    sessions_list = [
        SessionListItem(
            call_id=call_id,
            agent_type=info["agent_type"],
            call_type=info["call_type"]
        )
        for call_id, info in active_sessions.items()
    ]
    
    return SessionListResponse(
        active_sessions=len(active_sessions),
        sessions=sessions_list
    )


@app.delete(
    "/session/{call_id}",
    response_model=DeleteSessionResponse,
    summary="End Session",
    description="End an active session and clean up resources",
    responses={
        200: {"description": "Session ended successfully"},
        404: {"model": ErrorResponse, "description": "Session not found"}
    }
)
async def end_session(call_id: str):
    """End an active session."""
    if call_id not in active_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    session = active_sessions.pop(call_id)
    logger.info(f"Session ended: {call_id}")
    
    return DeleteSessionResponse(
        message="Session ended successfully",
        call_id=call_id,
        agent_type=session["agent_type"]
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9020, reload=True)