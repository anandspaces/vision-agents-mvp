import logging
from dotenv import load_dotenv

from vision_agents.core import User, Agent
from vision_agents.plugins import getstream, ultralytics, gemini

logger = logging.getLogger(__name__)
load_dotenv()


async def create_golf_agent(**kwargs) -> Agent:
    """
    Create an AI golf coach agent with pose detection.
    
    Features:
    - Real-time video processing with Gemini
    - YOLO pose detection for swing analysis
    - Streaming via GetStream Edge
    
    Returns:
        Agent: Configured golf coach agent
    """
    agent = Agent(
        edge=getstream.Edge(),  # Use stream for edge video transport
        agent_user=User(name="AI Golf Coach", id="golf-coach-agent"),
        instructions=(
            "You are an expert AI golf coach. Your role is to:\n"
            "1. Analyze the user's golf swing using pose detection\n"
            "2. Provide constructive, specific feedback on form and technique\n"
            "3. Focus on key elements: grip, stance, backswing, downswing, follow-through\n"
            "4. Keep feedback clear, actionable, and encouraging\n"
            "5. Use simple language without excessive technical jargon\n"
            "When analyzing, mention specific body positions and movements you observe."
        ),
        llm=gemini.Realtime(fps=3),  # Share video with Gemini at 3 FPS
        processors=[
            ultralytics.YOLOPoseProcessor(model_path="yolo11n-pose.pt")
        ],  # Real-time pose detection with YOLO
    )
    return agent


async def join_golf_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """
    Join a golf coaching call and start the session.
    
    Args:
        agent: The golf coach agent
        call_type: Type of call (e.g., "default", "video", etc.)
        call_id: Unique identifier for the call
        **kwargs: Additional arguments
    """
    try:
        # Ensure the agent user is created
        await agent.create_user()
        
        # Create a call
        call = await agent.create_call(call_type, call_id)
        
        logger.info(f"🏌️ Starting Golf Coach Agent for call: {call_id}")
        
        # Join the call
        with await agent.join(call):
            logger.info(f"Golf coach joined call: {call_id}")
            
            # Initial greeting and instructions
            await agent.llm.simple_response(
                "Hi! I'm your AI golf coach. "
                "Show me your golf swing when you're ready, and I'll analyze your form "
                "and provide helpful feedback to improve your technique."
            )
            
            # Run until the call ends
            await agent.finish()
            
        logger.info(f"Golf coaching call ended: {call_id}")
        
    except Exception as e:
        logger.error(f"Error in golf coaching call {call_id}: {e}")
        raise


# For standalone execution
if __name__ == "__main__":
    from vision_agents.core.agents import AgentLauncher
    from vision_agents.core import cli
    
    cli(AgentLauncher(create_agent=create_golf_agent, join_call=join_golf_call))