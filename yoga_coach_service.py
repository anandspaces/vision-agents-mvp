import logging
from dotenv import load_dotenv

from vision_agents.core import User, Agent
from vision_agents.plugins import getstream, ultralytics, gemini

logger = logging.getLogger(__name__)
load_dotenv()


async def create_yoga_agent(**kwargs) -> Agent:
    """
    Create an AI yoga instructor agent with pose detection.
    
    Features:
    - Real-time video processing with Gemini
    - YOLO pose detection for alignment analysis
    - Gentle guidance and corrections
    - Streaming via GetStream Edge
    
    Returns:
        Agent: Configured yoga instructor agent
    """
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="AI Yoga Instructor", id="yoga-agent"),
        instructions=(
            "You are a compassionate AI yoga instructor. Your role is to:\n"
            "1. Guide users through yoga poses with mindfulness\n"
            "2. Monitor alignment and posture using pose detection\n"
            "3. Provide gentle, encouraging corrections\n"
            "4. Emphasize breath awareness and mindful movement\n"
            "5. Ensure poses are safe and accessible for all levels\n"
            "6. Offer modifications for different flexibility levels\n"
            "7. Create a calming, supportive atmosphere\n"
            "8. Focus on the journey, not perfection\n"
            "Use soothing language and remind users to listen to their bodies."
        ),
        llm=gemini.Realtime(fps=3),
        processors=[
            ultralytics.YOLOPoseProcessor(model_path="yolo11n-pose.pt")
        ],
    )
    return agent


async def join_yoga_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """
    Join a yoga instruction call and start the session.
    
    Args:
        agent: The yoga instructor agent
        call_type: Type of call
        call_id: Unique identifier for the call
        **kwargs: Additional arguments
    """
    try:
        await agent.create_user()
        call = await agent.create_call(call_type, call_id)
        
        logger.info(f"🧘 Starting Yoga Instructor Agent for call: {call_id}")
        
        with await agent.join(call):
            logger.info(f"Yoga instructor joined call: {call_id}")
            
            await agent.llm.simple_response(
                "Namaste. I'm your AI yoga instructor. "
                "I'll guide you through your practice and help you find proper alignment. "
                "Take a deep breath, and let me know what pose you'd like to work on, "
                "or if you'd like me to guide you through a sequence."
            )
            
            await agent.finish()
            
        logger.info(f"Yoga instruction call ended: {call_id}")
        
    except Exception as e:
        logger.error(f"Error in yoga instruction call {call_id}: {e}")
        raise


# For standalone execution
if __name__ == "__main__":
    from vision_agents.core.agents import AgentLauncher
    from vision_agents.core import cli
    
    cli(AgentLauncher(create_agent=create_yoga_agent, join_call=join_yoga_call))