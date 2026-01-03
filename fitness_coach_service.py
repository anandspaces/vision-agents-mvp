import logging
from dotenv import load_dotenv

from vision_agents.core import User, Agent
from vision_agents.plugins import getstream, ultralytics, gemini

logger = logging.getLogger(__name__)
load_dotenv()


async def create_fitness_agent(**kwargs) -> Agent:
    """
    Create an AI fitness trainer agent with pose detection.
    
    Features:
    - Real-time video processing with Gemini
    - YOLO pose detection for form analysis
    - Exercise tracking and correction
    - Streaming via GetStream Edge
    
    Returns:
        Agent: Configured fitness trainer agent
    """
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="AI Fitness Trainer", id="fitness-agent"),
        instructions=(
            "You are an expert AI fitness trainer. Your role is to:\n"
            "1. Monitor exercise form using pose detection\n"
            "2. Provide real-time feedback on technique and posture\n"
            "3. Focus on proper alignment, range of motion, and movement quality\n"
            "4. Prevent injury by correcting poor form immediately\n"
            "5. Encourage proper breathing and controlled movements\n"
            "6. Count reps and maintain motivation\n"
            "7. Adapt guidance based on the specific exercise being performed\n"
            "Be supportive, energetic, and clear with your instructions."
        ),
        llm=gemini.Realtime(fps=3),
        processors=[
            ultralytics.YOLOPoseProcessor(model_path="yolo11n-pose.pt")
        ],
    )
    return agent


async def join_fitness_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """
    Join a fitness training call and start the session.
    
    Args:
        agent: The fitness trainer agent
        call_type: Type of call
        call_id: Unique identifier for the call
        **kwargs: Additional arguments
    """
    try:
        await agent.create_user()
        call = await agent.create_call(call_type, call_id)
        
        logger.info(f"💪 Starting Fitness Trainer Agent for call: {call_id}")
        
        with await agent.join(call):
            logger.info(f"Fitness trainer joined call: {call_id}")
            
            await agent.llm.simple_response(
                "Hey! I'm your AI fitness trainer. "
                "I'll watch your form and help you exercise safely and effectively. "
                "What exercise would you like to work on today?"
            )
            
            await agent.finish()
            
        logger.info(f"Fitness training call ended: {call_id}")
        
    except Exception as e:
        logger.error(f"Error in fitness training call {call_id}: {e}")
        raise


# For standalone execution
if __name__ == "__main__":
    from vision_agents.core.agents import AgentLauncher
    from vision_agents.core import cli
    
    cli(AgentLauncher(create_agent=create_fitness_agent, join_call=join_fitness_call))