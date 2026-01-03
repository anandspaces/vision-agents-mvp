import logging
from dotenv import load_dotenv

from vision_agents.core import User, Agent
from vision_agents.plugins import getstream, gemini

logger = logging.getLogger(__name__)
load_dotenv()


async def create_general_agent(**kwargs) -> Agent:
    """
    Create a general conversational AI agent.
    
    Features:
    - Friendly conversational interface
    - No pose detection (lightweight)
    - Real-time voice interaction
    - Streaming via GetStream Edge
    
    Returns:
        Agent: Configured general chat agent
    """
    agent = Agent(
        edge=getstream.Edge(),  # Low latency edge for real-time communication
        agent_user=User(name="AI Assistant", id="general-agent"),
        instructions=(
            "You're a friendly voice AI assistant. Your role is to:\n"
            "1. Have natural, engaging conversations with users\n"
            "2. Keep responses short and conversational\n"
            "3. Don't use special characters or complex formatting\n"
            "4. Be helpful, empathetic, and supportive\n"
            "5. Ask follow-up questions to better understand user needs\n"
            "6. Maintain a warm, approachable tone\n"
            "Remember: You're having a voice conversation, so keep things natural and flowing."
        ),
        llm=gemini.Realtime(),  # Gemini realtime without video processing
    )
    return agent


async def join_general_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """
    Join a general chat call and start the session.
    
    Args:
        agent: The general chat agent
        call_type: Type of call (e.g., "default", "audio", etc.)
        call_id: Unique identifier for the call
        **kwargs: Additional arguments
    """
    try:
        # Ensure the agent user is created
        await agent.create_user()
        
        # Create a call
        call = await agent.create_call(call_type, call_id)
        
        logger.info(f"💬 Starting General Chat Agent for call: {call_id}")
        
        # Join the call
        with await agent.join(call):
            logger.info(f"General assistant joined call: {call_id}")
            
            # Initial greeting
            await agent.llm.simple_response(
                "Hey there! I'm your AI assistant. "
                "How can I help you today? Feel free to chat about anything on your mind."
            )
            
            # Run until the call ends
            await agent.finish()
            
        logger.info(f"General chat call ended: {call_id}")
        
    except Exception as e:
        logger.error(f"Error in general chat call {call_id}: {e}")
        raise


# For standalone execution
if __name__ == "__main__":
    from vision_agents.core.agents import AgentLauncher
    from vision_agents.core import cli
    
    cli(AgentLauncher(create_agent=create_general_agent, join_call=join_general_call))