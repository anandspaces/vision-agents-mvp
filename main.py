import logging

from dotenv import load_dotenv

from vision_agents.core import User, Agent, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import getstream, gemini

logger = logging.getLogger(__name__)

load_dotenv()

async def create_agent(**kwargs) -> Agent:
    """Create the agent with Gemini Realtime."""
    # Initialize Gemini with realtime capabilities
    llm = gemini.Realtime()

    # create an agent to run with Stream's edge, Gemini llm
    agent = Agent(
        edge=getstream.Edge(),  # low latency edge. clients for React, iOS, Android, RN, Flutter etc.
        agent_user=User(name="My happy AI friend", id="agent"),  # the user object for the agent (name, image etc)
        instructions="You're a voice AI assistant. Keep responses short and conversational. Don't use special characters or formatting. Be friendly and helpful.",
        llm=llm,
    )
    return agent

async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    # Ensure the agent user is created
    await agent.create_user()
    # Create a call
    call = await agent.create_call(call_type, call_id)

    logger.info("🤖 Starting Gemini Realtime Agent...")

    # Have the agent join the call/room
    with await agent.join(call):
        logger.info("Joining call")

        await agent.edge.open_demo(call)
        logger.info("LLM ready")

        # Example 1: standardized simple response
        await agent.llm.simple_response("chat with the user about them.")
        
        # run till the call ends
        await agent.finish()


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))