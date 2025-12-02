import os
import asyncio

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.chat_models import init_chat_model

from developer_agent import DEVELOPER_SYSTEM_PROMPT, tavily_search, fetch_webpage_content

WORKSPACE_DIR = "agent_workspace"


# Model Claude 4.5 - can replace with whichever you like
model = init_chat_model(model="anthropic:claude-sonnet-4-5-20250929", temperature=0.0)


def initialize_backend():
    # Check if agent_workspace folder exists, create it if it doesn't
    if not os.path.exists(WORKSPACE_DIR):
        os.makedirs(WORKSPACE_DIR)
    # Create a FileSystem backend for our agent to do its work in
    return FilesystemBackend(
        root_dir=WORKSPACE_DIR,
        virtual_mode=True)


async def agent():
    backend = await asyncio.to_thread(initialize_backend)
    agent = create_deep_agent(
        model=model,
        backend=backend,
        system_prompt=DEVELOPER_SYSTEM_PROMPT,
        tools=[tavily_search, fetch_webpage_content],
    )
    return agent
