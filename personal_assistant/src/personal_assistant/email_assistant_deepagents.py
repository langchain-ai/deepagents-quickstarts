"""Email assistant using deepagents library with custom HITL middleware.

This is the migration of email_assistant_hitl_memory.py to use the deepagents library's
create_deep_agent() pattern instead of manual graph construction. All functionality is
preserved including HITL logic, memory system, and custom tools.

The agent now handles triage directly through a tool call instead of a separate routing step.

Usage:
    python -m examples.personal_assistant.email_assistant_deepagents
"""

from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from deepagents import create_deep_agent
from deepagents.backends import StoreBackend

from .middleware import MemoryInjectionMiddleware, PostInterruptMemoryMiddleware, GenUIMiddleware
from .schemas import EmailAssistantState
from .tools import get_tools
from .utils import format_email_markdown, parse_email, get_memory
from .prompts import agent_system_prompt_hitl_memory, default_user_profile

def create_email_assistant(for_deployment=False):
    """Create and configure the email assistant agent.

    Args:
        for_deployment: If True, don't pass store/checkpointer (for LangGraph deployment).
                       If False, create InMemoryStore and MemorySaver for local testing.

    Returns:
        CompiledStateGraph: Configured email assistant agent
    """
    # Initialize model
    model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)

    # Get tools - now includes triage_email
    tools = get_tools(
        [
            "triage_email",
            "write_email",
            "schedule_meeting",
            "check_calendar_availability",
            "Question",
            "Done",
        ]
    )

    # Initialize persistence based on deployment mode
    if for_deployment:
        # In deployment, LangGraph platform provides store and checkpointer
        # We need to pass a store to middleware, but it will be overridden by platform
        # Use a placeholder that the middleware can work with during initialization
        store = InMemoryStore()  # Placeholder - will be overridden by platform
        store_kwarg = {}  # Don't pass store to create_deep_agent
        checkpointer_kwarg = {}  # Don't pass checkpointer to create_deep_agent
    else:
        # Local testing mode - create and use our own store and checkpointer
        store = InMemoryStore()
        checkpointer = MemorySaver()
        store_kwarg = {"store": store}
        checkpointer_kwarg = {"checkpointer": checkpointer}

    # Define interrupt configurations with plain text descriptions
    interrupt_on_config = {
        "write_email": {
            "allowed_decisions": ["approve", "reject"],
            "description": "I've drafted an email response. Please review the content, recipients, and subject line below. Approve to send as-is, or Reject to cancel and end the workflow."
        },
        "schedule_meeting": {
            "allowed_decisions": ["approve", "reject"],
            "description": "I've prepared a calendar invitation. Please review the meeting details below. Approve to send the invite as-is, or Reject to cancel and end the workflow."
        },
        "Question": {
            "allowed_decisions": ["approve", "reject"],
            "description": "I need clarification before proceeding. Please review the question below and provide your response, or Reject to skip this action and end the workflow."
        }
    }

    # Create middleware instances
    memory_injection = MemoryInjectionMiddleware(store=store)
    post_interrupt_memory = PostInterruptMemoryMiddleware(store=store)
    genui = GenUIMiddleware(tool_to_genui_map={
        "write_email": {"component_name": "write_email"},
        "schedule_meeting": {"component_name": "schedule_meeting"},
        "check_calendar_availability": {"component_name": "check_calendar_availability"},
        "Question": {"component_name": "question"},
    })

    # Build system prompt with default user profile
    # Note: Memory-based profile can be accessed via the store in middleware
    tools_prompt = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
    system_prompt = agent_system_prompt_hitl_memory.format(
        tools_prompt=tools_prompt,
        user_profile=default_user_profile,
    )

    # Create agent with deepagents library
    agent = create_deep_agent(
        model=model,
        tools=tools,
        middleware=[memory_injection, post_interrupt_memory, genui],
        backend=lambda rt: StoreBackend(rt),
        context_schema=EmailAssistantState,
        system_prompt=system_prompt,
        interrupt_on=interrupt_on_config,  # NEW: Built-in interrupt handling
        **store_kwarg,
        **checkpointer_kwarg,
    )

    return agent


def main():
    """Example usage of the email assistant."""
    # Create agent
    agent = create_email_assistant()

    # Example email input
    email_input = {
        "author": "jane@example.com",
        "to": "lance@langchain.dev",
        "subject": "Quick question about next week",
        "email_thread": "Hi Lance,\n\nCan we meet next Tuesday at 2pm to discuss the project roadmap?\n\nBest,\nJane",
    }

    # Format email for message
    author, to, subject, email_thread = parse_email(email_input)
    email_markdown = format_email_markdown(subject, author, to, email_thread)

    # Configure thread
    config = {"configurable": {"thread_id": "test-thread-1"}}

    # Invoke agent
    print("=" * 80)
    print("EMAIL ASSISTANT EXAMPLE")
    print("=" * 80)
    print("\nProcessing email:")
    print(email_markdown)
    print("=" * 80)

    # Agent now accepts the email as a simple message string
    result = agent.invoke(
        {"messages": [{"role": "user", "content": email_markdown}]},
        config=config,
    )

    print("\nAgent result:")
    print(result)
    print("=" * 80)


if __name__ == "__main__":
    main()
