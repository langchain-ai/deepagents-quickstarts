"""Agent entry point for LangGraph deployment.

This file is referenced by langgraph.json and provides the graph for deployment.
It uses absolute imports to avoid relative import issues when loaded by LangGraph.

The for_deployment=True flag ensures we don't pass store/checkpointer to the graph,
allowing LangGraph platform to provide its own persistence infrastructure.
"""

from personal_assistant import create_email_assistant

# Export the graph for deployment
# Use for_deployment=True to let LangGraph platform provide store/checkpointer
graph = create_email_assistant(for_deployment=True)
