"""GenUI middleware for email assistant tool visualization."""

from typing import Annotated, Any, Sequence
from typing_extensions import TypedDict

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langgraph.graph.ui import AnyUIMessage, ui_message_reducer, push_ui_message
from langgraph.runtime import Runtime


class UIState(AgentState):
    """State schema with UI message support."""
    ui: Annotated[Sequence[AnyUIMessage], ui_message_reducer]


class ToolGenUI(TypedDict):
    """Configuration for tool UI component mapping."""
    component_name: str


class GenUIMiddleware(AgentMiddleware):
    """Middleware to push UI messages for tool calls.

    This middleware runs after the model generates tool calls and pushes
    UI messages for configured tools. This enables custom UI components
    to render tool calls in the interface.

    Args:
        tool_to_genui_map: Dict mapping tool names to UI component configurations
    """

    state_schema = UIState

    def __init__(self, tool_to_genui_map: dict[str, ToolGenUI]):
        self.tool_to_genui_map = tool_to_genui_map

    def after_model(self, state: UIState, runtime: Runtime) -> dict[str, Any] | None:
        """Push UI messages for tool calls after model generation.

        Args:
            state: Agent state with messages
            runtime: Runtime context

        Returns:
            None (UI messages are pushed via side effect)
        """
        messages = state.get("messages", [])
        if not messages:
            return

        last_message = messages[-1]
        if last_message.type != "ai":
            return

        if last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                if tool_call["name"] in self.tool_to_genui_map:
                    component_name = self.tool_to_genui_map[tool_call["name"]]["component_name"]
                    push_ui_message(
                        component_name,
                        {},
                        metadata={
                            "tool_call_id": tool_call["id"]
                        },
                        message=last_message
                    )
