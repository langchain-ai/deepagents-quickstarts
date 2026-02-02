"""Middleware for updating memory based on interrupt responses."""

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.store.base import BaseStore

from ..prompts import MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT
from ..utils import aupdate_memory, update_memory


class PostInterruptMemoryMiddleware(AgentMiddleware):
    """Middleware for updating memory when user rejects ANY tool call.

    This middleware detects when tool calls are rejected (never executed) and
    updates the user profile to learn from the rejection feedback.

    Memory updates:
    - reject write_email: Updates user_profile (triage and/or response style)
    - reject schedule_meeting: Updates user_profile (triage and/or scheduling)
    - reject Question: Updates user_profile (triage and/or response style)
    - approve: No memory update

    Args:
        store: LangGraph store for persistent memory
    """

    # Import state schema at the top of the file
    from ..schemas import EmailAssistantState
    state_schema = EmailAssistantState

    def __init__(self, store: BaseStore):
        self.store = store
        # Track ALL interrupt tools (not just specific ones)
        self.interrupt_tools = {"write_email", "schedule_meeting", "Question"}
        # Track which tool calls we've already processed (to avoid duplicates)
        self._processed_tool_calls = set()

    def before_model(self, state, runtime) -> dict | None:
        """Check for rejected tool calls before next model generation.

        After a rejection, the built-in HITL middleware adds a ToolMessage with
        status="error". We check for these BEFORE the next model call to update memory.
        """
        messages = state.get("messages", [])

        # Look for ToolMessages with status="error" (rejections)
        for message in reversed(messages):
            # Check if this is a ToolMessage with error status (indicates rejection)
            if (hasattr(message, "status") and
                message.status == "error" and
                hasattr(message, "tool_call_id")):

                tool_call_id = message.tool_call_id
                tool_name = message.name
                rejection_message = message.content

                # Check if this is a tool we care about and haven't processed yet
                if tool_name in self.interrupt_tools and tool_call_id not in self._processed_tool_calls:
                    # Find the original tool call args from the AIMessage
                    original_args = {}
                    for msg in reversed(messages):
                        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
                            for tc in msg.tool_calls:
                                if tc.get("id") == tool_call_id:
                                    original_args = dict(tc.get("args", {}))
                                    break
                            if original_args:
                                break

                    # Update memory with rejection message
                    self._update_memory_for_reject(tool_name, original_args, rejection_message, state, runtime)
                    # Mark as processed
                    self._processed_tool_calls.add(tool_call_id)

        return None

    def _get_store(self, runtime=None) -> BaseStore:
        """Get store from runtime if available, otherwise use instance store.

        In deployment, LangGraph platform provides store via runtime.
        In local testing, we use the store passed during initialization.

        Args:
            runtime: Optional runtime object with store attribute

        Returns:
            BaseStore instance
        """
        if runtime and hasattr(runtime, "store"):
            return runtime.store
        return self.store

    def after_tool(self, state, runtime) -> dict | None:
        """Check for rejected tool calls (fallback hook).

        Note: Rejections are primarily detected in before_model hook.
        This hook serves as a fallback in case after_tool runs after rejections.
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        # Look for ToolMessages with status="error" (rejections)
        for message in reversed(messages):
            # Check if this is a ToolMessage with error status (indicates rejection)
            if (hasattr(message, "status") and
                message.status == "error" and
                hasattr(message, "tool_call_id")):

                tool_call_id = message.tool_call_id
                tool_name = message.name
                rejection_message = message.content

                # Check if this is a tool we care about and haven't processed yet
                if tool_name in self.interrupt_tools and tool_call_id not in self._processed_tool_calls:
                    # Find the original tool call args from the AIMessage
                    original_args = {}
                    for msg in reversed(messages):
                        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
                            for tc in msg.tool_calls:
                                if tc.get("id") == tool_call_id:
                                    original_args = dict(tc.get("args", {}))
                                    break
                            if original_args:
                                break

                    # Update memory with rejection message
                    self._update_memory_for_reject(tool_name, original_args, rejection_message, state, runtime)
                    # Mark as processed
                    self._processed_tool_calls.add(tool_call_id)

        return None

    def _update_memory_for_reject(
        self, tool_name: str, original_args: dict, rejection_message: str, state, runtime
    ):
        """Update user profile when user rejects a tool call.

        Args:
            tool_name: Name of the tool that was rejected
            original_args: Original tool arguments that were rejected
            rejection_message: User's optional feedback explaining why they rejected
            state: Agent state containing messages
            runtime: Runtime context
        """
        namespace = ("email_assistant", "user_profile")

        # Create feedback based on tool type
        if tool_name == "write_email":
            feedback = (
                "The user rejected the email draft, meaning they did not want to respond to this email. "
                "Update the User Profile to refine triage criteria (which emails warrant responses) and/or "
                "response style preferences based on what was wrong with the draft."
            )
        elif tool_name == "schedule_meeting":
            feedback = (
                "The user rejected the meeting invitation, meaning they did not want to schedule this meeting. "
                "Update the User Profile to refine triage criteria (which meeting requests warrant scheduling) and/or "
                "meeting scheduling preferences based on what was wrong with the invitation."
            )
        elif tool_name == "Question":
            feedback = (
                "The user rejected the clarification question, meaning they did not want to ask this question. "
                "Update the User Profile to refine when questions should be asked or how they should be phrased."
            )
        else:
            # Generic fallback for any future interrupt tools
            feedback = (
                f"The user rejected the {tool_name} tool call. "
                "Update the User Profile to learn from this feedback and avoid similar rejections in the future."
            )

        # Get conversation context for memory update from state
        messages = state.get("messages", [])

        # Format args as JSON for context
        import json
        args_str = json.dumps(original_args, indent=2)

        # Include user's rejection message if provided
        rejection_context = f"\n\nUser's rejection reason: {rejection_message}" if rejection_message else ""

        messages_for_update = messages + [
            {
                "role": "user",
                "content": f"{feedback}\n\nRejected tool call: {tool_name}\nArguments: {args_str}{rejection_context}\n\nFollow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}",
            }
        ]

        store = self._get_store(runtime)
        update_memory(store, namespace, messages_for_update)

    # Async versions

    async def abefore_model(self, state, runtime) -> dict | None:
        """Async version - check for rejected tool calls before next model generation."""
        messages = state.get("messages", [])

        # Look for ToolMessages with status="error" (rejections)
        for message in reversed(messages):
            # Check if this is a ToolMessage with error status (indicates rejection)
            if (hasattr(message, "status") and
                message.status == "error" and
                hasattr(message, "tool_call_id")):

                tool_call_id = message.tool_call_id
                tool_name = message.name
                rejection_message = message.content

                # Check if this is a tool we care about and haven't processed yet
                if tool_name in self.interrupt_tools and tool_call_id not in self._processed_tool_calls:
                    # Find the original tool call args from the AIMessage
                    original_args = {}
                    for msg in reversed(messages):
                        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
                            for tc in msg.tool_calls:
                                if tc.get("id") == tool_call_id:
                                    original_args = dict(tc.get("args", {}))
                                    break
                            if original_args:
                                break

                    # Update memory with rejection message
                    await self._aupdate_memory_for_reject(tool_name, original_args, rejection_message, state, runtime)
                    # Mark as processed
                    self._processed_tool_calls.add(tool_call_id)

        return None

    async def _aupdate_memory_for_reject(
        self, tool_name: str, original_args: dict, rejection_message: str, state, runtime
    ):
        """Async version of _update_memory_for_reject.

        Args:
            tool_name: Name of the tool that was rejected
            original_args: Original tool arguments that were rejected
            rejection_message: User's optional feedback explaining why they rejected
            state: Agent state containing messages
            runtime: Runtime context
        """
        namespace = ("email_assistant", "user_profile")

        # Create feedback based on tool type
        if tool_name == "write_email":
            feedback = (
                "The user rejected the email draft, meaning they did not want to respond to this email. "
                "Update the User Profile to refine triage criteria (which emails warrant responses) and/or "
                "response style preferences based on what was wrong with the draft."
            )
        elif tool_name == "schedule_meeting":
            feedback = (
                "The user rejected the meeting invitation, meaning they did not want to schedule this meeting. "
                "Update the User Profile to refine triage criteria (which meeting requests warrant scheduling) and/or "
                "meeting scheduling preferences based on what was wrong with the invitation."
            )
        elif tool_name == "Question":
            feedback = (
                "The user rejected the clarification question, meaning they did not want to ask this question. "
                "Update the User Profile to refine when questions should be asked or how they should be phrased."
            )
        else:
            # Generic fallback for any future interrupt tools
            feedback = (
                f"The user rejected the {tool_name} tool call. "
                "Update the User Profile to learn from this feedback and avoid similar rejections in the future."
            )

        # Get conversation context for memory update from state
        messages = state.get("messages", [])

        # Format args as JSON for context
        import json
        args_str = json.dumps(original_args, indent=2)

        # Include user's rejection message if provided
        rejection_context = f"\n\nUser's rejection reason: {rejection_message}" if rejection_message else ""

        messages_for_update = messages + [
            {
                "role": "user",
                "content": f"{feedback}\n\nRejected tool call: {tool_name}\nArguments: {args_str}{rejection_context}\n\nFollow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}",
            }
        ]

        store = self._get_store(runtime)
        await aupdate_memory(store, namespace, messages_for_update)
