"""Custom HITL middleware for email assistant with memory updates."""

from typing import Callable

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.constants import END
from langgraph.store.base import BaseStore
from langgraph.types import Command, interrupt

from ..prompts import (
    MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT,
    agent_system_prompt_hitl_memory,
    default_background,
    default_cal_preferences,
    default_response_preferences,
    default_triage_instructions,
)
from ..tools.default.prompt_templates import HITL_MEMORY_TOOLS_PROMPT
from ..utils import format_email_markdown, format_for_display, get_memory, parse_email, update_memory, aget_memory, aupdate_memory


class EmailAssistantHITLMiddleware(AgentMiddleware):
    """Custom HITL middleware with memory updates and tool filtering.

    This middleware provides sophisticated human-in-the-loop functionality for the
    email assistant, including:
    - Tool filtering (only interrupts for specific tools)
    - Custom display formatting with email context
    - Per-tool action configurations
    - Four response types: accept, edit, ignore, response
    - Automatic memory updates based on user feedback

    Args:
        store: LangGraph store for persistent memory
        interrupt_on: Dict mapping tool names to whether they should trigger interrupts
        email_input_key: State key containing email context (default: "email_input")
    """

    def __init__(
        self,
        store: BaseStore,
        interrupt_on: dict[str, bool],
        email_input_key: str = "email_input",
    ):
        self.store = store
        self.interrupt_on = interrupt_on
        self.email_input_key = email_input_key
        self.tools_by_name = {}

    def _get_store(self, runtime=None):
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

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject memory into system prompt before LLM call.

        Fetches memory from the three namespaces (triage_preferences, response_preferences,
        cal_preferences) and injects them into the system prompt.
        """
        # Get store (from runtime in deployment, or from self in local testing)
        store = self._get_store(request.runtime if hasattr(request, "runtime") else None)

        # Fetch memory from store
        triage_prefs = get_memory(
            store,
            ("email_assistant", "triage_preferences"),
            default_triage_instructions,
        )
        response_prefs = get_memory(
            store,
            ("email_assistant", "response_preferences"),
            default_response_preferences,
        )
        cal_prefs = get_memory(
            store,
            ("email_assistant", "cal_preferences"),
            default_cal_preferences,
        )

        # Format system prompt with memory
        memory_prompt = agent_system_prompt_hitl_memory.format(
            tools_prompt=HITL_MEMORY_TOOLS_PROMPT,
            background=default_background,
            triage_instructions=triage_prefs,
            response_preferences=response_prefs,
            cal_preferences=cal_prefs,
        )

        # Append memory prompt to existing system prompt
        new_system_prompt = (
            request.system_prompt + "\n\n" + memory_prompt
            if request.system_prompt
            else memory_prompt
        )

        # Update request with new system prompt
        updated_request = request.override(system_prompt=new_system_prompt)

        return handler(updated_request)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Intercept tool calls for HITL filtering and memory updates.

        This is the core HITL logic that:
        1. Filters tools (only interrupts for configured tools)
        2. Formats display with email context
        3. Creates interrupt with per-tool config
        4. Handles user response (accept/edit/ignore/response)
        5. Updates memory based on feedback
        """
        tool_call = request.tool_call
        tool_name = tool_call["name"]
        tool = request.tool  # Get the tool directly from the request

        # Cache tool for later use
        if tool and tool_name not in self.tools_by_name:
            self.tools_by_name[tool_name] = tool

        # STEP 1: Filter - Execute non-HITL tools directly without interruption
        if tool_name not in self.interrupt_on or not self.interrupt_on[tool_name]:
            # Use handler to ensure proper middleware chaining
            return handler(request)

        # STEP 2: Format display - Get email context and format for display
        email_input = request.runtime.state.get(self.email_input_key)
        description = self._format_interrupt_display(email_input, tool_call)

        # STEP 3: Configure per-tool actions
        config = self._get_action_config(tool_name)

        # STEP 4: Create interrupt
        interrupt_request = {
            "action_request": {"action": tool_name, "args": tool_call["args"]},
            "config": config,
            "description": description,
        }

        # Call interrupt() to create the interrupt and wait for user decision
        # When creating a new interrupt: returns a list of responses [response]
        # When resuming from an interrupt: returns a dict mapping interrupt IDs to decisions
        result = interrupt([interrupt_request])
        print(f"DEBUG: interrupt() returned: {result}, type: {type(result)}")

        # Handle both new interrupt and resume cases
        if isinstance(result, list) and len(result) > 0:
            # New interrupt created, got user decision
            response = result[0]
        elif isinstance(result, dict):
            # Check if result is the decision itself (has "type" key) or a mapping of interrupt IDs
            if "type" in result:
                # Result is the decision dict directly (e.g., {"type": "accept"})
                response = result
            else:
                # Result is a mapping of interrupt IDs to decisions
                # Extract the decision (dict maps interrupt IDs to decisions)
                decisions = list(result.values())
                if decisions:
                    response = decisions[0]
                else:
                    # No decision found, execute tool with original args
                    tool = self.tools_by_name[tool_name]
                    observation = tool.invoke(tool_call["args"])
                    return ToolMessage(content=observation, tool_call_id=tool_call["id"])
        else:
            # Unexpected format, execute tool with original args
            tool = self.tools_by_name[tool_name]
            observation = tool.invoke(tool_call["args"])
            return ToolMessage(content=observation, tool_call_id=tool_call["id"])

        # STEP 5: Handle response type
        return self._handle_response(response, tool_call, request.runtime)

    def _format_interrupt_display(self, email_input: dict | None, tool_call: dict) -> str:
        """Format interrupt display with email context and tool details.

        Args:
            email_input: Email context from state
            tool_call: Tool call dict with name and args

        Returns:
            Formatted markdown string for interrupt display
        """
        # Get original email context if available
        if email_input:
            author, to, subject, email_thread = parse_email(email_input)
            email_markdown = format_email_markdown(subject, author, to, email_thread)
        else:
            email_markdown = ""

        # Format tool call for display
        tool_display = format_for_display(tool_call)

        return email_markdown + tool_display

    def _get_action_config(self, tool_name: str) -> dict:
        """Get per-tool action configuration.

        Args:
            tool_name: Name of the tool being called

        Returns:
            Config dict with allowed actions (allow_ignore, allow_respond, etc.)
        """
        if tool_name == "write_email":
            return {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": True,
                "allow_accept": True,
            }
        elif tool_name == "schedule_meeting":
            return {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": True,
                "allow_accept": True,
            }
        elif tool_name == "Question":
            return {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": False,  # Can't edit questions
                "allow_accept": False,  # Can't auto-accept questions
            }
        else:
            raise ValueError(f"Invalid tool call: {tool_name}")

    def _handle_response(
        self, response: dict, tool_call: dict, runtime: ToolRuntime
    ) -> ToolMessage | Command:
        """Route response to appropriate handler based on response type.

        Args:
            response: Response dict from interrupt
            tool_call: Tool call dict
            runtime: Tool runtime context

        Returns:
            ToolMessage or Command based on response type
        """
        response_type = response["type"]

        if response_type == "accept":
            return self._handle_accept(tool_call, runtime)
        elif response_type == "edit":
            return self._handle_edit(response, tool_call, runtime)
        elif response_type == "ignore":
            return self._handle_ignore(tool_call, runtime)
        elif response_type == "response":
            return self._handle_response_feedback(response, tool_call, runtime)
        else:
            raise ValueError(f"Invalid response type: {response_type}")

    def _handle_accept(self, tool_call: dict, _runtime: ToolRuntime) -> ToolMessage:
        """Handle accept response - execute tool with original args.

        Args:
            tool_call: Tool call dict
            _runtime: Tool runtime context (unused, kept for signature compatibility)

        Returns:
            ToolMessage with execution result
        """
        tool = self.tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        return ToolMessage(content=observation, tool_call_id=tool_call["id"])

    def _handle_edit(
        self, response: dict, tool_call: dict, runtime: ToolRuntime
    ) -> Command:
        """Handle edit response - execute with edited args, update AI message, update memory.

        Args:
            response: Response dict with edited args
            tool_call: Tool call dict
            runtime: Tool runtime context

        Returns:
            Command with updated AI message and tool result
        """
        tool = self.tools_by_name[tool_call["name"]]
        tool_name = tool_call["name"]
        initial_args = tool_call["args"]
        edited_args = response["args"]["args"]
        tool_call_id = tool_call["id"]

        # Execute tool with edited args
        observation = tool.invoke(edited_args)

        # Update AI message immutably with edited tool calls
        ai_message = runtime.state["messages"][-1]
        updated_tool_calls = [
            tc if tc["id"] != tool_call_id
            else {"type": "tool_call", "name": tool_name, "args": edited_args, "id": tool_call_id}
            for tc in ai_message.tool_calls
        ]
        updated_ai_message = ai_message.model_copy(update={"tool_calls": updated_tool_calls})

        # Update memory based on tool type
        self._update_memory_for_edit(tool_name, initial_args, edited_args, runtime)

        # Return Command with both updated AI message and tool message
        return Command(
            update={
                "messages": [
                    updated_ai_message,
                    ToolMessage(content=observation, tool_call_id=tool_call_id),
                ]
            }
        )

    def _handle_ignore(self, tool_call: dict, runtime: ToolRuntime) -> Command:
        """Handle ignore response - skip execution, goto END, update triage memory.

        Args:
            tool_call: Tool call dict
            runtime: Tool runtime context

        Returns:
            Command with goto END and feedback message
        """
        tool_name = tool_call["name"]
        tool_call_id = tool_call["id"]

        # Create feedback message
        if tool_name == "write_email":
            content = "User ignored this email draft. Ignore this email and end the workflow."
        elif tool_name == "schedule_meeting":
            content = "User ignored this calendar meeting draft. Ignore this email and end the workflow."
        elif tool_name == "Question":
            content = "User ignored this question. Ignore this email and end the workflow."
        else:
            raise ValueError(f"Invalid tool call: {tool_name}")

        # Update triage preferences to avoid future false positives
        self._update_triage_preferences_ignore(tool_name, runtime)

        # Return Command with goto END
        return Command(
            goto=END,
            update={
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]
            },
        )

    def _handle_response_feedback(
        self, response: dict, tool_call: dict, runtime: ToolRuntime
    ) -> ToolMessage:
        """Handle response feedback - add feedback to messages, update memory.

        Args:
            response: Response dict with user feedback
            tool_call: Tool call dict
            runtime: Tool runtime context

        Returns:
            ToolMessage with user feedback
        """
        tool_name = tool_call["name"]
        tool_call_id = tool_call["id"]
        user_feedback = response["args"]

        # Create feedback message based on tool type
        if tool_name == "write_email":
            content = f"User gave feedback, which can we incorporate into the email. Feedback: {user_feedback}"
        elif tool_name == "schedule_meeting":
            content = f"User gave feedback, which can we incorporate into the meeting request. Feedback: {user_feedback}"
        elif tool_name == "Question":
            content = f"User answered the question, which can we can use for any follow up actions. Feedback: {user_feedback}"
        else:
            raise ValueError(f"Invalid tool call: {tool_name}")

        # Update memory with feedback
        self._update_memory_for_feedback(tool_name, user_feedback, runtime)

        return ToolMessage(content=content, tool_call_id=tool_call_id)

    def _update_memory_for_edit(
        self, tool_name: str, initial_args: dict, edited_args: dict, _runtime: ToolRuntime
    ):
        """Update appropriate memory namespace when user edits.

        Args:
            tool_name: Name of the tool
            initial_args: Original tool arguments
            edited_args: Edited tool arguments
            _runtime: Tool runtime context (unused, kept for signature compatibility)
        """
        if tool_name == "write_email":
            namespace = ("email_assistant", "response_preferences")
            messages = [
                {
                    "role": "user",
                    "content": f"User edited the email response. Here is the initial email generated by the assistant: {initial_args}. Here is the edited email: {edited_args}. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.",
                }
            ]
        elif tool_name == "schedule_meeting":
            namespace = ("email_assistant", "cal_preferences")
            messages = [
                {
                    "role": "user",
                    "content": f"User edited the calendar invitation. Here is the initial calendar invitation generated by the assistant: {initial_args}. Here is the edited calendar invitation: {edited_args}. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.",
                }
            ]
        else:
            return  # No memory update for other tools

        store = self._get_store(_runtime)
        update_memory(store, namespace, messages)

    def _update_memory_for_feedback(
        self, tool_name: str, user_feedback: str, runtime: ToolRuntime
    ):
        """Update memory with user feedback.

        Args:
            tool_name: Name of the tool
            user_feedback: User's feedback text
            runtime: Tool runtime context
        """
        if tool_name == "write_email":
            namespace = ("email_assistant", "response_preferences")
        elif tool_name == "schedule_meeting":
            namespace = ("email_assistant", "cal_preferences")
        else:
            return  # No memory update for Question feedback

        messages = runtime.state["messages"] + [
            {
                "role": "user",
                "content": f"User gave feedback: {user_feedback}. Use this to update the preferences. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.",
            }
        ]

        store = self._get_store(runtime)
        update_memory(store, namespace, messages)

    def _update_triage_preferences_ignore(self, tool_name: str, runtime: ToolRuntime):
        """Update triage preferences when user ignores.

        Args:
            tool_name: Name of the tool
            runtime: Tool runtime context
        """
        namespace = ("email_assistant", "triage_preferences")

        if tool_name == "write_email":
            feedback = "The user ignored the email draft. That means they did not want to respond to the email. Update the triage preferences to ensure emails of this type are not classified as respond."
        elif tool_name == "schedule_meeting":
            feedback = "The user ignored the calendar meeting draft. That means they did not want to schedule a meeting for this email. Update the triage preferences to ensure emails of this type are not classified as respond."
        elif tool_name == "Question":
            feedback = "The user ignored the Question. That means they did not want to answer the question or deal with this email. Update the triage preferences to ensure emails of this type are not classified as respond."
        else:
            return

        messages = runtime.state["messages"] + [
            {
                "role": "user",
                "content": f"{feedback} Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.",
            }
        ]

        store = self._get_store(runtime)
        update_memory(store, namespace, messages)

    # Async versions of middleware methods for async invocation (astream, ainvoke)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Async version of wrap_model_call for async agent invocation.

        Identical logic to wrap_model_call but supports async context.
        """
        # Get store (from runtime in deployment, or from self in local testing)
        store = self._get_store(request.runtime if hasattr(request, "runtime") else None)

        # Fetch memory from store (using async methods)
        triage_prefs = await aget_memory(
            store,
            ("email_assistant", "triage_preferences"),
            default_response_preferences,
        )
        response_prefs = await aget_memory(
            store,
            ("email_assistant", "response_preferences"),
            default_response_preferences,
        )
        cal_prefs = await aget_memory(
            store,
            ("email_assistant", "cal_preferences"),
            default_cal_preferences,
        )

        # Format system prompt with memory
        memory_prompt = agent_system_prompt_hitl_memory.format(
            triage_instructions=triage_prefs,
            tools_prompt=HITL_MEMORY_TOOLS_PROMPT,
            background=default_background,
            response_preferences=response_prefs,
            cal_preferences=cal_prefs,
        )

        # Append memory prompt to existing system prompt
        new_system_prompt = (
            request.system_prompt + "\n\n" + memory_prompt
            if request.system_prompt
            else memory_prompt
        )

        # Update request with new system prompt
        updated_request = request.override(system_prompt=new_system_prompt)

        # Call handler (may or may not be async)
        return await handler(updated_request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Async version of wrap_tool_call for async agent invocation.

        Identical logic to wrap_tool_call but supports async context and uses async store operations.
        """
        tool_call = request.tool_call
        tool_name = tool_call["name"]
        tool = request.tool  # Get the tool directly from the request

        # Cache tool for later use
        if tool and tool_name not in self.tools_by_name:
            self.tools_by_name[tool_name] = tool

        # STEP 1: Filter - Execute non-HITL tools directly without interruption
        if tool_name not in self.interrupt_on or not self.interrupt_on[tool_name]:
            # Use handler to ensure proper middleware chaining
            return await handler(request)

        # STEP 2: Format display - Get email context and format for display
        email_input = request.runtime.state.get(self.email_input_key)
        description = self._format_interrupt_display(email_input, tool_call)

        # STEP 3: Configure per-tool actions
        config = self._get_action_config(tool_name)

        # STEP 4: Create interrupt
        interrupt_request = {
            "action_request": {"action": tool_name, "args": tool_call["args"]},
            "config": config,
            "description": description,
        }

        # Call interrupt() to create the interrupt and wait for user decision
        # When creating a new interrupt: returns a list of responses [response]
        # When resuming from an interrupt: returns a dict mapping interrupt IDs to decisions
        result = interrupt([interrupt_request])
        print(f"DEBUG: interrupt() returned: {result}, type: {type(result)}")

        # Handle both new interrupt and resume cases
        if isinstance(result, list) and len(result) > 0:
            # New interrupt created, got user decision
            response = result[0]
        elif isinstance(result, dict):
            # Check if result is the decision itself (has "type" key) or a mapping of interrupt IDs
            if "type" in result:
                # Result is the decision dict directly (e.g., {"type": "accept"})
                response = result
            else:
                # Result is a mapping of interrupt IDs to decisions
                # Extract the decision (dict maps interrupt IDs to decisions)
                decisions = list(result.values())
                if decisions:
                    response = decisions[0]
                else:
                    # No decision found, execute tool with original args
                    tool = self.tools_by_name[tool_name]
                    observation = tool.invoke(tool_call["args"])
                    return ToolMessage(content=observation, tool_call_id=tool_call["id"])
        else:
            # Unexpected format, execute tool with original args
            tool = self.tools_by_name[tool_name]
            observation = tool.invoke(tool_call["args"])
            return ToolMessage(content=observation, tool_call_id=tool_call["id"])

        # STEP 5: Handle response type (async version with async store operations)
        return await self._ahandle_response(response, tool_call, request.runtime)

    async def _ahandle_response(
        self, response: dict, tool_call: dict, runtime: ToolRuntime
    ) -> ToolMessage | Command:
        """Async version of _handle_response for async store operations."""
        response_type = response["type"]

        if response_type == "accept":
            return self._handle_accept(tool_call, runtime)
        elif response_type == "edit":
            return await self._ahandle_edit(response, tool_call, runtime)
        elif response_type == "ignore":
            return await self._ahandle_ignore(tool_call, runtime)
        elif response_type == "response":
            return await self._ahandle_response_feedback(response, tool_call, runtime)
        else:
            raise ValueError(f"Invalid response type: {response_type}")

    async def _ahandle_edit(
        self, response: dict, tool_call: dict, runtime: ToolRuntime
    ) -> Command:
        """Async version of _handle_edit with async memory updates."""
        tool = self.tools_by_name[tool_call["name"]]
        tool_name = tool_call["name"]
        initial_args = tool_call["args"]
        edited_args = response["args"]["args"]
        tool_call_id = tool_call["id"]

        # Execute tool with edited args
        observation = tool.invoke(edited_args)

        # Update AI message immutably with edited tool calls
        ai_message = runtime.state["messages"][-1]
        updated_tool_calls = [
            tc if tc["id"] != tool_call_id
            else {"type": "tool_call", "name": tool_name, "args": edited_args, "id": tool_call_id}
            for tc in ai_message.tool_calls
        ]
        updated_ai_message = ai_message.model_copy(update={"tool_calls": updated_tool_calls})

        # Update memory based on tool type (async)
        await self._aupdate_memory_for_edit(tool_name, initial_args, edited_args, runtime)

        # Return Command with both updated AI message and tool message
        return Command(
            update={
                "messages": [
                    updated_ai_message,
                    ToolMessage(content=observation, tool_call_id=tool_call_id),
                ]
            }
        )

    async def _ahandle_ignore(self, tool_call: dict, runtime: ToolRuntime) -> Command:
        """Async version of _handle_ignore with async memory updates."""
        tool_name = tool_call["name"]
        tool_call_id = tool_call["id"]

        # Create feedback message
        if tool_name == "write_email":
            content = "User ignored this email draft. Ignore this email and end the workflow."
        elif tool_name == "schedule_meeting":
            content = "User ignored this calendar meeting draft. Ignore this email and end the workflow."
        elif tool_name == "Question":
            content = "User ignored this question. Ignore this email and end the workflow."
        else:
            raise ValueError(f"Invalid tool call: {tool_name}")

        # Update triage preferences to avoid future false positives (async)
        await self._aupdate_triage_preferences_ignore(tool_name, runtime)

        # Return Command with goto END
        return Command(
            goto=END,
            update={
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]
            },
        )

    async def _ahandle_response_feedback(
        self, response: dict, tool_call: dict, runtime: ToolRuntime
    ) -> ToolMessage:
        """Async version of _handle_response_feedback with async memory updates."""
        tool_name = tool_call["name"]
        tool_call_id = tool_call["id"]
        user_feedback = response["args"]

        # Create feedback message based on tool type
        if tool_name == "write_email":
            content = f"User gave feedback, which can we incorporate into the email. Feedback: {user_feedback}"
        elif tool_name == "schedule_meeting":
            content = f"User gave feedback, which can we incorporate into the meeting request. Feedback: {user_feedback}"
        elif tool_name == "Question":
            content = f"User answered the question, which can we can use for any follow up actions. Feedback: {user_feedback}"
        else:
            raise ValueError(f"Invalid tool call: {tool_name}")

        # Update memory with feedback (async)
        await self._aupdate_memory_for_feedback(tool_name, user_feedback, runtime)

        return ToolMessage(content=content, tool_call_id=tool_call_id)

    async def _aupdate_memory_for_edit(
        self, tool_name: str, initial_args: dict, edited_args: dict, runtime: ToolRuntime
    ):
        """Async version of _update_memory_for_edit."""
        if tool_name == "write_email":
            namespace = ("email_assistant", "response_preferences")
            messages = [
                {
                    "role": "user",
                    "content": f"User edited the email response. Here is the initial email generated by the assistant: {initial_args}. Here is the edited email: {edited_args}. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.",
                }
            ]
        elif tool_name == "schedule_meeting":
            namespace = ("email_assistant", "cal_preferences")
            messages = [
                {
                    "role": "user",
                    "content": f"User edited the calendar invitation. Here is the initial calendar invitation generated by the assistant: {initial_args}. Here is the edited calendar invitation: {edited_args}. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.",
                }
            ]
        else:
            return  # No memory update for other tools

        store = self._get_store(runtime)
        await aupdate_memory(store, namespace, messages)

    async def _aupdate_memory_for_feedback(
        self, tool_name: str, user_feedback: str, runtime: ToolRuntime
    ):
        """Async version of _update_memory_for_feedback."""
        if tool_name == "write_email":
            namespace = ("email_assistant", "response_preferences")
        elif tool_name == "schedule_meeting":
            namespace = ("email_assistant", "cal_preferences")
        else:
            return  # No memory update for Question feedback

        messages = runtime.state["messages"] + [
            {
                "role": "user",
                "content": f"User gave feedback: {user_feedback}. Use this to update the preferences. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.",
            }
        ]

        store = self._get_store(runtime)
        await aupdate_memory(store, namespace, messages)

    async def _aupdate_triage_preferences_ignore(self, tool_name: str, runtime: ToolRuntime):
        """Async version of _update_triage_preferences_ignore."""
        namespace = ("email_assistant", "triage_preferences")

        if tool_name == "write_email":
            feedback = "The user ignored the email draft. That means they did not want to respond to the email. Update the triage preferences to ensure emails of this type are not classified as respond."
        elif tool_name == "schedule_meeting":
            feedback = "The user ignored the calendar meeting draft. That means they did not want to schedule a meeting for this email. Update the triage preferences to ensure emails of this type are not classified as respond."
        elif tool_name == "Question":
            feedback = "The user ignored the Question. That means they did not want to answer the question or deal with this email. Update the triage preferences to ensure emails of this type are not classified as respond."
        else:
            return

        messages = runtime.state["messages"] + [
            {
                "role": "user",
                "content": f"{feedback} Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.",
            }
        ]

        store = self._get_store(runtime)
        await aupdate_memory(store, namespace, messages)
