# Personal Assistant

Email assistant using deepagents library with two-tiered architecture, human-in-the-loop (HITL) interrupts, and persistent memory system.

## Quickstart

### Python Version

* Ensure you're using Python 3.11 or later
* Required for optimal compatibility with LangGraph

```shell
python3 --version
```

### API Keys

* Sign up for Anthropic API key [here](https://console.anthropic.com/)
* Sign up for LangSmith [here](https://smith.langchain.com/)

### Set Environment Variables

Create a `.env` file in the root directory:

```shell
ANTHROPIC_API_KEY=your_anthropic_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=personal-assistant
```

Or set them in your terminal:

```shell
export ANTHROPIC_API_KEY=your_anthropic_api_key
export LANGSMITH_API_KEY=your_langsmith_api_key
export LANGSMITH_TRACING=true
```

### Package Installation

**Using uv (recommended):**

```shell
# Install uv if you haven't already
pip install uv

# Install the package
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

### Local Deployment

Start LangGraph dev server for local testing:

```bash
langgraph dev
```

This starts a local LangGraph server at [http://localhost:2024](http://localhost:2024) that you can connect to from various UIs.

### Using the DeepAgents UI

The [DeepAgents UI](https://deepagentsui.vercel.app/) ([GitHub repo](https://github.com/langchain-ai/deep-agents-ui)) provides a user-friendly web interface for interacting with LangGraph agents and managing human-in-the-loop interrupts. Instead of programmatically handling interrupts, you can use the UI to review and respond to agent actions.

**Overall Flow:**
1. **Deploy your LangGraph app**: Start locally (`langgraph dev`) or deploy to LangGraph Platform
2. **Connect to the UI**: Open the DeepAgents UI and connect it to your deployment
3. **Interact with interrupts**: When your agent needs approval, review and respond via the UI

**Quick Setup:**

1. Start your local LangGraph deployment:
   ```bash
   langgraph dev
   ```

2. Open the DeepAgents UI in your browser:
   ```
   https://deepagentsui.vercel.app/
   ```

3. Connect to your local graph:
   - Click "Settings" or connection configuration in the UI
   - Add your LangSmith API key
   - Configure your deployment:
     - **Graph ID**: `personal_assistant` (from `langgraph.json`)
     - **Deployment URL**: `http://localhost:2024` (your local dev server)
     - Alternatively, use your LangGraph Cloud deployment URL

4. Process emails through the UI:
   - Send an email input to your graph
   - When the agent generates a tool call that requires approval, an interrupt appears in the UI
   - Review the email context and proposed action
   - Choose your response:
     - **Approve** - Execute the action as-is
     - **Reject** - Skip the action and end workflow

**What You'll See:**

When the email assistant interrupts (for `write_email`, `schedule_meeting`, or `Question`), the UI displays:
- Original email context (subject, sender, content)
- Proposed action (email draft or meeting invite)
- Tool call arguments (editable JSON)
- Available response options based on the tool configuration

**Memory Learning:**

UI responses automatically update the assistant's unified user profile:
- **Reject** ANY tool (`write_email`, `schedule_meeting`, `Question`) → Updates `user_profile` using optional rejection message for better context
- **Approve** → No memory update (agent did the right thing)

See the [Memory System](#memory-system) section below for detailed logic and examples.

**Using LangGraph Studio:**

You can also use LangGraph Studio (included with `langgraph dev`) for testing:
- Navigate to [http://localhost:2024](http://localhost:2024)
- View the graph visualization and execution trace
- Handle interrupts programmatically or via the Studio UI
- Inspect memory state and conversation history

## Architecture Overview

### Two-Tiered Design

The assistant uses a two-tiered architecture to efficiently handle emails:

```
Email Input → Triage Router → [Respond / Notify / Ignore]
                    ↓
            Response Agent (with HITL)
```

**Tier 1: Triage Tool**
- **Purpose**: Classify emails to avoid wasting time on irrelevant messages
- **Classifications**:
  - `respond` - Email requires a response → Continues to draft/schedule
  - `notify` - Important FYI email → Ends workflow with notification
  - `ignore` - Spam, marketing, or irrelevant → Ends workflow
- **Memory**: Learns from reject decisions - when user rejects a draft, updates `triage_preferences` to avoid future misclassifications
- **Location**: `src/personal_assistant/tools/default/email_tools.py` (triage_email tool)

**Tier 2: Response Agent**
- **Purpose**: Generate email drafts and schedule meetings with HITL approval
- **Built with**: `create_deep_agent()` from deepagents library
- **HITL Configuration**: Uses built-in `interrupt_on` parameter for tool interrupts
- **Custom Middleware**: `MemoryInjectionMiddleware` and `PostInterruptMemoryMiddleware` for memory management
- **Tools**: `write_email`, `schedule_meeting`, `check_calendar_availability`, `Question`, `Done`
- **Location**: `src/personal_assistant/email_assistant_deepagents.py`

### HITL System

The assistant uses `create_deep_agent`'s built-in `interrupt_on` parameter for HITL interrupts, with lightweight middleware for memory management:

**Built-in Interrupt Handling** (via `interrupt_on` parameter):
- **Configured tools**: `write_email`, `schedule_meeting`, `Question`
- **Non-interrupted tools**: Execute directly (e.g., `check_calendar_availability`, `Done`)
- **Static descriptions**: Each tool has a plain text description explaining the action and available decisions
- **Per-tool decision configuration**: Different tools allow different decision types

**Interrupt Configuration Format**:
```python
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
```

**Two Decision Types**:
1. **Approve** - Execute tool with original arguments (no memory update)
2. **Reject** - Skip execution, end workflow, update `triage_preferences` to avoid future false positives

**What the UI Displays**:
- The description text explaining the action and available decisions
- Tool arguments in JSON format (to, subject, content for emails; attendees, time for meetings)
- Decision buttons (Approve, Reject) based on `allowed_decisions` configuration

**Memory Middleware**:
- **MemoryInjectionMiddleware**: Injects learned preferences into system prompt before each LLM call
- **PostInterruptMemoryMiddleware**: Detects rejected tool calls and updates memory profiles
  - Uses `before_model()` hook to check for ToolMessages with `status="error"` (indicates rejection)
  - Extracts optional rejection message from ToolMessage content
  - **REJECT detection**: Updates `triage_preferences` with user's rejection feedback
  - **APPROVE**: No memory update needed
- Uses runtime store in deployment, local store in testing

**GenUI Middleware**:
- **GenUIMiddleware**: Pushes UI messages for tool calls to enable custom UI component rendering
- Maps tool names to UI component names for visualization in LangGraph Studio
- Runs after model generation to create UI messages for configured tools
- Example mapping:
  ```python
  tool_to_genui_map={
      "write_email": {"component_name": "write_email"},
      "schedule_meeting": {"component_name": "schedule_meeting"},
  }
  ```

**Locations**:
- Interrupt config: `src/personal_assistant/email_assistant_deepagents.py:66-100`
- Memory injection: `src/personal_assistant/middleware/email_memory_injection.py`
- Post-interrupt updates: `src/personal_assistant/middleware/email_post_interrupt.py`
- GenUI: `src/personal_assistant/middleware/email_genui.py`

### Memory System

The assistant maintains a persistent memory profile that learns from user interactions during HITL interrupts. The profile is stored in a LangGraph Store namespace and automatically updates based on user decisions.

#### Memory Profile

**Single namespace:** `("email_assistant", "user_profile")`

The assistant maintains a unified user profile containing:
1. **User context**: Identity and role information
2. **Triage criteria**: When to respond/notify/ignore emails
3. **Response style**: How to draft emails and handle different scenarios
4. **Meeting scheduling**: Calendar and scheduling preferences

**Storage:**
- Profile size: 15-25 lines of moderate detail
- Updated by: ANY tool rejection (not just specific tools)
- Update logic: LLM analyzes rejection and updates relevant profile section
- Preserves unrelated information during updates

#### Memory Update Trigger Matrix

| User Decision | Tool Call | Profile Section Updated | Update Reason |
|---------------|-----------|-------------------------|---------------|
| **REJECT** | `write_email` | Triage and/or response style | Email shouldn't have been drafted or style was wrong |
| **REJECT** | `schedule_meeting` | Triage and/or scheduling | Meeting shouldn't have been scheduled or timing was wrong |
| **REJECT** | `Question` | Triage and/or response style | Question shouldn't have been asked |
| **APPROVE** | any tool | _(none)_ | No update needed - agent did the right thing |

#### How Memory Updates Work

**Memory Injection** (`MemoryInjectionMiddleware`):
- Runs **before each LLM call** via `wrap_model_call()`
- Fetches unified user profile from single namespace
- Injects it into the system prompt's `< User Profile >` section
- Agent sees current profile on every turn

**Memory Update Detection** (`PostInterruptMemoryMiddleware`):
- **Before model generation** (`before_model`): Detects ToolMessages with `status="error"` (rejections)
- **ANY tool rejection**: Extracts optional rejection message and updates user profile
- **Update logic**: Tool-specific feedback guides LLM on which profile section to update
- **Agent behavior**: Agent is instructed to call Done tool immediately after receiving any rejection

**Memory Update Process**:
1. Build prompt with current memory profile + user's feedback (reject reason)
2. Call LLM with structured output to update profile
3. LLM returns updated profile with reasoning (via `UserPreferences` schema)
4. Save updated profile back to store namespace
5. Next LLM call will see the updated preferences

**Storage Backend**:
- **Local testing**: `InMemoryStore()` (ephemeral, resets on restart)
- **Deployment**: LangGraph platform store (persistent across sessions)
- **Integration**: `StoreBackend` for deepagents compatibility
- **Access**: Middleware uses `runtime.store` in deployment, `self.store` in local testing

### Deployment Modes

**Local Testing Mode** (`for_deployment=False`):
- Creates `InMemoryStore()` and `MemorySaver()`
- Passes store/checkpointer to both top-level workflow and response agent
- Use for: Notebook testing, standalone Python scripts

**Deployment Mode** (`for_deployment=True`):
- Does NOT pass store/checkpointer to graph compilation
- LangGraph platform provides persistence infrastructure
- Use for: `langgraph dev`, LangGraph Cloud, production deployments

## Code Structure

```
examples/personal_assistant/
├── agent.py                          # Deployment entry point (used by langgraph.json)
├── langgraph.json                    # LangGraph deployment config
├── pyproject.toml                    # Package dependencies and metadata
├── test.ipynb                        # Interactive testing notebook
├── README.md                         # This file
│
└── src/personal_assistant/
    ├── __init__.py                   # Package exports
    ├── email_assistant_deepagents.py # Main: Email assistant with triage tool
    ├── schemas.py                    # State schemas (EmailAssistantState, etc.)
    ├── prompts.py                    # System prompts and memory instructions
    ├── configuration.py              # Config loading (minimal)
    ├── utils.py                      # Utilities (memory, formatting, parsing)
    ├── ntbk_utils.py                 # Notebook utilities (rich formatting)
    │
    ├── middleware/
    │   ├── __init__.py
    │   ├── email_memory_injection.py # Memory injection from unified profile
    │   ├── email_post_interrupt.py   # Profile updates on ANY tool rejection
    │   └── email_genui.py            # GenUI integration for tool visualization
    │
    └── tools/
        ├── __init__.py
        ├── base.py                   # Tool loading and registry
        │
        ├── default/                  # Default tools (no external APIs)
        │   ├── __init__.py
        │   ├── email_tools.py        # write_email, Question, Done
        │   ├── calendar_tools.py     # schedule_meeting, check_calendar_availability
        │   └── prompt_templates.py   # Tool prompt templates
        │
        └── gmail/                    # Gmail integration (optional)
            ├── __init__.py
            ├── gmail_tools.py        # Gmail API tools
            └── gmail_utils.py        # Gmail utilities
```

## Test Email Examples

### Example 1: Direct Question (Will be RESPONDED to)

This email will trigger the `respond` classification because it's addressed to Lance with a direct technical question:

```markdown
**Subject**: Question about LangGraph deployment
**From**: Sarah Chen <sarah.chen@techcorp.com>
**To**: Lance Martin <lance@langchain.dev>

Hi Lance,

I'm working on deploying a LangGraph agent to production and ran into an issue with the store configuration. When I try to use a custom store, I get an error about the platform providing its own persistence.

Could you clarify when we should pass a store vs. letting the platform handle it?

Also, do you have any examples of deploying agents with custom HITL middleware?

Thanks!
Sarah
```

**Why this will be responded to:**
- ✅ Addressed TO Lance (not CC'd)
- ✅ Contains direct questions requiring response
- ✅ Technical question about LangGraph (Lance's area)

### Example 2: Meeting Request (Will be RESPONDED to)

```markdown
**Subject**: Sync on agent architecture
**From**: Alex Rodriguez <alex.rodriguez@company.com>
**To**: Lance Martin <lance@langchain.dev>

Hey Lance,

Can we schedule 30 minutes next week to discuss the multi-agent architecture for our project? I have some questions about routing between agents and want to get your input.

I'm free Tuesday afternoon or Thursday morning. Let me know what works!

Best,
Alex
```

**Why this will be responded to:**
- ✅ Direct meeting request to Lance
- ✅ Requires scheduling action
- ✅ Will trigger `schedule_meeting` tool with HITL

### Example 3: FYI Email (Will be IGNORED)

```markdown
**Subject**: Weekly AI Newsletter - Top Articles
**From**: Newsletter <newsletter@techblog.com>
**To**: Lance Martin <lance@langchain.dev>

This week's top articles:

1. New advances in RAG systems
2. Latest LLM benchmarks
3. Multi-agent coordination patterns

[Read more...]
```

**Why this will be ignored:**
- ❌ Newsletter/marketing content
- ❌ No action required
- ❌ No direct questions

## Usage

### Python API

```python
from personal_assistant import create_email_assistant

# Create agent (local testing mode)
agent = create_email_assistant(for_deployment=False)

# Example email as markdown string
email_markdown = """
**Subject**: Question about LangGraph deployment
**From**: sarah.chen@techcorp.com
**To**: lance@langchain.dev

Hi Lance,

I'm working on deploying a LangGraph agent to production and ran into an issue with the store configuration...

Could you clarify when we should pass a store vs. letting the platform handle it?

Thanks!
Sarah
"""

# Process email - agent accepts message as simple string
config = {"configurable": {"thread_id": "thread-1"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": email_markdown}]},
    config=config
)

# Check for interrupts (HITL)
if "__interrupt__" in result:
    print("Agent is waiting for approval")
    # Resume with decision
    from langgraph.types import Command
    result = agent.invoke(
        Command(resume=[{"type": "accept"}]),
        config=config
    )
```

### Notebook Testing

See `test.ipynb` for interactive examples with rich formatting:
- Process emails step-by-step
- Visualize agent reasoning
- Test all HITL response types (accept, edit, ignore, response)
- View memory updates in real-time

### Deployment

The agent is configured for LangGraph deployment via `agent.py` and `langgraph.json`.

**Key difference**: In deployment mode, the agent doesn't create its own store/checkpointer:

```python
# agent.py uses deployment mode
graph = create_email_assistant(for_deployment=True)
```

This allows the LangGraph platform to provide persistent storage, checkpointing, and the LangGraph Studio UI.

## Features

- **Integrated Triage**: Single deepagent with triage tool for efficient email classification
- **Built-in Interrupt System**: Uses `interrupt_on` parameter with clear descriptions for UI display
- **Persistent Memory**: Learns from user feedback to improve triage classification
- **Two Decision Types**: Approve or reject actions with automatic memory updates
- **GenUI Integration**: Custom UI components for tool visualization in LangGraph Studio
- **Async Support**: Works with both sync and async invocation (invoke/ainvoke, stream/astream)
- **Deployment Ready**: Optimized for LangGraph Cloud with platform-provided persistence

## Testing

Run the notebook for interactive testing:

```bash
jupyter notebook test.ipynb
```

Or run the standalone script:

```bash
python -m personal_assistant.email_assistant_deepagents
```
