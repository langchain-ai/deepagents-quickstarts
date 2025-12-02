# 🚀🧠 Deepagent Quickstarts

[Deepagents](https://github.com/langchain-ai/deepagents) is a simple, open source agent harness. It uses some common principle seen in popular agents such as [Claude Code](https://code.claude.com/docs) and [Manus](https://www.youtube.com/watch?v=6_BcCthVvb8), including **planning** (prior to task execution), **computer access** (giving the able access to a shell and a filesystem), and **sub-agent delegation** (isolated task execution). This repo has a collection of quickstarts that demonstrate different agents that can be easily configured on top of the  `deepagents` harness.

<img width="1536" height="1024" alt="quick" src="https://github.com/user-attachments/assets/d3d94751-2c33-4686-9d11-a43b975fc4fe" />

## 📚 Resources

- **[Documentation](https://docs.langchain.com/oss/python/deepagents/overview)** - Full overview and API reference
- **[Deepagents Repo](https://github.com/langchain-ai/deepagents)** - Deepagents package 

## Quickstarts

Here are the currently supported quickstarts:

| Quickstart Name | Location | Description | Usage Options |
|----------------|----------|-------------|---------------|
| [Deep Research](deep_research/README.md) | `deep_research/` | A research agent that conducts multi-step web research using Tavily for URL discovery, fetches full webpage content, and coordinates work through parallel sub-agents and strategic reflection | **Jupyter Notebook** or **LangGraph Server** |
| [Developer](developer/README.md) | `developer/` | A software developer agent that will break down a coding task into a series of steps and execute on them, using web search via `Tavily` to help find the best tooling and practices. | **Jupyter Notebook** or **LangGraph Server** | 

## Built-In Deepagent Components

To use these quickstarts, it's important to understand the built-in components of the deepagent harness. You can see the deepagents [repo](https://github.com/langchain-ai/deepagents) for more details, but as a quick reference, here are the built-in tools and middleware:

### Tools

Every deepagent comes with a set of general tools by default:

<img width="1536" height="1024" alt="deepagent" src="https://github.com/user-attachments/assets/e16f8e5c-ae76-4716-8e14-d21216cc1ab3" />

| Tool Name | Description |
|-----------|-------------|
| `write_todos` | Create and manage structured task lists for tracking progress through complex workflows |
| `ls` | List all files in a directory (requires absolute path) |
| `read_file` | Read content from a file with optional pagination (offset/limit parameters) |
| `write_file` | Create a new file or completely overwrite an existing file |
| `edit_file` | Perform exact string replacements in files |
| `glob` | Find files matching a pattern (e.g., `**/*.py`) |
| `grep` | Search for text patterns within files |
| `execute` | Run shell commands in a sandboxed environment (only if backend supports SandboxBackendProtocol) |
| `task` | Delegate tasks to specialized sub-agents with isolated context windows |

### Middleware

Deepagent also use some built-in ["middleware"](https://docs.langchain.com/oss/python/langchain/middleware/overview), which can:

1. **Provide tools** - Add new tools to the agent's toolkit (e.g., `FilesystemMiddleware` adds `ls`, `read_file`, `write_file`, etc.)
2. **Wrap model calls** - Inject system prompts and modify model requests before they're sent
3. **Wrap tool calls** - Process tool call results after tools execute (e.g., `SummarizationMiddleware` summarizes large conversation history)

Every deepagent includes the following middleware by default (applied in order). Some middleware are provided by the `deepagents` package (`FilesystemMiddleware`, `SubAgentMiddleware`, `PatchToolCallsMiddleware`), while others come from `langchain` (`TodoListMiddleware`, `SummarizationMiddleware`, `HumanInTheLoopMiddleware`) and `langchain-anthropic` (`AnthropicPromptCachingMiddleware`):

| Middleware | Tools Added | What It Does |
|------------|-------------|--------------|
| **TodoListMiddleware** | `write_todos`, `read_todos` | Task planning and progress tracking. Enables agents to create todo lists, break down tasks, and track completion. |
| **FilesystemMiddleware** | `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`* | File system operations and context offloading. Automatically saves large tool results (>20K tokens) to files to prevent context overflow. |
| **SubAgentMiddleware** | `task` | Task delegation to specialized subagents with isolated contexts. Subagents handle complex subtasks independently and return summaries. |
| **SummarizationMiddleware** | N/A | Automatic conversation summarization when context exceeds 170K tokens. Keeps last 6 messages intact while summarizing older content. |
| **AnthropicPromptCachingMiddleware** | N/A | Prompt caching for Anthropic models to reduce API costs. Marks static system prompts for server-side caching. |
| **PatchToolCallsMiddleware** | N/A | Fixes "dangling" tool calls from interruptions. Adds placeholder responses to prevent validation errors. |
| **HumanInTheLoopMiddleware** | N/A | Human approval workflow for sensitive operations. Creates breakpoints for specified tools (only when `interrupt_on` configured). |

\* The `execute` tool is only available if the backend implements `SandboxBackendProtocol`

## Writing Custom Instructions

When building your own custom deepagent, you can provide a `system_prompt` parameter to `create_deep_agent()`. This custom prompt is **appended to** default instructions that are automatically injected by middleware. Understanding this layering is crucial for writing effective custom instructions. Read about the [default instructions in the deepagents README](https://github.com/langchain-ai/deepagents?tab=readme-ov-file#built-in-tools) below. You can follow some general guidelines below, and see specific examples in the quickstart folders. 

**Do:**
- ✅ Define domain-specific workflows (e.g., research methodology, data analysis steps)
- ✅ Provide concrete examples for your use case
- ✅ Add specialized guidance (e.g., "batch similar research tasks into a single TODO")
- ✅ Define stopping criteria and resource limits
- ✅ Explain how tools work together in your workflow

**Don't:**
- ❌ Re-explain what standard tools do (already covered by middleware)
- ❌ Duplicate middleware instructions about tool usage
- ❌ Contradict default instructions (work with them, not against them)