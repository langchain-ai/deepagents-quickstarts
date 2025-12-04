from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool

@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    # Placeholder response - in real app would send email
    return f"Email sent to {to} with subject '{subject}' and content: {content}"

class TriageEmailInput(BaseModel):
    """Input schema for triaging an email."""
    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email: 'ignore' for irrelevant emails, "
        "'notify' for important information that doesn't need a response, "
        "'respond' for emails that need a reply",
    )

@tool(args_schema=TriageEmailInput)
def triage_email(reasoning: str, classification: Literal["ignore", "notify", "respond"]) -> str:
    """Analyze the email content and classify it into one of three categories:
    - 'ignore' for irrelevant emails (marketing, spam, FYI threads)
    - 'notify' for important information that doesn't need a response (announcements, status updates)
    - 'respond' for emails that need a reply (direct questions, meeting requests, critical issues)

    This tool MUST be called first to determine how to handle the email before taking any other actions.
    """
    return f"Classification Decision: {classification}. Reasoning: {reasoning}"

@tool
class Done(BaseModel):
    """E-mail has been sent."""
    done: bool

@tool
class Question(BaseModel):
      """Question to ask user."""
      content: str
