from pydantic import BaseModel, Field
from typing_extensions import TypedDict, NotRequired

class EmailAssistantState(TypedDict):
    """State for email assistant agent using deepagents library.

    The agent accepts email content as a simple message string and handles
    triage through the triage_email tool.
    """
    messages: list  # Required by MessagesState
    email_input: NotRequired[dict]  # Optional email context for middleware display formatting

class EmailData(TypedDict):
    id: str
    thread_id: str
    from_email: str
    subject: str
    page_content: str
    send_time: str
    to_email: str

class UserPreferences(BaseModel):
    """Updated user preferences based on user's feedback."""
    chain_of_thought: str = Field(description="Reasoning about which user preferences need to add/update if required")
    user_preferences: str = Field(description="Updated user preferences")