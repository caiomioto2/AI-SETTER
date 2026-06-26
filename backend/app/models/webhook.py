from pydantic import BaseModel, Field
from typing import Optional


class DMWebhookPayload(BaseModel):
    """Pydantic model for incoming DM webhook payload from GoHighLevel.

    Maps query parameter names from Trigger.dev to Python-friendly field names.

    Attributes:
        message_body: The incoming message text from the lead.
        lead_id: The GoHighLevel lead identifier.
        ghl_account_id: The GoHighLevel location/account ID.
        name: Optional lead name.
        email: Optional lead email.
        phone: Optional lead phone number.
        setter_number: Which setter configuration to use (default: "1").
    """
    # Based on trigger.dev payload format passed as query params
    message_body: str = Field(alias="Message_Body")
    lead_id: str = Field(alias="Lead_ID")
    ghl_account_id: str = Field(alias="GHL_Account_ID")
    name: Optional[str] = Field(default="", alias="Name")
    email: Optional[str] = Field(default="", alias="Email")
    phone: Optional[str] = Field(default="", alias="Phone")
    setter_number: Optional[str] = Field(default="1", alias="Setter_Number")


class AgentResponse(BaseModel):
    """Response model for the AI agent text engine.

    Contains 1-5 message chunks that will be sent to the lead.
    Each field is optional to allow flexible response generation.

    Attributes:
        Message_1: Required - First message chunk to send.
        Message_2: Optional - Second message chunk.
        Message_3: Optional - Third message chunk.
        Message_4: Optional - Fourth message chunk.
        Message_5: Optional - Fifth message chunk.
    """
    Message_1: str
    Message_2: Optional[str] = None
    Message_3: Optional[str] = None
    Message_4: Optional[str] = None
    Message_5: Optional[str] = None
