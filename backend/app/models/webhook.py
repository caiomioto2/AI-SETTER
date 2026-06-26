from pydantic import BaseModel, Field
from typing import Optional

class DMWebhookPayload(BaseModel):
    # Based on trigger.dev payload format passed as query params
    message_body: str = Field(alias="Message_Body")
    lead_id: str = Field(alias="Lead_ID")
    ghl_account_id: str = Field(alias="GHL_Account_ID")
    name: Optional[str] = Field(default="", alias="Name")
    email: Optional[str] = Field(default="", alias="Email")
    phone: Optional[str] = Field(default="", alias="Phone")
    setter_number: Optional[str] = Field(default="1", alias="Setter_Number")

class AgentResponse(BaseModel):
    Message_1: str
    Message_2: Optional[str] = None
    Message_3: Optional[str] = None
    Message_4: Optional[str] = None
    Message_5: Optional[str] = None
