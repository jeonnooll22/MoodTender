from pydantic import BaseModel

class UserCreate(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class LLMRequest(BaseModel):
    user_id: int
    text: str

class LLMResponse(BaseModel):
    reply: str
    
class ChatRequest(BaseModel):
    user_id: int
    text: str
