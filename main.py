from fastapi import FastAPI,status
from openai import OpenAI
from pydantic import BaseModel,Field,EmailStr,field_validator
from fastapi.responses import RedirectResponse
#ex1
app=FastAPI()
openai_client=OpenAI(api_key="")

@app.get("/",include_in_schema=False)
def root_controller():
    return{"status":"healthy"}

# @app.get("/", include_in_schema=False) 
# def docs_redirect_controller():
#     return RedirectResponse(url="/docs", status_code=status.HTTP_303_SEE_OTHER)

@app.get("/chat")
def chat_controller(prompt:str="Inspire me"):
    response=openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role":"system","content":"you are a helpful assistant ."},
            {"role":"user","content":prompt}
        ],
    )
    statement=response.choices[0].message.content
    return{"statement":statement}

#ex2
class UserCreate(BaseModel):
    username:str
    password:str

    @field_validator("password")
    def validate_password(cls,value):
        if len(value) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(char.isdigit() for char in value):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in value):
            raise ValueError('Password must contain at least one uppercase letter')
        return value

@app.post("/users")
async def create_user_controller(user:UserCreate):
    return{"name":user.username,"message": "Account successfully created"}





