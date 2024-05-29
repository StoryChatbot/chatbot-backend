from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class StoryRequest(BaseModel):
    prompt: str


class StoryResponse(BaseModel):
    story: str


# Initialize the Ollama model with LangChain
ollama_model = Ollama(model="chatbot")

# Define a prompt template
prompt_template = PromptTemplate(
    input_variables=["prompt"],
    template="""### Instruction:
{prompt}
### Response:"""
)

# Create a LangChain with the model and prompt template
story_chain = prompt_template | ollama_model


@app.post("/generate_story", response_model=StoryResponse)
def generate_story(request: StoryRequest):
    # Print the request payload
    print("Received request:", request.dict())

    # Process the request
    prompt = request.prompt

    try:
        print("hello i m in")
        # Generate story using the LangChain
        story = ollama_model.invoke(prompt)
        print("the story:", story)
        return JSONResponse(content={"response": story})
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))



# Run the application using Uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
