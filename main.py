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

def format_story(story: str) -> str:
    # Remove "### Instruction"
    story = story.replace("### Instruction", "")

    # Ensure the title shows up without "###"
    story = story.replace("#### Title:", "<h2>Title:</h2>")

    # Address the bold formatting issue
    formatted_story = story.replace("**", "<b>").replace("\n", "<br>")

    return formatted_story

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

        # Add HTML formatting
        formatted_story = story.replace("**", "<b>").replace("\n", "<br>")

        print("the story:", formatted_story)
        return JSONResponse(content={"response": formatted_story})
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


# Run the application using Uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
