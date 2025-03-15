from fastapi import FastAPI, HTTPException
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from app.config import ANTHROPIC_API_KEY
from app.models import ResearchRequest, ResearchResponse
from app.tools import get_tools
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="AI Research Assistant API", version="1.0")

# Load tools (Wikipedia, Web Search, File Save)
tools = get_tools()

# Define LLM
llm = ChatAnthropic(api_key=ANTHROPIC_API_KEY, model="gpt-3.5-turbo")

# Define response parser
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Define prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research assistant. Answer the user query using available tools.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Create agent
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


@app.post("/research/", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    """Research API Endpoint"""
    try:
        raw_response = agent_executor.invoke({"query": request.query})
        structured_response = parser.parse(
            raw_response.get("output", {}).get("text", "")
        )
        return structured_response
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
