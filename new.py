from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from langchain.tools import tool
from langchain_core.callbacks import Callbacks
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from duckduckgo_search import DDGS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize the model
model = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.0,
    model_name="gpt-3.5-turbo",
    streaming=True,  # ! important
    callbacks=[StreamingStdOutCallbackHandler()]  # ! important
)

# Define the prompt template
template = """
        Write a comprehensive 4000-5000 word blog post covering the following query:
        {input} 
        The blog post should be well-structured with the following elements: 
        - Clear and informative introduction that provides an overview of the key topics 
        - Detailed sections with subheadings for each major topic 
        - Bullet point lists to highlight key facts, statistics, and insights 
        - Smooth transitions between sections to guide the reader 
        - In-depth analysis and commentary on the implications and significance of the topics 
        - Comparisons to industry trends and competitors where relevant 
        - Forward-looking predictions and speculations on the future developments 
        - Conclusion that summarizes the main takeaways and leaves the reader with a strong impression Use a conversational yet authoritative tone throughout the blog post. 
        Avoid jargon and explain technical concepts in simple terms. Incorporate relevant data points, expert quotes, and real-world examples to support the analysis. 
        The writing should be concise, engaging, and informative, avoiding unnecessary fluff or filler content. 
        Maintain a brisk pace and focus on delivering valuable insights to the reader. 
        The final output should be between 4000-5000 words, with a well-structured flow that keeps the reader interested and informed from start to finish. 
        {agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template=template)

# Initialize tools and agent
search = DuckDuckGoSearchRun()
tools = [search]
agent = create_openai_tools_agent(model.with_config({"tags": ["agent_llm"]}), tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools).with_config(
    {"run_name": "Agent"}
)

# Define the async response generator
async def generate_response(message: str):
    async for chunk in agent_executor.astream({"input": message}):
        content = chunk['output'].replace("\n", "<br>")
        yield f"data: {content}\n\n"

# Initialize FastAPI app
app = FastAPI()

@app.get("/chat_stream/{message}")
async def chat_stream(message: str):
    return StreamingResponse(generate_response(message=message), media_type="text/event-stream")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
