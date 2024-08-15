import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import AsyncGenerator

load_dotenv()

app = FastAPI()

search = DuckDuckGoSearchRun()
tools = [search]

prompt = """
        Write a comprehensive 4000-5000 word blog post in markdown format covering the following query:
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
prompt_template = PromptTemplate.from_template(template=prompt)

llm = ChatOpenAI(temperature=0, streaming=True)

agent = create_openai_functions_agent(llm.with_config({"tags": ["agent_llm"]}), tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools).with_config(
    {"run_name": "Agent"}
)
async def generate_response(message: str):
    async for chunk in agent_executor.astream({"input": message}):
        content = chunk['output'].replace("\n", "<br>")
        yield f"data: {content}\n\n"

@app.get("/chat_stream/{message}")
async def chat_stream(message: str):
    return StreamingResponse(generate_response(message = message), media_type="text/event-stream")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
