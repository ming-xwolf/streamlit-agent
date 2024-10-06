from langchain_experimental.tools import PythonREPLTool
from langchain import hub
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain.agents import AgentExecutor,AgentType,create_structured_chat_agent
from langchain_openai import ChatOpenAI
from langchain_community.utilities import WikipediaAPIWrapper
from ollama_llm import get_llm

@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int


@tool
def add(first_int: int, second_int: int) -> int:
    "Add two integers."
    return first_int + second_int


@tool
def exponentiate(base: int, exponent: int) -> int:
    "Exponentiate the base to the exponent power."
    return base ** exponent


pythonREPLTool = PythonREPLTool()
search = DuckDuckGoSearchRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

prompt = hub.pull("hwchase17/structured-chat-agent")
tools = [pythonREPLTool, wikipedia, search, multiply, add, exponentiate]

llm = get_llm()
agent = create_structured_chat_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True)

agent_executor.invoke({"input": "2 multiple 3"})

agent_executor.invoke({"input": "HUNTER X HUNTER是什么?"})