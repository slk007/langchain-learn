from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import OpenAI

load_dotenv()

llm = OpenAI(temperature=0.5)

tools = load_tools(["wikipedia", "llm-math"], llm=llm)

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

print(agent.run("When was Elon Musk born ? What is his age right now in 2023 ?"))

