from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferWindowMemory

# remembers all conversations
convo = ConversationChain(llm=OpenAI(temperature=0.7))
print(convo.prompt.template)

convo.run("Question 1")
convo.run("Question 2")
convo.run("Question 3")

print(convo.memory)
print(convo.memory.buffer)

# remembers conversation till last 2 QnA
memory = ConversationBufferWindowMemory(k=2)
convo_with_less_memory = ConversationChain(llm=OpenAI(temperature=0.7), memory=memory)

convo.run("Question 1")
convo.run("Question 2")
convo.run("Question 3")

print(convo.memory)
print(convo.memory.buffer)
