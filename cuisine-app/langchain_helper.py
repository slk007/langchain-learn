from dotenv import load_dotenv
from langchain.chains import LLMChain, SequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

load_dotenv()

llm = OpenAI(temperature=0.5)
memory = ConversationBufferMemory()


def generate_restaurant_name_and_items(cuisine):
    # Chain 1: Restaurant Name
    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template="I want to open restaurant for {cuisine} food. Suggest a fancy name for this."
    )
    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")

    # Chain 2: Menu Items
    prompt_template_items = PromptTemplate(
        input_variables=['restaurant_name'],
        template="""Suggest only veg menu items for {restaurant_name}. Return it is as a comma separated string"""
    )
    food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")

    chain = SequentialChain(
        chains=[name_chain, food_items_chain],
        input_variables=["cuisine"],
        output_variables=["restaurant_name", "menu_items"]
    )
    response = chain({'cuisine': cuisine})

    # memory
    chain_with_memory = LLMChain(llm=llm, prompt=prompt_template_name, memory=memory)
    name = chain_with_memory.run("Indian")
    print(name)
    print(chain_with_memory.memory)
    print(chain_with_memory.memory.buffer)

    return response


if __name__ == "__main__":
    print((generate_restaurant_name_and_items("Italian")))
