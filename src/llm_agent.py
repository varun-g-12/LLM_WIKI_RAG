import logging
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from src.prompt import WIKI_ROUTER_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class AgentState(TypedDict):
    user_query: str
    route: Literal["wiki", "no_wiki"]


def initialize_llm_model(model: str = "gpt-4o-mini", temp: float = 0.0) -> ChatOpenAI:
    logging.debug(f"Initializing LLM model with model={model} and temperature={temp}")
    return ChatOpenAI(model=model, temperature=temp)


def route_query(state: AgentState) -> dict[str, Literal["wiki", "no_wiki"]]:
    logging.info("Entering route_query function")
    llm_model = initialize_llm_model()
    prompt = WIKI_ROUTER_PROMPT
    messages = [
        SystemMessage(prompt),
        HumanMessage(f"""User's query: {state["user_query"]}"""),
    ]
    logging.debug(f"Messages prepared for LLM model: {messages}")
    response = llm_model.invoke(messages)

    if hasattr(response, "content"):
        logging.info(f"LLM model response: {response.content}")
        return {"route": response.content}  # type: ignore
    else:
        logging.error("Error in response from LLM model")
        raise Exception("Error in response")


def create_agent():
    logging.info("Creating state graph agent")
    state_graph_agent = StateGraph(AgentState)

    state_graph_agent.add_node("route_query", route_query)

    state_graph_agent.add_edge(START, "route_query")
    state_graph_agent.add_edge("route_query", END)

    logging.info("State graph agent created successfully")
    return state_graph_agent.compile()


def load_environment_variables() -> None:
    logging.info("Loading environment variables")
    env_status = load_dotenv()
    if not env_status:
        logging.error("Error loading environment variables")
        raise Exception("Error loading environment variables")
    logging.info("Environment variables loaded successfully")


def process_user_query(user_query: str):
    logging.info(f"Received user query: {user_query}")

    # Load environment variables
    load_environment_variables()

    # Create agent
    agent = create_agent()

    try:
        logging.info("Invoking agent with user query")
        response = agent.invoke({"user_query": user_query})
        logging.info(f"Agent response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error in agent invocation: {e}")
        raise Exception("Error in agent invocation :{}".format(e))


if __name__ == "__main__":
    query = "Who won IPL in 2024"
    logging.info(f"Main execution started with query: {query}")
    response = process_user_query(query)
    logging.info(f"Response received: {response}")

    print(response)
