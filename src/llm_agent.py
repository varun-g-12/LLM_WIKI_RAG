import logging
from typing import Any, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class AgentState(TypedDict):
    user_query: str
    context: Optional[str]
    answer: str
    review: Literal["yes", "no"]


def evaluate_answer_quality(state: AgentState):
    logging.info("Evaluating answer quality for user query: %s", state["user_query"])

    prompt = """You are a Quality Assurance expert. Your task is to evaluate if the provided answer directly addresses the user's query.

    Evaluation criteria:
    - Relevance: Does the answer specifically address the main question?
    - Completeness: Is the answer sufficiently complete?
    - Clarity: Is the answer clear and understandable?

    Respond ONLY with:
    'yes' - if the answer meets all criteria
    'no' - if the answer fails any criterion"""

    chat_model = initialize_chat_model()

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(
            content=f"User's query: {state['user_query']}\nAnswer: {state['answer']}"
        ),
    ]
    logging.info("Sending evaluation request to chat model")
    response = chat_model.invoke(messages)
    logging.info("Received evaluation response: %s", response.content)

    return {"review": response.content}


def generate_answer(state: AgentState) -> dict[str, Any]:
    logging.info("Generating answer for user query: %s", state["user_query"])

    prompt = """You are an expert knowledge assistant. Your role is to provide accurate and relevant answers.

    When responding:
    - If no context is provided (context is None): Answer directly based on your general knowledge
    - If context is provided: Use the given context to formulate a precise and relevant answer
    
    Keep responses concise and factual. Always maintain a professional tone."""

    chat_model = initialize_chat_model()

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(
            content=f"Query: {state['user_query']}\nContext: {state['context']}"
        ),
    ]
    response = chat_model.invoke(messages)
    logging.info("Generated answer: %s", response.content)
    return {"answer": response.content}


def initialize_chat_model(
    model: str = "gpt-4o-mini", temperature: int = 0
) -> ChatOpenAI:
    logging.info(
        "Initializing chat model with model: %s and temperature: %d", model, temperature
    )
    return ChatOpenAI(model=model, temperature=temperature)


def create_agent() -> CompiledStateGraph:
    logging.info("Creating state graph agent")
    state_graph = StateGraph(AgentState)

    state_graph.add_node("answer_node", generate_answer)
    state_graph.add_node("review_node", evaluate_answer_quality)

    state_graph.add_edge(START, "answer_node")
    state_graph.add_edge("answer_node", "review_node")
    state_graph.add_edge("review_node", END)

    compiled_graph = state_graph.compile()
    logging.info("State graph agent created and compiled")
    return compiled_graph


def load_environment_variables() -> None:
    logging.info("Loading environment variables")
    env_loaded = load_dotenv()
    if env_loaded:
        logging.info("Environment variables loaded successfully")
    else:
        logging.error("Error loading environment variables")
        raise Exception("Error loading environment variables")


def get_response(query: str) -> dict[str, Any]:
    logging.info("Received query: %s", query)

    load_environment_variables()

    agent = create_agent()

    response = agent.invoke({"user_query": query, "context": None})
    logging.info("Response generated: %s", response)
    return response


if __name__ == "__main__":
    # query = "What is the capital of Nigeria?"
    query = "Who won IPL in 2024?"
    response = get_response(query)
    print(response)
