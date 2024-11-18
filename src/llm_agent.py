import logging
from typing import Any, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_community.retrievers import WikipediaRetriever
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
    context: Optional[list[str]]
    answer: str
    review: Literal["yes", "no"]
    feedback: str
    wiki_query: str


def wiki_search(state: AgentState) -> dict[str, Any]:
    query = state["wiki_query"]
    logging.info("Searching Wikipedia for query: %s", query)
    retriever = WikipediaRetriever(top_k_results=3)  # type: ignore
    response = retriever.invoke(query)
    return {"context": response}


def provide_feedback(state: AgentState) -> dict[str, Any]:
    logging.info("Providing feedback for user query: %s", state["user_query"])

    prompt = """You are a precise feedback evaluator. Analyze the alignment between questions and answers.

    Task:
    1. Compare user query with provided answer
    2. Identify information gaps

    Required output format (JSON):
    {
        "feedback": "One clear sentence describing what's missing from the answer",
        "wiki_query": "Specific search term for Wikipedia to fill the gap"
    }

    Guidelines:
    - Feedback must be actionable and specific
    - Wiki query should target the missing information directly
    - Keep feedback concise and constructive"""

    chat_model = initialize_chat_model().with_structured_output(method="json_mode")

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(
            content=f"User's query: {state['user_query']}\nAnswer: {state['answer']}"
        ),
    ]
    logging.info("Sending feedback request to chat model")
    response = chat_model.invoke(messages)
    logging.info("Received feedback response: %s", response)

    return {
        "feedback": response["feedback"],
        "wiki_query": response["wiki_query"],
    }


def router(state: AgentState) -> str:
    return state["review"]


def evaluate_answer_quality(state: AgentState) -> dict[str, Any]:
    logging.info("Evaluating answer quality for user query: %s", state["user_query"])

    prompt = """You are a Quality Assurance expert. Your task is to evaluate if the provided answer directly addresses the user's query.

    Respond ONLY with:
    'yes' - if the user query is answered 
    'no' - if the user query is been not answered"""

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
    state_graph.add_node("feedback_node", provide_feedback)
    state_graph.add_node("wiki_node", wiki_search)

    state_graph.add_edge(START, "answer_node")
    state_graph.add_edge("answer_node", "review_node")
    state_graph.add_conditional_edges(
        "review_node", router, {"yes": END, "no": "feedback_node"}
    )
    state_graph.add_edge("feedback_node", "wiki_node")
    state_graph.add_edge("wiki_node", "answer_node")

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
    query = "when is the 2025 IPL is scheduled "
    response = get_response(query)

    print("=" * 100)
    print(response["answer"])
