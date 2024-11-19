from setuptools import setup, find_packages

setup(
    name="LLM_WIKI_RAG",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain==0.3.7",
        "langchain-openai==0.2.8",
        "langgraph==0.2.50",
        "python-dotenv==1.0.1",
        "langchain-community==0.3.7",
        "wikipedia==1.4.0",
    ],
)
