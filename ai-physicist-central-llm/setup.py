from setuptools import setup, find_packages

setup(
    name="ai-physicist-central-llm",
    version="0.1.0",
    author="Your Name",
    description="A specialized LLM for physics with RAG and tools",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
)
