from pathlib import Path
from setuptools import setup, find_packages

README = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="ai-physicist-central-llm",
    version="0.1.0",
    author="Eva Paunova",
    author_email="",
    description="A specialized LLM for physics with RAG and external tools (SymPy, unit checks).",
    long_description=README,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.9",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research",
    ],
    # Do NOT pin heavy deps here; let users choose minimal/full requirements file.
    install_requires=[
        # Keep this minimal or empty; requirements.txt will drive installs.
    ],
)
