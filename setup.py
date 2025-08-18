# setup.py
import setuptools

setuptools.setup(
    name="ai_agent_pkg",
    version="0.0.1",
    author="Your Name",
    author_email="your.email@example.com",
    description="An AI News Agent package",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"}
)