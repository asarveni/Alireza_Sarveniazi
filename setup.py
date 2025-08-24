from setuptools import setup, find_packages

setup(
    name=\"tropigem\",
    version=\"0.1.0\",
    packages=find_packages(),
    install_requires=[
        \"numpy>=1.21.0\",
        \"networkx>=2.6.0\", 
        \"matplotlib>=3.5.0\",
        \"scipy>=1.7.0\",
    ],
    author=\"Alireza Sarve Niazi\",
    description=\"Tropical Graph Embedding Framework\",
    python_requires=\">=3.8\",
)
