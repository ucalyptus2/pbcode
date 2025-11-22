from setuptools import setup, find_packages

__version__ = "2025.2.3"

setup(
    name="predibase",
    version=__version__,
    package_dir={"": "predibase"},
    packages=find_packages(where="predibase"),
    install_requires=[
        "pandas",
        "requests",
        "semantic-version",
        "websockets",
        "python-dateutil",
        "decorator",
        "progress-table",
        "tqdm",
        "urllib3",
        "gevent",
        "numpy",
        "tritonclient[all]",
        "grpcio",
        "grpcio-tools",
        "pydantic",
        "tabulate",
        "dataclasses-json",
        "PyYAML",
        "rich",
        "typer",
        "predibase-api",
        "openai",
        "lorax-client",
    ],
    entry_points={
        "console_scripts": [
            "predibase = predibase.cli:main",
        ],
    },
)
