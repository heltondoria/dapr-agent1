"""Declarative Multi-Agent System for Dapr Agents."""

from .orchestrator import main as orchestrator_main
from .worker import main as worker_main
from .worker import DynamicAgent, DynamicAgentWorker

__all__ = [
    "orchestrator_main",
    "worker_main",
    "DynamicAgent",
    "DynamicAgentWorker",
]
