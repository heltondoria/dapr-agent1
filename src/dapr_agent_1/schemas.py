"""Schemas Pydantic para validação de configurações YAML."""

from typing import Any

from pydantic import BaseModel, Field


class AgentProfileSpec(BaseModel):
    """Especificação do perfil de um agente."""

    name: str
    role: str
    goal: str
    instructions: list[str] = Field(default_factory=list)
    style_guidelines: list[str] = Field(default_factory=list)


class AgentPubSubSpec(BaseModel):
    """Especificação de Pub/Sub para um agente."""

    agent_topic: str
    pubsub_name: str | None = None
    broadcast_topic: str | None = None


class AgentStateSpec(BaseModel):
    """Especificação de estado para um agente."""

    key_prefix: str
    store_name: str | None = None


class AgentMemorySpec(BaseModel):
    """Especificação de memória para um agente."""

    session_id: str
    store_name: str | None = None


class AgentSpec(BaseModel):
    """Especificação completa de um agente."""

    name: str
    profile: AgentProfileSpec
    pubsub: AgentPubSubSpec
    state: AgentStateSpec
    memory: AgentMemorySpec


class WorkflowConfiguration(BaseModel):
    """Configuração técnica de um workflow."""

    team_name: str
    registry_store: str
    pubsub_name: str
    broadcast_topic: str
    workflow_state_store: str
    memory_store: str


class AgentReference(BaseModel):
    """Referência a um arquivo de especificação de agente."""

    spec_file: str


class WorkflowConfig(BaseModel):
    """Configuração completa de um workflow."""

    id: str
    name: str
    description: str
    objective: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    configuration: WorkflowConfiguration
    agents: list[AgentReference]

