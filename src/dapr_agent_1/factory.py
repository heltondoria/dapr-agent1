"""Factory para criar agentes a partir de especificações."""

import logging

from dapr_agents.agents.configs import (
    AgentMemoryConfig,
    AgentProfileConfig,
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
)
from dapr_agents.agents.durable import DurableAgent
from dapr_agents.llm.chat import ChatClientBase
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.storage.daprstores.stateservice import StateStoreService

from .schemas import AgentSpec, WorkflowConfig

logger = logging.getLogger(__name__)


def create_agent_from_spec(
    spec: AgentSpec,
    workflow_config: WorkflowConfig,
    registry: AgentRegistryConfig,
    llm: ChatClientBase,
) -> DurableAgent:
    """
    Cria uma instância de DurableAgent a partir de uma especificação.

    Args:
        spec: Especificação do agente a ser criado
        workflow_config: Configuração do workflow (para valores padrão)
        registry: Configuração do registry compartilhado
        llm: Cliente LLM a ser usado pelo agente

    Returns:
        Instância configurada de DurableAgent
    """
    # Criar configuração de perfil
    profile = AgentProfileConfig(
        name=spec.profile.name,
        role=spec.profile.role,
        goal=spec.profile.goal,
        instructions=list(spec.profile.instructions),
        style_guidelines=list(spec.profile.style_guidelines),
    )

    # Criar configuração de Pub/Sub
    pubsub_name = spec.pubsub.pubsub_name or workflow_config.configuration.pubsub_name
    broadcast_topic = spec.pubsub.broadcast_topic or workflow_config.configuration.broadcast_topic

    pubsub = AgentPubSubConfig(
        pubsub_name=pubsub_name,
        agent_topic=spec.pubsub.agent_topic,
        broadcast_topic=broadcast_topic,
    )

    # Criar configuração de estado
    state_store_name = spec.state.store_name or workflow_config.configuration.workflow_state_store
    state_store = StateStoreService(store_name=state_store_name, key_prefix=spec.state.key_prefix)
    state = AgentStateConfig(store=state_store)

    # Criar configuração de memória
    memory_store_name = spec.memory.store_name or workflow_config.configuration.memory_store
    memory_store = ConversationDaprStateMemory(store_name=memory_store_name, session_id=spec.memory.session_id)
    memory = AgentMemoryConfig(store=memory_store)

    # Criar o agente
    agent = DurableAgent(
        profile=profile,
        pubsub=pubsub,
        state=state,
        registry=registry,
        memory=memory,
        llm=llm,
    )

    logger.info("Created agent '%s' from specification", spec.name)
    return agent
