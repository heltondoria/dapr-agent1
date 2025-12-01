"""Loader para carregar configurações de workflows e agentes a partir de YAML."""

import logging
import os
from pathlib import Path

import yaml
from dapr_agents.agents.configs import AgentRegistryConfig
from dapr_agents.storage.daprstores.stateservice import StateStoreService

from .schemas import AgentSpec, WorkflowConfig

logger = logging.getLogger(__name__)


def list_workflow_configs(workflows_dir: str) -> list[str]:
    """
    Lista todos os arquivos de workflow disponíveis.

    Args:
        workflows_dir: Diretório contendo os arquivos de workflow

    Returns:
        Lista de IDs de workflows disponíveis (nomes dos arquivos sem extensão)
    """
    workflows_path = Path(workflows_dir)
    if not workflows_path.exists():
        logger.warning("Workflows directory does not exist: %s", workflows_dir)
        return []

    workflow_files = [f.stem for f in workflows_path.glob("*.yaml") if f.is_file() and not f.name.startswith(".")]
    workflow_files.extend([f.stem for f in workflows_path.glob("*.yml") if f.is_file() and not f.name.startswith(".")])

    return sorted(workflow_files)


def load_workflow_config(config_path: str) -> WorkflowConfig:
    """
    Carrega um arquivo de workflow.

    Args:
        config_path: Caminho para o arquivo YAML do workflow

    Returns:
        Configuração do workflow validada

    Raises:
        FileNotFoundError: Se o arquivo não existir
        ValueError: Se o arquivo YAML for inválido
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Workflow config file not found: {config_path}")

    try:
        with config_file.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"Empty workflow config file: {config_path}")

        workflow_config = WorkflowConfig.model_validate(data)
        logger.info("Loaded workflow config: %s (%s)", workflow_config.id, workflow_config.name)
        return workflow_config
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in workflow config: {config_path}") from e
    except Exception as e:
        raise ValueError(f"Error loading workflow config: {config_path}") from e


def load_agent_spec(spec_path: str, base_path: str | None) -> AgentSpec:
    """
    Carrega spec individual de um agente.

    Args:
        spec_path: Caminho para o arquivo YAML do agente (pode ser relativo)
        base_path: Caminho base para resolver caminhos relativos

    Returns:
        Especificação do agente validada

    Raises:
        FileNotFoundError: Se o arquivo não existir
        ValueError: Se o arquivo YAML for inválido
    """
    spec_file = Path(spec_path)
    if not spec_file.is_absolute() and base_path:
        spec_file = Path(base_path) / spec_file

    if not spec_file.exists():
        raise FileNotFoundError(f"Agent spec file not found: {spec_file}")

    try:
        with spec_file.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"Empty agent spec file: {spec_file}")

        agent_spec = AgentSpec.model_validate(data)
        logger.info("Loaded agent spec: %s", agent_spec.name)
        return agent_spec
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in agent spec: {spec_file}") from e
    except Exception as e:
        raise ValueError(f"Error loading agent spec: {spec_file}") from e


def load_agent_specs(workflow_config: WorkflowConfig, base_path: str | None = None) -> list[AgentSpec]:
    """
    Carrega todas as specs de agentes de um workflow.

    Args:
        workflow_config: Configuração do workflow
        base_path: Caminho base para resolver caminhos relativos dos agentes

    Returns:
        Lista de especificações de agentes

    Raises:
        FileNotFoundError: Se algum arquivo de agente não existir
        ValueError: Se algum arquivo YAML for inválido
    """
    agent_specs = []
    config_dir = base_path or os.getcwd()

    for agent_ref in workflow_config.agents:
        spec_path = agent_ref.spec_file
        # Se o caminho não começa com "agents/", assumimos que é relativo ao configs/
        if not spec_path.startswith("agents/"):
            spec_path = f"agents/{spec_path}"

        full_spec_path = os.path.join(config_dir, spec_path)
        agent_spec = load_agent_spec(full_spec_path, base_path=config_dir)
        agent_specs.append(agent_spec)

    logger.info("Loaded %d agent specs for workflow '%s'", len(agent_specs), workflow_config.id)
    return agent_specs


def create_registry_config(workflow_config: WorkflowConfig) -> AgentRegistryConfig:
    """
    Cria configuração do registry compartilhado a partir da configuração do workflow.

    Args:
        workflow_config: Configuração do workflow

    Returns:
        Configuração do registry
    """
    registry_store = StateStoreService(store_name=workflow_config.configuration.registry_store)
    registry = AgentRegistryConfig(store=registry_store, team_name=workflow_config.configuration.team_name)
    return registry
