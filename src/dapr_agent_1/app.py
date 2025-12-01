"""Aplicação principal para sistema multi-agente declarativo."""

import argparse
import contextlib
import logging
import os
import sys
from pathlib import Path

import dapr.ext.workflow as wf
from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    AgentPubSubConfig,
    AgentStateConfig,
)
from dapr_agents.agents.durable import DurableAgent
from dapr_agents.agents.orchestrators.llm import LLMOrchestrator
from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow.runners import AgentRunner
from dotenv import load_dotenv

from .factory import create_agent_from_spec
from .loader import (
    create_registry_config,
    list_workflow_configs,
    load_agent_specs,
    load_workflow_config,
)

# Load environment
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("dapr-agent-1")


def get_configs_dir() -> Path:
    """Retorna o diretório de configurações."""
    # Tenta encontrar configs/ relativo ao diretório atual ou ao diretório do script
    script_dir = Path(__file__).parent.parent.parent
    configs_dir = script_dir / "configs"
    if configs_dir.exists():
        return configs_dir

    # Fallback para diretório atual
    configs_dir = Path.cwd() / "configs"
    if configs_dir.exists():
        return configs_dir

    raise FileNotFoundError(
        "Configs directory not found. Expected 'configs/' in project root."
    )


def list_available_workflows() -> None:
    """Lista todos os workflows disponíveis."""
    try:
        configs_dir = get_configs_dir()
        workflows_dir = configs_dir / "workflows"
        workflows = list_workflow_configs(str(workflows_dir))

        if not workflows:
            print("No workflows found in configs/workflows/")
            return

        print("\nAvailable workflows:")
        print("-" * 50)
        for workflow_id in workflows:
            try:
                workflow_path = workflows_dir / f"{workflow_id}.yaml"
                if not workflow_path.exists():
                    workflow_path = workflows_dir / f"{workflow_id}.yml"

                if workflow_path.exists():
                    config = load_workflow_config(str(workflow_path))
                    print(f"  ID: {config.id}")
                    print(f"  Name: {config.name}")
                    print(f"  Description: {config.description}")
                    print()
            except Exception as e:
                logger.warning("Error loading workflow %s: %s", workflow_id, e)
                print(f"  ID: {workflow_id} (error loading)")
                print()

    except Exception as e:
        logger.error("Error listing workflows: %s", e)
        sys.exit(1)


def main() -> None:
    """Função principal da aplicação."""
    parser = argparse.ArgumentParser(
        description="Declarative Multi-Agent System"
    )
    parser.add_argument(
        "--workflow",
        type=str,
        help="ID do workflow a ser carregado",
        default=os.getenv("WORKFLOW_ID"),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Lista todos os workflows disponíveis",
    )
    parser.add_argument(
        "--orchestrator-port",
        type=int,
        default=int(os.getenv("ORCHESTRATOR_PORT", "8004")),
        help="Porta do orquestrador",
    )

    args = parser.parse_args()

    # Listar workflows se solicitado
    if args.list:
        list_available_workflows()
        return

    # Validar workflow
    if not args.workflow:
        print("Error: --workflow ou WORKFLOW_ID deve ser especificado")
        print("\nUse --list para ver workflows disponíveis")
        sys.exit(1)

    try:
        configs_dir = get_configs_dir()
        workflows_dir = configs_dir / "workflows"

        # Tentar carregar o workflow
        workflow_path = workflows_dir / f"{args.workflow}.yaml"
        if not workflow_path.exists():
            workflow_path = workflows_dir / f"{args.workflow}.yml"

        if not workflow_path.exists():
            print(f"Error: Workflow '{args.workflow}' not found")
            print("\nUse --list para ver workflows disponíveis")
            sys.exit(1)

        # Carregar configuração do workflow
        workflow_config = load_workflow_config(str(workflow_path))
        logger.info(
            "Loaded workflow: %s (%s)", workflow_config.id, workflow_config.name
        )

        # Criar registry compartilhado
        registry = create_registry_config(workflow_config)
        logger.info("Created registry for team: %s", workflow_config.configuration.team_name)

        # Instanciar LLM client usando componente Dapr
        # O componente 'openai' deve estar configurado em components/openai.yaml
        llm = DaprChatClient(component_name="openai")
        logger.info("Initialized Dapr LLM client (component: openai)")

        # Carregar e criar agentes
        agent_specs = load_agent_specs(workflow_config, base_path=str(configs_dir))
        agents: list[DurableAgent] = []

        for spec in agent_specs:
            agent = create_agent_from_spec(spec, workflow_config, registry, llm)
            agent.start()
            agents.append(agent)
            logger.info("Started agent: %s", agent.name)

        # Criar orquestrador LLM
        orchestrator_pubsub = AgentPubSubConfig(
            pubsub_name=workflow_config.configuration.pubsub_name,
            agent_topic=os.getenv(
                "ORCHESTRATOR_TOPIC", f"{workflow_config.id}.orchestrator.requests"
            ),
            broadcast_topic=workflow_config.configuration.broadcast_topic,
        )

        orchestrator_state = AgentStateConfig(
            store=StateStoreService(
                store_name=workflow_config.configuration.workflow_state_store,
                key_prefix=f"{workflow_config.id}.orchestrator:",
            ),
        )

        orchestrator_execution = AgentExecutionConfig(
            max_iterations=int(os.getenv("MAX_ITERATIONS", "10"))
        )

        def on_summary(summary: str):
            print("\n" + "=" * 60)
            print("Workflow Summary:")
            print("=" * 60)
            print(summary)
            print("=" * 60)

        # Criar orquestrador com metadata incluindo o objective
        orchestrator = LLMOrchestrator(
            name=f"{workflow_config.id}-orchestrator",
            llm=llm,
            pubsub=orchestrator_pubsub,
            state=orchestrator_state,
            registry=registry,
            execution=orchestrator_execution,
            agent_metadata={
                "type": "LLMOrchestrator",
                "description": f"LLM-driven Orchestrator for {workflow_config.name}",
                "workflow_id": workflow_config.id,
                "objective": workflow_config.objective,
            },
            timeout_seconds=int(os.getenv("TIMEOUT_SECONDS", "45")),
            runtime=wf.WorkflowRuntime(),
            final_summary_callback=on_summary,
        )
        orchestrator.start()
        logger.info("Started orchestrator: %s", orchestrator.name)

        # Registrar rotas e aguardar
        runner = AgentRunner()
        try:
            # Registrar rotas para todos os agentes
            for agent in agents:
                runner.register_routes(agent)

            # Servir o orquestrador
            runner.serve(orchestrator, port=args.orchestrator_port)
            logger.info("Application started. Orchestrator available on port %d", args.orchestrator_port)
            logger.info("Workflow objective: %s", workflow_config.objective[:100] + "...")
            # runner.serve() blocks until shutdown
        finally:
            logger.info("Shutting down...")
            runner.shutdown()
            orchestrator.stop()
            for agent in agents:
                agent.stop()
            logger.info("Shutdown complete")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception("Error running application: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()

