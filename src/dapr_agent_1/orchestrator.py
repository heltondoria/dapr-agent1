"""Orquestrador LLM para sistema multi-agente declarativo.

Este módulo contém apenas a lógica do orquestrador. Os agentes são
processados pelo DynamicAgentWorker em um processo separado.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import signal
import sys
import uuid
from pathlib import Path
from typing import Any

import dapr.ext.workflow as wf  # type: ignore[import-untyped]
import uvicorn
from dapr_agents.agents.configs import (  # type: ignore[import-untyped]
    AgentExecutionConfig,
    AgentPubSubConfig,
    AgentStateConfig,
)
from dapr_agents.agents.orchestrators.llm import LLMOrchestrator  # type: ignore[import-untyped]
from dapr_agents.llm.openai import OpenAIChatClient  # type: ignore[import-untyped]
from dapr_agents.storage.daprstores.stateservice import StateStoreService  # type: ignore[import-untyped]
from dapr_agents.workflow.runners import AgentRunner  # type: ignore[import-untyped]
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from .loader import create_registry_config, list_workflow_configs, load_workflow_config

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("dapr-orchestrator")


def get_configs_dir() -> Path:
    """Retorna o diretório de configurações."""
    script_dir = Path(__file__).parent.parent.parent
    configs_dir = script_dir / "configs"
    if configs_dir.exists():
        return configs_dir

    configs_dir = Path.cwd() / "configs"
    if configs_dir.exists():
        return configs_dir

    raise FileNotFoundError("Configs directory not found. Expected 'configs/' in project root.")


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


async def async_main(args: argparse.Namespace) -> None:
    """Função principal assíncrona do orquestrador."""
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
    logger.info("Loaded workflow: %s (%s)", workflow_config.id, workflow_config.name)

    # Criar registry compartilhado (usado para descobrir agentes)
    registry = create_registry_config(workflow_config)
    logger.info("Created registry for team: %s", workflow_config.configuration.team_name)

    # Instanciar LLM client OpenAI (suporta function calling para o orquestrador)
    llm = OpenAIChatClient(model=os.getenv("OPENAI_MODEL", "gpt-4o"))
    logger.info("Initialized OpenAI LLM client (model: %s)", llm.model)

    # Criar configuração de Pub/Sub do orquestrador
    orchestrator_pubsub = AgentPubSubConfig(
        pubsub_name=workflow_config.configuration.pubsub_name,
        agent_topic=os.getenv("ORCHESTRATOR_TOPIC", f"{workflow_config.id}.orchestrator.requests"),
        broadcast_topic=workflow_config.configuration.broadcast_topic,
    )

    # Criar configuração de estado do orquestrador
    orchestrator_state = AgentStateConfig(
        store=StateStoreService(
            store_name=workflow_config.configuration.workflow_state_store,
            key_prefix=f"{workflow_config.id}.orchestrator:",
        ),
    )

    # Criar configuração de execução
    orchestrator_execution = AgentExecutionConfig(max_iterations=int(os.getenv("MAX_ITERATIONS", "15")))

    def on_summary(summary: str) -> None:
        """Callback chamado quando o workflow é finalizado."""
        print("\n" + "=" * 60)
        print("Workflow Summary:")
        print("=" * 60)
        print(summary)
        print("=" * 60)

    # Criar orquestrador LLM
    # O orquestrador se comunica com os agentes via pub/sub
    # Os agentes são processados pelo DynamicAgentWorker
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

    # Criar FastAPI app e runner
    app = FastAPI(title="Dapr Orchestrator Service", version="1.0.0")
    runner = AgentRunner()

    # Criar workflow client próprio para evitar o lock compartilhado do runner
    workflow_client = wf.DaprWorkflowClient()

    # Descobrir o workflow entry do orchestrator
    workflow_entry = runner.discover_entry(orchestrator)
    logger.info("Discovered workflow entry: %s", workflow_entry.__name__)

    # Endpoint customizado /run - agenda workflow sem lock bloqueante
    @app.post("/run", tags=["workflow"])
    async def start_workflow(body: dict[str, Any] | None = None) -> dict[str, Any]:
        body = body or {}
        """
        Inicia uma nova missão/workflow.

        Retorna imediatamente com o instance_id, permitindo múltiplas
        missões em paralelo.
        """
        instance_id = uuid.uuid4().hex

        # Agendar workflow diretamente (sem usar o runner que tem lock)
        try:
            workflow_client.schedule_new_workflow(
                workflow=workflow_entry,
                input=body,
                instance_id=instance_id,
            )
            logger.info("Scheduled workflow instance: %s", instance_id)
        except Exception as e:
            logger.exception("Failed to schedule workflow: %s", e)
            raise HTTPException(status_code=500, detail=str(e)) from e

        return {
            "instance_id": instance_id,
            "status": "scheduled",
            "status_url": f"/run/{instance_id}",
        }

    # Endpoint customizado /run/{instance_id} - consulta status
    @app.get("/run/{instance_id}", tags=["workflow"])
    async def get_workflow_status(instance_id: str) -> dict[str, Any]:
        """Consulta o status de um workflow."""
        try:
            state = await asyncio.to_thread(
                workflow_client.get_workflow_state,
                instance_id,
                fetch_payloads=True,
            )
        except Exception as e:
            logger.exception("Failed to get workflow state: %s", e)
            raise HTTPException(status_code=500, detail=str(e)) from e

        if state is None:
            raise HTTPException(status_code=404, detail="Workflow instance not found")

        return {
            "instance_id": instance_id,
            "status": getattr(state.runtime_status, "name", str(state.runtime_status)),
            "created_at": state.created_at.isoformat() if state.created_at else None,
            "last_updated_at": state.last_updated_at.isoformat() if state.last_updated_at else None,
            "input": state.serialized_input,
            "output": state.serialized_output,
        }

    # Configurar rotas pub/sub do runner (sem expor o endpoint /run padrão)
    # Usar delivery_mode="async" para permitir processamento paralelo de missões
    runner.serve(
        orchestrator,
        app=app,
        port=args.port,
        expose_entry=False,  # Não usar endpoint padrão que tem lock bloqueante
        delivery_mode="async",
        queue_maxsize=1024,
    )

    logger.info("Orchestrator available on port %d", args.port)
    logger.info("Custom /run endpoint enabled for parallel workflow scheduling")
    logger.info("Workflow objective: %s", workflow_config.objective[:100] + "...")
    logger.info("NOTE: Agents are processed by DynamicAgentWorker in a separate process")

    # Configurar servidor uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=args.port, log_level="info")
    server = uvicorn.Server(config)

    # Configurar sinais de shutdown
    shutdown_event = asyncio.Event()

    def signal_handler() -> None:
        logger.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, signal_handler)

    # Iniciar servidor em tarefa separada
    server_task = asyncio.create_task(server.serve())

    try:
        # Aguardar sinal de shutdown ou servidor terminar
        _, pending = await asyncio.wait(
            [server_task, asyncio.create_task(shutdown_event.wait())],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancelar tarefas pendentes
        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    finally:
        logger.info("Shutting down orchestrator...")
        runner.shutdown()
        orchestrator.stop()
        logger.info("Shutdown complete")


def main() -> None:
    """Função principal do orquestrador."""
    parser = argparse.ArgumentParser(description="LLM Orchestrator for Multi-Agent System")
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
        "--port",
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
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception("Error running orchestrator: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
