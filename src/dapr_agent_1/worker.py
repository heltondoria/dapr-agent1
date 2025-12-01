"""Dynamic Agent Worker - Processa múltiplos agentes dinâmicos em um único processo."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import dapr.ext.workflow as wf  # type: ignore[import-untyped]
import uvicorn
from dapr.clients import DaprClient  # type: ignore[import-untyped]
from dapr.ext.workflow import DaprWorkflowContext, WorkflowActivityContext  # type: ignore[import-untyped]
from dapr_agents.agents.base import AgentBase  # type: ignore[import-untyped]
from dapr_agents.agents.configs import (  # type: ignore[import-untyped]
    AgentMemoryConfig,
    AgentProfileConfig,
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
)
from dapr_agents.agents.schemas import (  # type: ignore[import-untyped]
    AgentTaskResponse,
    BroadcastMessage,
    TriggerAction,
)
from dapr_agents.llm.chat import ChatClientBase  # type: ignore[import-untyped]
from dapr_agents.llm.openai import OpenAIChatClient  # type: ignore[import-untyped]
from dapr_agents.memory import ConversationDaprStateMemory  # type: ignore[import-untyped]
from dapr_agents.storage.daprstores.stateservice import StateStoreService  # type: ignore[import-untyped]
from dapr_agents.types import AssistantMessage, LLMChatResponse  # type: ignore[import-untyped]
from dapr_agents.types.workflow import PubSubRouteSpec  # type: ignore[import-untyped]
from dapr_agents.workflow.utils.pubsub import (  # type: ignore[import-untyped]
    broadcast_message,
    send_message_to_agent,
)
from dapr_agents.workflow.utils.registration import (  # type: ignore[import-untyped]
    register_message_routes,
)
from dotenv import load_dotenv
from fastapi import FastAPI

from .loader import create_registry_config, load_agent_specs, load_workflow_config
from .schemas import AgentSpec, WorkflowConfig

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("dapr-agent-worker")


class DynamicAgent(AgentBase):
    """
    Agente dinâmico baseado em AgentBase.

    Não usa DurableAgent para evitar conflitos de registro de workflow.
    O comportamento de agente vem do AgentBase, mas a durabilidade é
    gerenciada pelo DynamicAgentWorker.
    """

    def __init__(
        self,
        *,
        spec: AgentSpec,
        workflow_config: WorkflowConfig,
        registry: AgentRegistryConfig,
        llm: ChatClientBase,
    ) -> None:
        """
        Inicializa um agente dinâmico a partir de uma especificação.

        Args:
            spec: Especificação do agente (carregada de YAML)
            workflow_config: Configuração do workflow
            registry: Registry compartilhado
            llm: Cliente LLM
        """
        # Criar configurações a partir da spec
        profile = AgentProfileConfig(
            name=spec.profile.name,
            role=spec.profile.role,
            goal=spec.profile.goal,
            instructions=list(spec.profile.instructions),
            style_guidelines=list(spec.profile.style_guidelines),
        )

        pubsub_name = spec.pubsub.pubsub_name or workflow_config.configuration.pubsub_name
        broadcast_topic = spec.pubsub.broadcast_topic or workflow_config.configuration.broadcast_topic

        pubsub = AgentPubSubConfig(
            pubsub_name=pubsub_name,
            agent_topic=spec.pubsub.agent_topic,
            broadcast_topic=broadcast_topic,
        )

        state_store_name = spec.state.store_name or workflow_config.configuration.workflow_state_store
        state_store = StateStoreService(store_name=state_store_name, key_prefix=spec.state.key_prefix)
        state = AgentStateConfig(store=state_store)

        memory_store_name = spec.memory.store_name or workflow_config.configuration.memory_store
        memory_store = ConversationDaprStateMemory(
            store_name=memory_store_name,
            session_id=spec.memory.session_id,
        )
        memory = AgentMemoryConfig(store=memory_store)

        super().__init__(
            profile=profile,
            pubsub=pubsub,
            state=state,
            registry=registry,
            memory=memory,
            llm=llm,
        )

        self._spec = spec
        logger.info("Created dynamic agent: %s (%s)", self.name, spec.profile.role)

    @property
    def spec(self) -> AgentSpec:
        """Retorna a especificação original do agente."""
        return self._spec

    async def process_task(self, task: str, instance_id: str) -> AssistantMessage:
        """
        Processa uma tarefa e retorna a resposta.

        Args:
            task: Tarefa a ser processada
            instance_id: ID da instância do workflow

        Returns:
            Mensagem de resposta do assistente
        """
        # Carregar estado
        try:
            self.load_state()
        except Exception:
            logger.debug("State load failed, using defaults", exc_info=True)

        # Construir mensagens
        chat_history = self._reconstruct_conversation_history(instance_id)
        messages = self.prompting_helper.build_initial_messages(
            user_input=task,
            chat_history=chat_history,
        )

        # Executar loop de conversação
        final_reply = await self._conversation_loop(instance_id, messages, task)
        return final_reply

    async def _conversation_loop(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        task: str,
    ) -> AssistantMessage:
        """Loop de conversação com suporte a tool calls."""
        pending_messages = list(messages)
        final_reply: AssistantMessage | None = None

        for turn in range(1, self.execution.max_iterations + 1):
            logger.debug("Agent %s turn %d/%d", self.name, turn, self.execution.max_iterations)

            response: LLMChatResponse = self.llm.generate(
                messages=pending_messages,
                tools=self.get_llm_tools(),
            )

            assistant_message = response.get_message()
            if assistant_message is None:
                raise RuntimeError("LLM returned no assistant message")

            assistant_dict = assistant_message.model_dump()
            self._save_assistant_message(instance_id, assistant_dict)
            self.text_formatter.print_message(assistant_dict)

            if assistant_message.has_tool_calls():
                tool_calls = assistant_message.get_tool_calls()
                if tool_calls:
                    pending_messages.append(assistant_dict)
                    tool_msgs = await self._execute_tool_calls(instance_id, tool_calls)
                    pending_messages.extend(tool_msgs)
                    continue

            final_reply = assistant_message
            break
        else:
            # Max iterations reached
            content = "I reached the maximum number of reasoning steps."
            final_reply = AssistantMessage(role="assistant", content=content)
            logger.warning("Agent %s hit max iterations", self.name)

        self.save_state()
        return final_reply

    async def _execute_tool_calls(
        self,
        instance_id: str,
        tool_calls: list[Any],
    ) -> list[dict[str, Any]]:
        """Executa tool calls em paralelo."""
        results = []
        for tool_call in tool_calls:
            fn_name = tool_call.function.name
            fn_args = tool_call.function.arguments_dict

            try:
                result = await self.tool_executor.run_tool(fn_name, **fn_args)
                serialized = json.dumps(result) if not isinstance(result, str) else result
            except Exception as e:
                serialized = f"Error: {e}"
                logger.error("Tool %s failed: %s", fn_name, e)

            tool_msg = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": fn_name,
                "content": serialized,
            }
            results.append(tool_msg)
            self.text_formatter.print_message(tool_msg)

        return results


class DynamicAgentWorker:
    """
    Worker que processa múltiplos agentes dinâmicos.

    Registra um workflow genérico e roteia mensagens para os agentes
    corretos baseado no tópico pub/sub.
    """

    def __init__(
        self,
        workflow_config: WorkflowConfig,
        llm: ChatClientBase,
    ) -> None:
        """
        Inicializa o worker.

        Args:
            workflow_config: Configuração do workflow
            llm: Cliente LLM compartilhado
        """
        self._workflow_config = workflow_config
        self._llm = llm
        self._registry = create_registry_config(workflow_config)
        self._agents: dict[str, DynamicAgent] = {}
        self._agent_topics: dict[str, str] = {}  # topic -> agent_name
        self._runtime = wf.WorkflowRuntime()
        self._wf_client = wf.DaprWorkflowClient()
        self._dapr_client: DaprClient | None = None
        self._pubsub_closers: list[Callable[[], None]] = []
        self._started = False

        logger.info(
            "DynamicAgentWorker initialized for workflow: %s",
            workflow_config.id,
        )

    @property
    def workflow_config(self) -> WorkflowConfig:
        return self._workflow_config

    @property
    def agents(self) -> dict[str, DynamicAgent]:
        return self._agents

    def load_agents(self, configs_dir: str) -> None:
        """
        Carrega agentes a partir das especificações do workflow.

        Args:
            configs_dir: Diretório base das configurações
        """
        agent_specs = load_agent_specs(self._workflow_config, base_path=configs_dir)

        for spec in agent_specs:
            agent = DynamicAgent(
                spec=spec,
                workflow_config=self._workflow_config,
                registry=self._registry,
                llm=self._llm,
            )
            self._agents[spec.name] = agent
            self._agent_topics[spec.pubsub.agent_topic] = spec.name
            logger.info("Loaded agent: %s (topic: %s)", spec.name, spec.pubsub.agent_topic)

        logger.info("Loaded %d agents for workflow '%s'", len(self._agents), self._workflow_config.id)

    def get_agent_by_topic(self, topic: str) -> DynamicAgent | None:
        """Retorna o agente associado a um tópico."""
        agent_name = self._agent_topics.get(topic)
        if agent_name:
            return self._agents.get(agent_name)
        return None

    def get_agent_by_name(self, name: str) -> DynamicAgent | None:
        """Retorna o agente pelo nome."""
        # Busca por nome exato ou nome do perfil
        if name in self._agents:
            return self._agents[name]
        for agent in self._agents.values():
            if agent.name == name:
                return agent
        return None

    def register_agents_in_registry(self) -> None:
        """Registra todos os agentes no registry compartilhado."""
        for agent in self._agents.values():
            try:
                # Usar método público register_agentic_system() de AgentComponents
                # Este método já encapsula toda a lógica de registro corretamente
                agent.register_agentic_system()
                logger.info(
                    "Registered agent '%s' in team '%s'",
                    agent.name,
                    self._registry.team_name,
                )
            except Exception as e:
                logger.error("Failed to register agent '%s': %s", agent.name, e)

    def _register_workflows(self) -> None:
        """Registra o workflow genérico e atividades."""

        # Workflow principal para processar tarefas de agentes
        @self._runtime.workflow(name="dynamic_agent_workflow")  # type: ignore[misc]
        def dynamic_agent_workflow(ctx: DaprWorkflowContext, payload: dict[str, Any]):
            """Workflow genérico que processa tarefas de agentes dinâmicos."""
            agent_name = payload.get("agent_name")
            task = payload.get("task", "")
            instance_id = ctx.instance_id
            source = payload.get("source", "direct")
            trigger_instance_id = payload.get("trigger_instance_id")

            if not ctx.is_replaying:
                logger.info(
                    "Processing task for agent '%s' (instance: %s)",
                    agent_name,
                    instance_id,
                )

            # Executar a tarefa do agente
            result = yield ctx.call_activity(  # type: ignore[misc]
                process_agent_task,
                input={
                    "agent_name": agent_name,
                    "task": task,
                    "instance_id": instance_id,
                },
            )

            # Enviar resposta de volta se houver origem
            if source and source != "direct" and trigger_instance_id:
                yield ctx.call_activity(  # type: ignore[misc]
                    send_response_activity,
                    input={
                        "agent_name": agent_name,
                        "response": result,
                        "target_agent": source,
                        "target_instance_id": trigger_instance_id,
                    },
                )

            # Broadcast se configurado
            broadcast_topic = payload.get("broadcast_topic")
            if broadcast_topic:
                yield ctx.call_activity(  # type: ignore[misc]
                    broadcast_response_activity,
                    input={
                        "agent_name": agent_name,
                        "response": result,
                        "broadcast_topic": broadcast_topic,
                    },
                )

            return result

        # Atividade para processar tarefa
        @self._runtime.activity(name="process_agent_task")  # type: ignore[misc]
        def process_agent_task(
            ctx: WorkflowActivityContext,
            payload: dict[str, Any],
        ) -> dict[str, Any]:
            """Processa uma tarefa usando o agente apropriado."""
            agent_name = payload.get("agent_name", "")
            task = payload.get("task", "")
            instance_id = payload.get("instance_id", "")

            if not agent_name:
                return {"role": "assistant", "content": "Error: agent_name not provided"}

            agent = self.get_agent_by_name(agent_name)
            if not agent:
                logger.error("Agent not found: %s", agent_name)
                return {
                    "role": "assistant",
                    "content": f"Error: Agent '{agent_name}' not found",
                }

            try:
                # Executar tarefa de forma síncrona (activity é sync)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(agent.process_task(task, instance_id))
                finally:
                    loop.close()

                if hasattr(result, "model_dump"):
                    return result.model_dump()  # type: ignore[no-any-return]
                return {"role": "assistant", "content": str(result)}
            except Exception as e:
                logger.exception(
                    "Error processing task for agent '%s': %s",
                    agent_name,
                    e,
                )
                return {"role": "assistant", "content": f"Error: {e}"}

        # Atividade para enviar resposta
        @self._runtime.activity(name="send_response_activity")  # type: ignore[misc]
        def send_response_activity(
            ctx: WorkflowActivityContext,
            payload: dict[str, Any],
        ) -> None:
            """Envia resposta de volta ao agente que originou a tarefa."""
            agent_name = payload.get("agent_name", "")
            response = payload.get("response", {})
            target_agent = payload.get("target_agent", "")
            target_instance_id = payload.get("target_instance_id", "")

            if not target_agent or not target_instance_id or not agent_name:
                return

            agent = self.get_agent_by_name(agent_name)
            if not agent:
                return

            response["role"] = "user"
            response["name"] = agent.name
            response["workflow_instance_id"] = target_instance_id

            agent_response = AgentTaskResponse(**response)
            agents_metadata = agent.get_agents_metadata()
            source_name = agent.name or ""

            async def _send() -> None:
                await send_message_to_agent(
                    source=source_name,
                    target_agent=target_agent,
                    message=agent_response,
                    agents_metadata=agents_metadata,
                )

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_send())
            except Exception as e:
                logger.exception("Failed to send response: %s", e)
            finally:
                loop.close()

        # Atividade para broadcast
        @self._runtime.activity(name="broadcast_response_activity")  # type: ignore[misc]
        def broadcast_response_activity(
            ctx: WorkflowActivityContext,
            payload: dict[str, Any],
        ) -> None:
            """Broadcast da resposta para todos os agentes."""
            agent_name = payload.get("agent_name", "")
            response = payload.get("response", {})
            broadcast_topic = payload.get("broadcast_topic", "")

            if not broadcast_topic or not agent_name:
                return

            agent = self.get_agent_by_name(agent_name)
            if not agent:
                return

            response["role"] = "user"
            response["name"] = agent.name

            broadcast_msg = BroadcastMessage(**response)
            agents_metadata = agent.get_agents_metadata()
            source_name = agent.name or ""
            message_bus = agent.message_bus_name or ""

            async def _broadcast() -> None:
                await broadcast_message(
                    message=broadcast_msg,
                    broadcast_topic=broadcast_topic,
                    message_bus=message_bus,
                    source=source_name,
                    agents_metadata=agents_metadata,
                )

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_broadcast())
            except Exception as e:
                logger.exception("Failed to broadcast: %s", e)
            finally:
                loop.close()

        # Armazenar referência ao workflow para uso no scheduler
        self._dynamic_workflow = dynamic_agent_workflow
        logger.info("Registered dynamic_agent_workflow and activities")

    def _create_pubsub_handler(
        self,
        agent_name: str,
    ) -> Callable[[Any, dict[str, Any]], dict[str, Any]]:
        """
        Cria um handler pub/sub para um agente específico.

        O handler recebe mensagens TriggerAction e agenda o workflow
        dynamic_agent_workflow para processar a tarefa.
        """

        def handler(
            ctx: DaprWorkflowContext,
            message: dict[str, Any],
        ) -> dict[str, Any]:
            """Handler que roteia mensagens para o workflow do agente."""
            task = message.get("task", "")
            metadata = message.get("_message_metadata", {}) or {}
            source = metadata.get("source", "direct")
            trigger_instance_id = message.get("workflow_instance_id")

            agent = self.get_agent_by_name(agent_name)
            broadcast_topic = agent.broadcast_topic_name if agent else None

            # Agendar o workflow
            return {
                "agent_name": agent_name,
                "task": task,
                "source": source,
                "trigger_instance_id": trigger_instance_id,
                "broadcast_topic": broadcast_topic,
            }

        # Marcar com metadata do message_router
        setattr(  # noqa: B010
            handler,
            "_message_router_data",
            {
                "message_schemas": [TriggerAction],
                "pubsub": self._workflow_config.configuration.pubsub_name,
                "topic": self._agent_topics.get(agent_name),
            },
        )
        handler.__name__ = f"handler_{agent_name}"

        return handler

    def _register_pubsub_routes(self) -> None:
        """Registra rotas pub/sub para todos os agentes."""
        if self._dapr_client is None:
            self._dapr_client = DaprClient()

        routes: list[PubSubRouteSpec] = []

        for agent_name, agent in self._agents.items():
            topic = agent.agent_topic_name
            if not topic:
                logger.warning("Agent '%s' has no topic configured", agent_name)
                continue

            # Criar spec de rota com closure para capturar variáveis
            handler = self._make_agent_handler(agent_name, topic)

            routes.append(
                PubSubRouteSpec(
                    pubsub_name=self._workflow_config.configuration.pubsub_name,
                    topic=topic,
                    handler_fn=handler,
                    message_model=TriggerAction,
                )
            )

            logger.info(
                "Created pub/sub route for agent '%s' on topic '%s'",
                agent_name,
                topic,
            )

        if routes:
            # Criar scheduler customizado que extrai agent_name do handler
            def custom_scheduler(
                workflow_callable: Callable[..., Any],
                wf_input: dict[str, Any],
            ) -> str | None:
                """Scheduler que extrai agent_name e usa dynamic_agent_workflow."""
                # Extrair agent_name do nome do handler (agent_entry_{name})
                handler_name = getattr(workflow_callable, "__name__", "")
                agent_name = None
                if handler_name.startswith("agent_entry_"):
                    agent_name = handler_name[len("agent_entry_"):]

                if not agent_name:
                    logger.warning("Could not extract agent_name from handler: %s", handler_name)

                # Buscar agent pelo nome para obter broadcast_topic
                agent = self.get_agent_by_name(agent_name) if agent_name else None
                broadcast_topic = agent.broadcast_topic_name if agent else None

                # Extrair metadata
                metadata = wf_input.get("_message_metadata", {}) or {}
                source = metadata.get("source", "direct")
                trigger_instance_id = wf_input.get("workflow_instance_id")

                # Construir input completo com agent_name
                full_input = {
                    "agent_name": agent_name,
                    "task": wf_input.get("task", ""),
                    "source": source,
                    "trigger_instance_id": trigger_instance_id,
                    "broadcast_topic": broadcast_topic,
                }

                return self._wf_client.schedule_new_workflow(
                    workflow=self._dynamic_workflow,
                    input=full_input,
                )

            # Usar delivery_mode="async" para permitir processamento paralelo de tarefas
            # Tentar obter o loop rodando, ou criar um novo se necessário
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            closers = register_message_routes(
                dapr_client=self._dapr_client,
                routes=routes,
                wf_client=self._wf_client,
                scheduler=custom_scheduler,
                await_result=False,
                log_outcome=True,
                delivery_mode="async",
                queue_maxsize=1024,
                loop=loop,
            )
            self._pubsub_closers.extend(closers)
            logger.info("Registered %d pub/sub routes", len(routes))

    def _make_agent_handler(
        self,
        name: str,
        topic: str,
    ) -> Callable[[Any, dict[str, Any]], dict[str, Any]]:
        """Cria handler com closure para capturar nome e tópico do agente."""

        def workflow_handler(
            ctx: DaprWorkflowContext,
            message: dict[str, Any],
        ) -> dict[str, Any]:
            task = message.get("task", "")
            metadata = message.get("_message_metadata", {}) or {}
            source = metadata.get("source", "direct")
            trigger_instance_id = message.get("workflow_instance_id")

            ag = self.get_agent_by_name(name)
            broadcast_topic = ag.broadcast_topic_name if ag else None

            return {
                "agent_name": name,
                "task": task,
                "source": source,
                "trigger_instance_id": trigger_instance_id,
                "broadcast_topic": broadcast_topic,
            }

        # Registrar como workflow para o runtime
        setattr(workflow_handler, "_is_workflow_entry", True)  # noqa: B010
        setattr(  # noqa: B010
            workflow_handler,
            "_message_router_data",
            {
                "message_schemas": [TriggerAction],
                "pubsub": self._workflow_config.configuration.pubsub_name,
                "topic": topic,
            },
        )
        workflow_handler.__name__ = f"agent_entry_{name}"
        return workflow_handler

    def start(self) -> None:
        """Inicia o worker."""
        if self._started:
            raise RuntimeError("Worker already started")

        self._register_workflows()
        self._runtime.start()
        self._register_pubsub_routes()
        self._started = True
        logger.info("DynamicAgentWorker started")

    def stop(self) -> None:
        """Para o worker."""
        if not self._started:
            return

        # Fechar subscriptions pub/sub
        for closer in self._pubsub_closers:
            try:
                closer()
            except Exception:
                logger.debug("Error closing pub/sub subscription", exc_info=True)

        # Fechar runtime
        try:
            self._runtime.shutdown()
        except Exception:
            logger.debug("Error shutting down runtime", exc_info=True)

        # Fechar DaprClient
        if self._dapr_client:
            try:
                self._dapr_client.close()
            except Exception:
                logger.debug("Error closing Dapr client", exc_info=True)

        self._started = False
        logger.info("DynamicAgentWorker stopped")

    def schedule_agent_task(
        self,
        agent_name: str,
        task: str,
        source: str | None = None,
        trigger_instance_id: str | None = None,
    ) -> str:
        """
        Agenda uma tarefa para um agente.

        Args:
            agent_name: Nome do agente
            task: Tarefa a ser executada
            source: Origem da mensagem (para resposta)
            trigger_instance_id: ID da instância que originou

        Returns:
            ID da instância do workflow
        """
        client = wf.DaprWorkflowClient()

        agent = self.get_agent_by_name(agent_name)
        broadcast_topic = agent.broadcast_topic_name if agent else None

        instance_id: str = client.schedule_new_workflow(
            workflow=self._dynamic_workflow,
            input={
                "agent_name": agent_name,
                "task": task,
                "source": source,
                "trigger_instance_id": trigger_instance_id,
                "broadcast_topic": broadcast_topic,
            },
        )

        logger.info(
            "Scheduled task for agent '%s' (instance: %s)",
            agent_name,
            instance_id,
        )
        return instance_id


def get_configs_dir() -> Path:
    """Retorna o diretório de configurações."""
    script_dir = Path(__file__).parent.parent.parent
    configs_dir = script_dir / "configs"
    if configs_dir.exists():
        return configs_dir

    configs_dir = Path.cwd() / "configs"
    if configs_dir.exists():
        return configs_dir

    raise FileNotFoundError("Configs directory not found")


async def async_main(args: argparse.Namespace) -> None:
    """Função principal assíncrona do worker."""
    configs_dir = get_configs_dir()
    workflows_dir = configs_dir / "workflows"

    # Carregar workflow
    workflow_path = workflows_dir / f"{args.workflow}.yaml"
    if not workflow_path.exists():
        workflow_path = workflows_dir / f"{args.workflow}.yml"

    if not workflow_path.exists():
        print(f"Error: Workflow '{args.workflow}' not found")
        sys.exit(1)

    workflow_config = load_workflow_config(str(workflow_path))
    logger.info("Loaded workflow: %s (%s)", workflow_config.id, workflow_config.name)

    # Criar LLM client OpenAI (suporta function calling para os agentes)
    llm = OpenAIChatClient(model=os.getenv("OPENAI_MODEL", "gpt-4o"))
    logger.info("Initialized OpenAI LLM client (model: %s)", llm.model)

    # Criar e configurar worker
    worker = DynamicAgentWorker(workflow_config, llm)
    worker.load_agents(str(configs_dir))
    worker.register_agents_in_registry()
    worker.start()

    logger.info("Worker started with %d agents", len(worker.agents))
    for agent_name in worker.agents:
        logger.info("  - Agent: %s", agent_name)

    # Criar FastAPI app para health check (necessário para Dapr)
    app = FastAPI(title="Dapr Agent Worker", version="1.0.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "agents": str(len(worker.agents))}

    @app.get("/agents")
    async def list_agents() -> dict[str, Any]:
        """Lista os agentes disponíveis."""
        return {
            "workflow": workflow_config.id,
            "agents": list(worker.agents.keys()),
        }

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
        logger.info("Shutting down worker...")
        worker.stop()
        logger.info("Worker stopped")


def main() -> None:
    """Função principal do worker."""
    parser = argparse.ArgumentParser(description="Dynamic Agent Worker")
    parser.add_argument(
        "--workflow",
        type=str,
        required=True,
        help="ID do workflow a ser carregado",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("WORKER_PORT", "8005")),
        help="Porta do worker",
    )

    args = parser.parse_args()

    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception("Error running worker: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
