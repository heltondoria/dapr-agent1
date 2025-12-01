# Dapr Multi-Agent Orchestrator

Sistema multi-agente declarativo usando Dapr para orquestração de workflows com agentes LLM colaborativos.

## Visão Geral

Este projeto implementa um sistema de múltiplos agentes que trabalham juntos para resolver tarefas complexas. O sistema é composto por:

- **Orchestrator**: Coordena os agentes usando um LLM para decidir qual agente deve executar cada tarefa
- **Worker**: Processa tarefas dos agentes dinâmicos definidos em YAML
- **Agents**: Personagens com perfis únicos (Frodo, Sam, Gandalf, Legolas) que colaboram em missões

## Pré-requisitos

- Python 3.14+
- [Dapr CLI](https://docs.dapr.io/getting-started/install-dapr-cli/) instalado
- Redis (para state stores e pub/sub)
- Chave da API OpenAI

## Instalação

```bash
# Clonar o repositório
cd dapr-agent1

# Instalar dependências com uv
uv sync

# Copiar exemplo de secrets e configurar
cp secrets.json.example secrets.json
# Editar secrets.json com sua chave OpenAI
```

## Configuração

### Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
OPENAI_API_KEY=sk-sua-chave-aqui
OPENAI_MODEL=gpt-4o
MAX_ITERATIONS=15
TIMEOUT_SECONDS=45
```

### Secrets (secrets.json)

```json
{
  "openAIKey": "sk-sua-chave-aqui"
}
```

## Executando

### Iniciar todos os serviços com Dapr

```bash
dapr run -f .
```

Isso iniciará:

- **Orchestrator** na porta `8004`
- **AgentWorker** na porta `8005`

### Listar workflows disponíveis

```bash
python -m dapr_agent_1.orchestrator --list
```

## API Endpoints

### Orchestrator (porta 8004)

#### POST /run - Iniciar uma missão

Inicia um novo workflow/missão e retorna imediatamente (processamento assíncrono).

```bash
curl -X POST http://localhost:8004/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Sua missão aqui"}'
```

**Resposta:**

```json
{
  "instance_id": "abc123def456",
  "status": "scheduled",
  "status_url": "/run/abc123def456"
}
```

#### GET /run/{instance_id} - Consultar status

Consulta o status de um workflow em execução ou finalizado.

```bash
curl http://localhost:8004/run/{instance_id}
```

**Resposta:**

```json
{
  "instance_id": "abc123def456",
  "status": "COMPLETED",
  "created_at": "2024-12-01T00:00:00.000000",
  "last_updated_at": "2024-12-01T00:05:00.000000",
  "input": "{\"task\": \"...\"}",
  "output": "..."
}
```

### Worker (porta 8005)

#### GET /health - Health check

```bash
curl http://localhost:8005/health
```

**Resposta:**

```json
{
  "status": "healthy",
  "agents": "4"
}
```

#### GET /agents - Listar agentes

```bash
curl http://localhost:8005/agents
```

**Resposta:**

```json
{
  "workflow": "fellowship",
  "agents": ["frodo", "sam", "gandalf", "legolas"]
}
```

## Exemplos de Missões

### Missão 1: Jornada à Montanha da Perdição

```bash
curl -X POST http://localhost:8004/run \
  -H "Content-Type: application/json" \
  -d '{
    "task": "A Sociedade do Anel precisa planejar a jornada até a Montanha da Perdição para destruir o Um Anel. Frodo é o portador do Anel e deve ser protegido a todo custo. Sam é responsável pela logística e provisões. Gandalf deve fornecer conhecimento sobre os perigos do caminho e orientação estratégica. Legolas deve avaliar as rotas mais seguras e vigiar contra ameaças. Trabalhem juntos para criar um plano detalhado de viagem, incluindo: rotas possíveis, provisões necessárias, perigos esperados e estratégias de defesa."
  }'
```

### Missão 2: Conselho de Elrond

```bash
curl -X POST http://localhost:8004/run \
  -H "Content-Type: application/json" \
  -d '{
    "task": "O Conselho de Elrond foi convocado em Valfenda. O Um Anel foi encontrado e uma decisão crucial deve ser tomada. Gandalf deve explicar a história e o poder do Anel. Frodo deve decidir se aceita o fardo de ser o Portador. Sam deve avaliar o que será necessário para a jornada. Legolas deve compartilhar notícias das terras élficas e avaliar ameaças. Discutam: Quem deve carregar o Anel? Qual o melhor caminho a seguir? Quais são os maiores perigos? Como a Sociedade pode ter sucesso onde outros falharam?"
  }'
```

### Missão 3: Travessia de Moria

```bash
curl -X POST http://localhost:8004/run \
  -H "Content-Type: application/json" \
  -d '{
    "task": "A Sociedade está bloqueada na Passagem de Caradhras devido a uma tempestade. Devem decidir: tentar as Minas de Moria ou buscar outro caminho? Gandalf conhece os perigos de Moria mas também seus segredos. Frodo sente o peso do Anel aumentar. Sam está preocupado com as provisões que diminuem. Legolas percebe que estão sendo observados. Avaliem juntos: Quais são os riscos de Moria versus outros caminhos? O que sabemos sobre o que habita nas profundezas? Temos provisões suficientes para uma rota mais longa? Tomem uma decisão e planejem os próximos passos."
  }'
```

### Missão 4: Preparação em Valfenda

```bash
curl -X POST http://localhost:8004/run \
  -H "Content-Type: application/json" \
  -d '{
    "task": "A Sociedade está em Valfenda se preparando para partir. Cada membro deve contribuir com sua expertise para garantir que estejam prontos. Gandalf deve reunir informações e mapas sobre as terras à frente. Frodo precisa se preparar mentalmente para carregar o Anel e aprender a resistir sua influência. Sam deve organizar provisões, equipamentos de acampamento e suprimentos para a jornada. Legolas deve verificar armas, preparar flechas e identificar os melhores pontos de observação na rota inicial. Criem uma lista completa de preparativos e verifiquem se todos estão prontos para partir ao amanhecer."
  }'
```

## Executando Múltiplas Missões em Paralelo

O sistema suporta execução de múltiplas missões simultaneamente:

```bash
# Terminal 1: Iniciar missão 1
curl -X POST http://localhost:8004/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Planejar a jornada até a Montanha da Perdição."}'

# Terminal 2: Iniciar missão 2 (ao mesmo tempo)
curl -X POST http://localhost:8004/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Organizar o Conselho de Elrond."}'

# Acompanhar status de cada missão
curl http://localhost:8004/run/{instance_id_1}
curl http://localhost:8004/run/{instance_id_2}
```

## Estrutura do Projeto

```shell
dapr-agent1/
├── components/           # Componentes Dapr (state stores, pub/sub, etc.)
│   ├── local-secret-store.yaml
│   ├── memorystore.yaml
│   ├── openai.yaml
│   ├── pubsub.yaml
│   ├── registrystore.yaml
│   └── statestore.yaml
├── configs/
│   ├── agents/          # Definições dos agentes em YAML
│   │   ├── frodo.yaml
│   │   ├── gandalf.yaml
│   │   ├── legolas.yaml
│   │   └── sam.yaml
│   └── workflows/       # Definições dos workflows
│       └── fellowship.yaml
├── src/
│   └── dapr_agent_1/
│       ├── orchestrator.py  # Orquestrador LLM
│       ├── worker.py        # Worker de agentes dinâmicos
│       ├── loader.py        # Carregador de configs YAML
│       ├── schemas.py       # Schemas Pydantic
│       └── factory.py       # Factory de agentes
├── dapr.yaml            # Configuração multi-app Dapr
├── secrets.json         # Secrets (não commitar!)
├── .env                 # Variáveis de ambiente
└── pyproject.toml       # Dependências Python
```

## Agentes Disponíveis

| Agente | Nome | Papel | Especialidade |
|--------|------|-------|---------------|
| frodo | Frodo Baggins | Hobbit & Ring-bearer | Portador do Anel, decisões cruciais |
| sam | Samwise Gamgee | Logistics & Provisions | Logística, provisões, moral |
| gandalf | Gandalf | Wizard & Loremaster | Conhecimento, estratégia, magia |
| legolas | Legolas | Elf Scout & Marksman | Reconhecimento, combate à distância |

## Troubleshooting

### Erro: "No running event loop"

Certifique-se de estar usando Python 3.14+ e que o Dapr está rodando corretamente.

### Erro: "Unsupported format type: dapr"

O sistema usa `OpenAIChatClient` diretamente ao invés de `DaprChatClient` para suportar function calling.

### Agentes não respondem

Verifique se o AgentWorker está rodando e escutando na porta 8005:

```bash
curl http://localhost:8005/health
```

### Missões bloqueadas

O sistema suporta múltiplas missões em paralelo. Se uma missão parece bloqueada, verifique o status:

```bash
curl http://localhost:8004/run/{instance_id}
```

## Licença

MIT
