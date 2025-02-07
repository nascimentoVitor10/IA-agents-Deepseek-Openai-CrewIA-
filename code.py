# instalação do ollama, deepseek e crewai pelo terminal linux
"""
curl -fsSL https://ollama.com/install.sh|sh
ollama pull deeepseek-r1:8b
pip install crewai crewai_tools
"""

# criação de agentes de IA

"""
temperature - grau de imprevisibilidade nas respostas

role - definição da função/expertise do agente
goal - objetivo individual que guiará a tomada de decisão do agente
backstory - história de fundo para prover contexto de comportamento e interação

description - descrição detalhada do que se trata a tarefa
expected_output - saída esperada para a tarefa
agent - agente responsável pela execução da tarefa
"""

from crewai import LLM, Agent, Task, Crew
from pydantic import BaseModel
import os


class TaskFormat(BaseModel):
    task_overview: str
    task_description: str

class TaskOutput(BaseModel):
    tasks : list(TaskFormat)

gemini = LLM(
    model="gemini/gemini-1.5-flash",
    temperature=0.5,
    api_key=os.getenv("GEMINI_API_KEY")
)

deepseek = LLM(
    model="ollama/deepseek-r1:8b",
    temperature=0.5,
    base_url="http://localhost:11434"
)

# agentes instanciados
especialista_eventos = Agent(
    role="Especialista em planejamento de palestras sobre o deepseek.",
    goal="criar um plano detalhado para organizar uma plaestra de sucesso.",
    backstory="""Você é um especialista em planejamento de eventos com anos de
    experiência em organização de lives de 1 a 2 horas de duração e tem apenas
    uma semana de prazo para realização da live.
    """,
    llm=deepseek
)

revisor = Agent(
    role="Revisor de planejamento",
    goal="revisar o planejamento da palestra",
    backstory="""
    Você é um agente especialista em planejar eventos, que atua como revisor de planejamento feito
    por outras pessoas. Você fornece feedbacks sobre os planejamentos de outros agentes e propõe se
    necessário, melhorias.
    """,
    llm=deepseek
)


lista_tarefas_task = Task(
    description="Crie uma lista de tarefas detalhadas para organizar um evento corporativo",
    expected_output="uma lista detalhada de tarefas, incluindo prazos e responsáveis para a organização dp evento corporativo",
    agent=especialista_eventos
)

revisar_lista_tarefas = Task(
    description="revisar a lista de tarefas gerada por seu companheiro agente",
    expected_output="um d=feedback de pontos que podem ser melhorados no planjemaneto de eventos",
    agent=revisor,
    context=[lista_tarefas_task],
    output_pydantic=TaskOutput
)


crew = Crew(
    agents=[especialista_eventos, revisor],
    tasks=[lista_tarefas_task, revisar_lista_tarefas],
    verbose=True
)

resultado = crew.kickoff()

# Ferramentas (TOOLS)
"""
ferramentas são funções que agentes conseguem evocar que o permitem
interagir com o ambiente, coletar dados, escrevendo novas informações...

- leitor de PDF
- leitor de descrição de vídeos do youtube
- leitor de bases de dados PostgreeSQL
"""
