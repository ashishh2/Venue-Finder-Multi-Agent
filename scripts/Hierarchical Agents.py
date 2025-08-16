# ---
# jupyter:
#   jupytext:
#     formats: ipynb,../scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import warnings
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
os.environ['PIP_ROOT_USER_ACTION'] = 'ignore'

# Load environment variables from .env file
load_dotenv()

serper_api_key = os.getenv("SERPER_API_KEY")
os.environ["SERPER_API_KEY"] = serper_api_key

# %% [markdown]
# Imports essential modules from CrewAI and initializes two tools, one for web scraping and one for search engine interaction.

# %%
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, SerperDevTool

# Create a search tool
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# %% [markdown]
# Initializes a large language model (LLM) using the specified model.

# %%
from crewai import Agent, Task, Crew
from crewai.llm import LLM
from crewai_tools import SerperDevTool

# Run Local LLM to escape API quota limitations
llm = LLM(model="ollama/tinyllama")

# %% [markdown]
# Defines three agents with specific roles and goals: a Research Analyst, Report Writer, and Report Editor, each leveraging LLM and specialized tools to accomplish their tasks.

# %%
research_analyst_agent = Agent(
    role="Research Analyst",
    goal="Create and analyze research points to provide comprehensive insights on various topics.",
    backstory="Specializing in research analysis, this agent employs advanced methodologies to generate detailed research points and insights. With a deep understanding of research frameworks and a talent for synthesizing information, the Research Analyst Agent is instrumental in delivering thorough and actionable research outcomes.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool],
    llm=llm,
    max_iter=3
)

report_writer_agent = Agent(
    role="Report Writer",
    goal="Compile the analyzed data into a comprehensive and well-structured research report.",
    backstory="You are skilled at transforming complex information into clear, concise, and informative reports.",
    verbose=True,
    allow_delegation=True,
    llm=llm,
    max_iter=3
)

report_editor_agent = Agent(
    role="Report Editor",
    goal="Review and refine research reports to ensure clarity, accuracy, and adherence to standards.",
    backstory="With a keen eye for detail and a strong background in report editing, this agent ensures that research reports are polished, coherent, and meet high-quality standards. Skilled in revising content for clarity and consistency, the Report Editor Agent plays a critical role in finalizing research outputs.",
    verbose=True,
    llm=llm,
    max_iter=3
)

# %% [markdown]
# Defines tasks for data collection, analysis, report writing, and assessment, each assigned to the appropriate agent with clear descriptions and expected outputs.

# %%
# Define tasks
data_collection_task = Task(
    description=(
        "Collect data from relevant sources about the given {topic}."
        "Focus on identifying key trends, benefits, and challenges."
    ),
    expected_output=(
        "A comprehensive dataset that includes recent studies, statistics, and expert opinions."
    ),
    agent=research_analyst_agent,
)

data_analysis_task = Task(
    description=(
        "Analyze the collected data to identify key trends, benefits, and challenges for the {topic}."
    ),
    expected_output=(
        "A detailed analysis report highlighting the most significant findings."
    ),
    agent=research_analyst_agent,
)

report_writing_task = Task(
    description=(
        "Write a comprehensive research report that clearly presents the findings from the data analysis report"
    ),
    expected_output=(
        "A well-structured research report that provides insights about the topic."
    ),
    agent=report_writer_agent,
)

report_assessment_task = Task(
    description=(
        "Review and rewrite the research report to ensure clarity, accuracy, and adherence to standards."
    ),
    expected_output=(
        "A polished, coherent research report that meets high-quality standards and effectively communicates the findings."
    ),
    agent=report_editor_agent,
)

# %% [markdown]
# Creates a hierarchical crew with a management LLM and assigns agents to tasks, then kicks off the project.

# %%
# Define the hierarchical crew with a management LLM
research_crew = Crew(
    agents=[research_analyst_agent, report_writer_agent, report_editor_agent],
    tasks=[data_collection_task, data_analysis_task, report_writing_task, report_assessment_task],
    manager_llm=llm,
    process=Process.hierarchical,
    verbose=True
)

# Define the input for the research topic
research_inputs = {
    'topic': 'The impact of AI on modern healthcare systems'
}

# Kickoff the project with the specified topic
result = research_crew.kickoff(inputs=research_inputs)

# %% [markdown]
# Converts the crew’s output into a Markdown string for better readability and displays the results in a formatted manner.

# %%
from IPython.display import Markdown

# Convert the CrewOutput object to a Markdown string
result_markdown = result.raw

# Display the result as Markdown
Markdown(result_markdown)

# %%
