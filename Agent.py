import os
import logging
import google.cloud.logging
from dotenv import load_dotenv

from google.adk import Agent
from google.adk.agents import SequentialAgent, LoopAgent, ParallelAgent
from google.adk.models import Gemini
from google.adk.tools import exit_loop
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from google.adk.tools.langchain_tool import LangchainTool


google.cloud.logging.Client().setup_logging()
load_dotenv()

MODEL_NAME = os.getenv("MODEL")

RETRY_OPTIONS = types.HttpRetryOptions(initial_delay=1, attempts=6)

SHARED_MODEL = Gemini(model=MODEL_NAME, retry_options=RETRY_OPTIONS)


# STATE MANAGEMENT TOOLS (STRICT REQUIREMENT)


def set_state(tool_context: ToolContext, field: str, value: str):
    tool_context.state[field] = value
    return {"status": "success"}


def append_state(tool_context: ToolContext, field: str, value: str):
    existing = tool_context.state.get(field, [])
    if not isinstance(existing, list):
        existing = []
    existing.append(value)
    tool_context.state[field] = existing
    return {"status": "success"}


def reset_case(tool_context: ToolContext):
    tool_context.state["topic"] = ""
    tool_context.state["pos_data"] = []
    tool_context.state["neg_data"] = []
    tool_context.state["judge_feedback"] = ""
    tool_context.state["pos_round"] = 0
    tool_context.state["neg_round"] = 0
    return {"status": "reset"}


def write_file(tool_context: ToolContext, directory: str, filename: str, content: str):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return {"status": "success", "path": path}


# WIKIPEDIA TOOL
wiki_tool = LangchainTool(tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()))


# STEP 1 — THE INQUIRY (CLERK)
clerk = Agent(
    name="clerk",
    model=SHARED_MODEL,
    instruction="""
Extract ONLY the historical topic from user input.

Call:
set_state(field="topic", value="<topic_name>")

Do NOT output text.
Stop immediately.
""",
    tools=[set_state],
)


# STEP 2 — THE INVESTIGATION (PARALLEL)
admirer = Agent(
    name="admirer",
    model=SHARED_MODEL,
    instruction="""
You are Agent A: The Admirer.

Goal:
Collect POSITIVE achievements of {topic}.

SEARCH STRATEGY:
- Use keywords:
  "{topic} achievements"
  "{topic} accomplishments"
  "{topic} positive impact"
  "{topic} achievements"
  "{topic} inventions and anatomy studies"  
  "{topic} scientific contributions"
- If judge_feedback exists, refine search using it.

RULES:
- Each round add ONLY 2 NEW distinct achievements.
- Avoid duplication.
- Write concise Thai bullet-style sentences.
- Append EACH point using:
  append_state(field="pos_data", value="...")

Do not output explanation text.
""",
    tools=[wiki_tool, append_state],
)


critic = Agent(
    name="critic",
    model=SHARED_MODEL,
    instruction="""
You are Agent B: The Critic.

Goal:
Collect NEGATIVE aspects or controversies of {topic}.

SEARCH STRATEGY:
- Use keywords:
  "{topic} controversy"
  "{topic} criticism"
  "{topic} failure"
  "{topic} unfinished works and failures"
  "{topic} historical rivalry"
- If judge_feedback exists, refine search using it.

RULES:
- Each round add ONLY 2 NEW distinct criticisms.
- Avoid duplication.
- Write concise Thai bullet-style sentences.
- Append EACH point using:
  append_state(field="neg_data", value="...")

Do not output explanation text.
""",
    tools=[wiki_tool, append_state],
)

investigation_team = ParallelAgent(
    name="investigation_team",
    sub_agents=[admirer, critic],
)


# STEP 3 — LOOP TRIAL (JUDGE)
judge = Agent(
    name="judge",
    model=SHARED_MODEL,
    instruction="""
You are Agent C: The Judge.

Review:

Positive Points:
{pos_data?}

Negative Points:
{neg_data?}

Decision Rules:

1. If negative side has < 4 points:
   Call set_state(field="judge_feedback", 
                  value="Found only {len(neg_data?)} points. Please find MORE specific controversies or failed projects.")
   DO NOT call exit_loop.

2. If positive side has < 4 points:
   Call set_state(field="judge_feedback", 
                  value="Found only {len(pos_data?)} points. Please find MORE diverse achievements in science, art, or engineering.")
   DO NOT call exit_loop.

3. If BOTH sides have at least 4 clear bullet points:
   Call exit_loop EXACTLY ONCE.

Do not output any text.
""",
    tools=[set_state, exit_loop],
)


court_trial = LoopAgent(
    name="court_trial",
    sub_agents=[investigation_team, judge],
    max_iterations=6,
)


# STEP 4 — THE VERDICT (SCRIBE)
verdict_scribe = Agent(
    name="verdict_scribe",
    model=SHARED_MODEL,
    instruction="""
Create a neutral academic Thai report about {topic}.

Use:
Positive Data: {pos_data}
Negative Data: {neg_data}

STRUCTURE:

1) ความสำเร็จ
- bullet points

2) ข้อโต้แย้งและข้อวิจารณ์
- bullet points

3) บทสรุปเชิงเป็นกลาง
- Analytical, balanced, no bias

Then call:
write_file(
    directory="court_reports",
    filename="{topic}_verdict.txt",
    content="<full_report>"
)

Do not output text.
""",
    tools=[write_file],
)


# ROOT SYSTEM (FINAL WORKFLOW)
resetter = Agent(
    name="resetter",
    model=SHARED_MODEL,
    instruction="""
Call reset_case once.
Do not output text.
""",
    tools=[reset_case],
)

root_agent = SequentialAgent(
    name="historical_court_system",
    sub_agents=[resetter, clerk, court_trial, verdict_scribe],
)
