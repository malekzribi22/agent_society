"""
SupervisorAgent: A coach-like supervisor that understands natural language commands
and manages agent assignments with starter/substitution logic.
"""

from __future__ import annotations

import ast
import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import CREDIT_DECAY, MAX_AGENT_MEMORY
from .llm_client import LLMClient
from .models import Agent, Supervisor, Task, Tool, get_tool_registry
from .policy import score_agent

# Optional AutoGen support
HAS_AUTOGEN = False
AssistantAgent = None  # type: ignore

# Try to import AutoGen, but suppress all errors (including numpy/matplotlib issues)
import sys
_original_stderr = sys.stderr
try:
    sys.stderr = open('/dev/null', 'w')  # Suppress stderr during import
    from autogen.agentchat import AssistantAgent
    HAS_AUTOGEN = True
except Exception:
    pass
finally:
    sys.stderr.close()
    sys.stderr = _original_stderr


@dataclass
class SupervisorAgent(Supervisor):
    """Extended Supervisor that acts like a coach with natural language understanding."""
    
    llm_client: LLMClient | None = None
    coach_brain: Any = None  # Optional AutoGen AssistantAgent
    tools: List[Tool] = field(default_factory=get_tool_registry)
    # History of all assignments & results
    events: List[Dict[str, Any]] = field(default_factory=list)
    # Global task memory: task_id -> task execution details
    task_memory: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize tools if not provided."""
        if not self.tools:
            self.tools = get_tool_registry()

    def parse_user_command(self, user_input: str) -> List[Task]:
        """
        Parse a natural language command into Task objects.
        
        If llm_client is available, use it to parse JSON.
        Otherwise, use simple keyword parsing.
        """
        if self.llm_client is not None:
            return self._parse_with_llm(user_input)
        else:
            return self._parse_simple(user_input)

    def _parse_with_llm(self, user_input: str) -> List[Task]:
        """Parse command using LLM to extract structured task information."""
        prompt = f"""You are a coach managing a team of agents. Convert the user command to JSON.

Available task types: people_count, news_summarize, route_plan, sensor_anomaly, math_eval, qa_fact, shoot, run_drill, math_word, multi_step_reasoning

Available skills: shooting, running, goalkeeping, counting, planning, analysis, math_word, multi_step_reasoning

Example commands:
- "Send 50 agents to count people in section A" → {{"task_type": "people_count", "quantity": 50, "target_area": "A", "required_skills": ["counting"]}}
- "Send 3 attackers to shoot at goal in zone B" → {{"task_type": "shoot", "quantity": 3, "target_area": "B", "required_skills": ["shooting"], "position_filter": "attacker"}}
- "Give 10 agents a running drill in zone C" → {{"task_type": "run_drill", "quantity": 10, "target_area": "C", "required_skills": ["running"]}}
- "Solve: John's cow weighs 400 pounds..." → {{"task_type": "math_word", "required_skills": ["math_word"], "quantity": 1, "question": "John's cow weighs 400 pounds..."}}
- "Explain in 5 steps how Alice can change a flat tire" → {{"task_type": "multi_step_reasoning", "quantity": 1, "required_skills": ["multi_step_reasoning"], "question": "Explain in 5 steps how Alice can change a flat tire"}}

Always include the original question text in the "question" field for math_word or multi_step_reasoning commands.

User command: "{user_input}"

Return ONLY valid JSON, no other text:"""

        try:
            response = self.llm_client.generate(prompt, max_tokens=256)
            # Try to extract JSON from response
            json_str = response.strip()
            if json_str.startswith("```"):
                # Remove markdown code blocks
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
            json_str = json_str.strip()
            
            data = json.loads(json_str)
            if isinstance(data, list):
                tasks: List[Task] = []
                for item in data:
                    if isinstance(item, dict):
                        item.setdefault("question", user_input)
                        tasks.extend(self._tasks_from_parsed_data(item))
                return tasks
            else:
                if isinstance(data, dict):
                    data.setdefault("question", user_input)
                return self._tasks_from_parsed_data(data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback to simple parsing
            return self._parse_simple(user_input)

    def _parse_simple(self, user_input: str) -> List[Task]:
        """Simple keyword-based parser for commands."""
        user_lower = user_input.lower()
        tasks = []
        
        # Extract quantity
        quantity = 1
        for word in user_lower.split():
            if word.isdigit():
                quantity = int(word)
                break
        
        # Extract task type
        task_type = "people_count"  # default
        target_area = None
        
        expression_text = None
        if self._looks_like_math_expression(user_input):
            task_type = "math_eval"
            quantity = 1
            expression_text = user_input.strip()
        elif self._looks_like_math_word_problem(user_input):
            task_type = "math_word"
            quantity = 1
            expression_text = user_input.strip()
        elif self._looks_like_multi_step_problem(user_input):
            task_type = "multi_step_reasoning"
            quantity = 1
        elif "count" in user_lower and "people" in user_lower:
            task_type = "people_count"
        elif "shoot" in user_lower or "shot" in user_lower:
            task_type = "shoot"
        elif "run" in user_lower and "drill" in user_lower:
            task_type = "run_drill"
        elif "plan" in user_lower or "route" in user_lower:
            task_type = "route_plan"
        elif "analyze" in user_lower or "anomaly" in user_lower:
            task_type = "sensor_anomaly"
        
        # Extract area/zone
        if "section a" in user_lower or "zone a" in user_lower:
            target_area = "A"
        elif "section b" in user_lower or "zone b" in user_lower:
            target_area = "B"
        elif "section c" in user_lower or "zone c" in user_lower:
            target_area = "C"
        
        # Extract position filter
        position_filter = None
        positions = ["attacker", "runner", "goalkeeper", "counter", "planner", "analyzer"]
        for pos in positions:
            if pos in user_lower:
                position_filter = pos
                break
        
        # Map task types to required skills
        skill_map = {
            "people_count": {"counting"},
            "shoot": {"shooting"},
            "run_drill": {"running"},
            "route_plan": {"planning"},
            "sensor_anomaly": {"analysis"},
            "news_summarize": {"analysis"},
            "math_eval": {"math_eval"},
            "qa_fact": {"analysis"},
            "math_word": {"math_word"},
            "multi_step_reasoning": {"multi_step_reasoning"},
        }
        required_skills = skill_map.get(task_type, {"general"})
        
        # Create tasks
        for i in range(quantity):
            metadata = {
                "target_area": target_area,
                "quantity": quantity,
                "index": i,
                "position_filter": position_filter,
            }
            if task_type == "math_eval":
                metadata["expression"] = expression_text or user_input.strip()
            elif task_type == "math_word":
                metadata["question"] = user_input.strip()
                if expression_text:
                    metadata["expression"] = expression_text
            elif task_type == "multi_step_reasoning":
                metadata["question"] = user_input.strip()

            task = Task(
                task_id=f"task-{int(time.time() * 1000)}-{i}",
                task_type=task_type,
                required_skills=required_skills,
                metadata=metadata,
            )
            tasks.append(task)
        
        return tasks

    def _tasks_from_parsed_data(self, data: Dict[str, Any]) -> List[Task]:
        """Create Task objects from parsed JSON data."""
        task_type = data.get("task_type", "people_count")
        quantity = data.get("quantity", 1)
        target_area = data.get("target_area")
        required_skills = set(data.get("required_skills", ["general"]))
        question = data.get("question")
        expression = data.get("expression")
        text_for_detection = question or expression or ""
        if task_type == "math_eval":
            if text_for_detection and not self._looks_like_math_expression(text_for_detection):
                if self._looks_like_math_word_problem(text_for_detection):
                    task_type = "math_word"

        
        tasks = []
        for i in range(quantity):
            task = Task(
                task_id=f"task-{int(time.time() * 1000)}-{i}",
                task_type=task_type,
                required_skills=required_skills,
                metadata={
                    "target_area": target_area,
                    "quantity": quantity,
                    "index": i,
                    "position_filter": data.get("position_filter"),
                    "question": question,
                    "expression": expression,
                },
            )
            tasks.append(task)
        
        return tasks

    def _looks_like_math_expression(self, text: str) -> bool:
        """Detect simple arithmetic expressions like '2+2' or '10 / 5'."""
        if not text:
            return False
        stripped = text.strip()
        # Pure expression with operators only
        if re.fullmatch(r"[\d\.\s\+\-\*/\(\)=]+", stripped):
            return True
        # Expressions embedded in a sentence, e.g., "what is 2+2?"
        return bool(re.search(r"\d+(?:\s*[\+\-\*/]\s*\d+)+", text))

    def _looks_like_math_word_problem(self, text: str) -> bool:
        """Heuristic detection of math word problems."""
        user_lower = text.lower()
        has_number = bool(re.search(r"\d", user_lower))
        if not has_number:
            return False
        math_keywords = [
            "how much",
            "how many",
            "total",
            "difference",
            "per",
            "each",
            "times",
            "increase",
            "decrease",
            "worth",
            "cost",
            "price",
            "$",
            "convert",
            "conversion",
            "pound",
            "mile",
            "kilogram",
            "kilometer",
        ]
        return any(keyword in user_lower for keyword in math_keywords)

    def _looks_like_multi_step_problem(self, text: str) -> bool:
        """Heuristic detection of multi-step reasoning requests."""
        user_lower = text.lower()
        keywords = [
            "explain",
            "describe",
            "steps",
            "how to",
            "how do",
            "how can",
            "process",
            "procedure",
            "plan",
            "guide",
            "sequence",
            "what is the process",
            "what are the steps",
        ]
        if any(keyword in user_lower for keyword in keywords):
            return not self._looks_like_math_word_problem(text)
        return False

    def _call_llm(self, prompt: str, max_tokens: int = 512) -> Optional[str]:
        """Utility to call the configured LLM client, returning None on failure."""
        if self.llm_client is None:
            return None
        try:
            return self.llm_client.generate(prompt, max_tokens=max_tokens)
        except Exception:
            return None

    def _safe_eval_expression(self, expression: str) -> float:
        """Safely evaluate a simple arithmetic expression."""
        if not re.fullmatch(r"[0-9\.\s\+\-\*/\(\)]+", expression):
            raise ValueError("Expression contains unsupported characters")
        node = ast.parse(expression, mode="eval")

        def _eval(n: ast.AST) -> float:
            if isinstance(n, ast.Expression):
                return _eval(n.body)
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                return float(n.value)
            if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                left = _eval(n.left)
                right = _eval(n.right)
                if isinstance(n.op, ast.Add):
                    return left + right
                if isinstance(n.op, ast.Sub):
                    return left - right
                if isinstance(n.op, ast.Mult):
                    return left * right
                if isinstance(n.op, ast.Div):
                    return left / right
            if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
                operand = _eval(n.operand)
                return operand if isinstance(n.op, ast.UAdd) else -operand
            raise ValueError("Unsupported expression form")

        return _eval(node)

    def _evaluate_math_expression(self, expression: str) -> tuple[Optional[str], Optional[float]]:
        """Evaluate expression and format response text."""
        expr = expression.strip()
        if not expr:
            return ("No expression provided.", None)
        try:
            value = self._safe_eval_expression(expr)
            response = f"Expression: {expr}\nResult: {value}\n#### {value}"
            return response, float(value)
        except Exception as exc:
            return (f"Failed to evaluate '{expr}': {exc}", None)

    def _parse_final_answer(self, text: str | None) -> Optional[float]:
        """Extract the numeric answer from a line starting with ####."""
        if not text:
            return None
        for line in reversed([ln.strip() for ln in text.splitlines() if ln.strip()]):
            if line.startswith("####"):
                match = re.match(r"^####\s*([-+]?\d+(?:\.\d+)?)", line)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        return None
        return None

    def _fallback_math_word_solution(self, question: str) -> str:
        """Fallback response when no LLM is available."""
        return (
            "Unable to solve this math word problem without LLM support.\n"
            "Please rerun with --use-llm so the math specialists can reason about the story."
        )

    def _desired_step_count(self, question: str) -> int:
        """Extract requested step count (2-8) if specified."""
        match = re.search(r"(\d+)\s+steps?", question.lower())
        if match:
            return max(2, min(8, int(match.group(1))))
        return 4

    def _fallback_multi_step_response(self, question: str) -> str:
        """Deterministic multi-step response when no LLM is available."""
        steps = self._desired_step_count(question)
        stripped_question = question.strip().rstrip("?")
        lines = []
        for idx in range(1, steps + 1):
            lines.append(f"{idx}. Address part {idx} of '{stripped_question}' with a clear action.")
        lines.append("#### Provide the requested outcome confidently.")
        return "\n".join(lines)

    def _generate_math_word_output(self, question: str) -> str:
        prompt = f"""
You are an expert math word problem solver.
Carefully read the following story problem and solve it using the provided numbers only.
- Break the solution into clear steps.
- For each arithmetic step, show it as a*b=<<a*b=result>>result (or use +, -, / similarly).
- Do not invent extra data.
- After the reasoning steps, output a final line with ONLY the numeric answer prefixed by '#### '.
  Example: #### 600

Problem:
{question}
"""
        response = self._call_llm(prompt, max_tokens=500)
        if response:
            return response.strip()
        return self._fallback_math_word_solution(question)

    def _solve_math_word_problem(self, question: str) -> tuple[Optional[str], Optional[float]]:
        """Return (reasoning_text, final_numeric_answer)."""
        output_text = self._generate_math_word_output(question)
        final_answer = self._parse_final_answer(output_text)
        return output_text, final_answer

    def _generate_multi_step_output(self, question: str) -> str:
        steps = self._desired_step_count(question)
        prompt = f"""
You are a reasoning specialist.
Answer the following request using {steps} numbered steps (each one sentence).
After the steps, add a final line '#### short_conclusion'.

Question:
{question}
"""
        response = self._call_llm(prompt, max_tokens=400)
        if response:
            return response.strip()
        return self._fallback_multi_step_response(question)

    def _estimate_latency(self, agent: Agent, task_type: str) -> int:
        latencies = [
            tool.avg_latency_ms
            for tool in self.tools
            if tool.tool_id in agent.tools and task_type in tool.tags
        ]
        if latencies:
            return min(latencies)
        return 150

    def _agent_has_required_tool(self, agent: Agent, task_type: str) -> bool:
        if task_type == "math_eval":
            return True  # simple expressions do not require a tool
        return any(
            tool.tool_id in agent.tools and task_type in tool.tags for tool in self.tools
        )

    def _score_candidates_for_task(
        self,
        task: Task,
        exclude_agents: Optional[set[int]] = None,
        require_tool: bool = False,
    ) -> List[Dict[str, Any]]:
        candidates = self.get_candidates_for_task(task.task_type)
        if task.metadata and task.metadata.get("position_filter"):
            position_filter = task.metadata["position_filter"]
            filtered = [a for a in candidates if a.position == position_filter]
            candidates = filtered or candidates
        if exclude_agents:
            candidates = [agent for agent in candidates if agent.agent_id not in exclude_agents]
        stats: List[Dict[str, Any]] = []
        for agent in candidates:
            if require_tool and not self._agent_has_required_tool(agent, task.task_type):
                continue
            score = score_agent(agent, task, self.tools)
            stats.append(
                {
                    "agent": agent,
                    "agent_id": agent.agent_id,
                    "score": score,
                    "credit_mean": agent.credit_mean(task.task_type),
                    "skill": agent.skills.get(task.task_type, 0.0),
                    "min_latency": self._estimate_latency(agent, task.task_type),
                    "tools": ", ".join(agent.tools),
                }
            )
        stats.sort(key=lambda x: x["score"], reverse=True)
        return stats

    def _sample_agents(self, scored: List[Dict[str, Any]], n: int) -> List[Agent]:
        if not scored:
            return []
        weights = [max(entry["score"], 0.01) ** 2 for entry in scored]
        total = sum(weights)
        if total == 0:
            weights = [1.0] * len(scored)
            total = len(scored)
        normalized = [w / total for w in weights]
        selected: List[Agent] = []
        available = list(range(len(scored)))
        for _ in range(min(n, len(scored))):
            if not available:
                break
            available_weights = [normalized[i] for i in available]
            total_available = sum(available_weights)
            if total_available == 0:
                probs = [1.0 / len(available)] * len(available)
            else:
                probs = [w / total_available for w in available_weights]
            idx = random.choices(available, weights=probs)[0]
            selected.append(scored[idx]["agent"])
            available.remove(idx)
        return selected

    def choose_agents_for_task(
        self, task: Task, n: int, exclude_agents: Optional[set[int]] = None
    ) -> tuple[List[Agent], List[Dict[str, Any]]]:
        """
        Choose n agents for a task using weighted sampling.
        High-skill agents (starters) are chosen more often,
        but lower-skill agents (subs) still get chances.
        """
        scored = self._score_candidates_for_task(task, exclude_agents=exclude_agents)
        selected = self._sample_agents(scored, n)
        return selected, scored

    def execute_task(self, agent: Agent, task: Task) -> Dict[str, Any]:
        """
        Simulate task execution by the agent.
        Returns an event dict with success/failure and updated metrics.
        """
        question_text = None
        expression_text = None
        if task.metadata:
            question_text = task.metadata.get("question") or None
            expression_text = task.metadata.get("expression") or None

        # Determine relevant skill for this task
        skill_used = None
        skill_value = 0.5  # default
        
        # Try to match required skills to agent's skills
        for req_skill in task.required_skills:
            # Check position-based skills first
            skill_key = f"skill_{req_skill}"
            if skill_key in agent.skills:
                skill_value = agent.skills[skill_key]
                skill_used = req_skill
                break
            # Fallback to task_type skill
            elif task.task_type in agent.skills:
                skill_value = agent.skills[task.task_type]
                skill_used = task.task_type
                break
        
        has_required_tool = self._agent_has_required_tool(agent, task.task_type)
        generated_output = None
        final_answer = None
        success_override = None
        suggested_agent_id = None
        suggestion_text = None
        if not has_required_tool:
            alternative = self._suggest_alternate_agent(task, exclude_agent=agent.agent_id)
            suggestion = (
                f"Agent {agent.agent_id} reports: I lack the required tools for task '{task.task_type}'."
            )
            if alternative:
                suggested_agent_id = alternative["agent_id"]
                suggestion += (
                    f" Suggested reassignment: agent {alternative['agent_id']} "
                    f"(score={alternative['score']:.3f}, tools={alternative['tools']})."
                )
            else:
                suggestion += " No alternate agent available right now."
            generated_output = suggestion
            suggestion_text = suggestion
            success_override = False
        elif task.task_type == "math_eval":
            expr = expression_text or question_text or ""
            generated_output, final_answer = self._evaluate_math_expression(expr)
            success_override = final_answer is not None
        elif task.task_type == "math_word":
            question_prompt = question_text or "Solve the math word problem carefully."
            generated_output, final_answer = self._solve_math_word_problem(question_prompt)
            success_override = final_answer is not None
        elif task.task_type == "multi_step_reasoning":
            question_prompt = question_text or "Provide a multi-step explanation."
            generated_output = self._generate_multi_step_output(question_prompt)
            success_override = bool(generated_output)

        # Base success probability
        credit_mean = agent.credit_mean(task.task_type)
        base_success_prob = (skill_value * 0.6 + credit_mean * 0.4)
        
        # Tool match bonus
        tool_used = None
        for tool in self.tools:
            if tool.tool_id in agent.tools and task.task_type in tool.tags:
                base_success_prob *= (1.0 - tool.base_error_rate)
                tool_used = tool.tool_id
                break
        
        # Malicious agents sometimes sabotage
        if agent.is_malicious and random.random() < 0.3:
            success = False
        else:
            success = random.random() < base_success_prob

        if success_override is not None:
            success = success_override
        
        # Store credit before update
        credit_before = agent.credit_mean(task.task_type)
        score_used = score_agent(agent, task, self.tools)
        
        # Update credit with decay (history-aware)
        if task.task_type not in agent.credit:
            agent.credit[task.task_type] = (5.0, 5.0)
        
        a, b = agent.credit[task.task_type]
        # Apply decay so older tasks matter less
        a *= CREDIT_DECAY
        b *= CREDIT_DECAY
        
        if success:
            a += 1.0
        else:
            b += 1.0
        
        agent.credit[task.task_type] = (a, b)
        credit_after = agent.credit_mean(task.task_type)
        
        # Update skill via EMA (exponential moving average)
        if skill_used and skill_used in agent.skills:
            alpha = 0.1  # learning rate
            current_skill = agent.skills[skill_used]
            target_skill = 1.0 if success else 0.0
            agent.skills[skill_used] = (1 - alpha) * current_skill + alpha * target_skill
        
        # Update load
        agent.load += 1
        
        timestamp = time.time()
        # Store in agent memory
        agent_memory_entry = {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "success": success,
            "credit_mean_before": credit_before,
            "credit_mean_after": credit_after,
            "score_used": score_used,
            "skill_used": skill_used,
            "tool_used": tool_used,
            "timestamp": timestamp,
            "question": question_text,
            "output_text": generated_output,
            "final_answer": final_answer,
            "notes": None,  # Can be filled with LLM reasoning if available
            "expression": expression_text,
        }
        agent.memory.append(agent_memory_entry)
        # Limit memory size
        if len(agent.memory) > MAX_AGENT_MEMORY:
            agent.memory = agent.memory[-MAX_AGENT_MEMORY:]
        
        # Create event and task memory entry
        latency_ms = random.randint(80, 160)
        event = {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "agent_id": agent.agent_id,
            "position": agent.position,
            "success": success,
            "new_credit_mean": credit_after,
            "skill_used": skill_used or "unknown",
            "tool_used": tool_used,
            "timestamp": timestamp,
            "output_preview": (generated_output.splitlines()[0][:120] if generated_output else None),
            "latency_ms": latency_ms,
            "response_text": generated_output,
            "suggested_agent": suggested_agent_id,
            "suggestion_text": suggestion_text,
        }
        
        # Store in global task memory
        task_record = {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "agent_id": agent.agent_id,
            "position": agent.position,
            "success": success,
            "score_used": score_used,
            "credit_before": credit_before,
            "credit_after": credit_after,
            "skill_used": skill_used,
            "tool_used": tool_used,
            "timestamp": timestamp,
            "output_text": generated_output,
            "final_answer": final_answer,
            "llm_reasoning": None,  # Can be filled if LLM was used
            "expression": expression_text,
        }
        if question_text:
            task_record["question"] = question_text
        self.task_memory[task.task_id] = task_record
        
        return event

    def _suggest_alternate_agent(self, task: Task, exclude_agent: Optional[int] = None) -> Optional[Dict[str, Any]]:
        exclude = {exclude_agent} if exclude_agent is not None else None
        candidates = self._score_candidates_for_task(
            task, exclude_agents=exclude, require_tool=True
        )
        return candidates[0] if candidates else None

    def run_user_command(self, user_input: str, forced_agent_id: Optional[int] = None) -> Dict[str, Any]:
        """
        End-to-end pipeline:
        - Parse tasks from command
        - Choose agents for each task
        - Execute assignments
        - Append events to history
        """
        tasks = self.parse_user_command(user_input)
        if forced_agent_id is not None:
            forced_agent = self.get_agent(forced_agent_id)
            if forced_agent is None:
                return {"events": [], "insights": [], "error": f"Agent {forced_agent_id} not found."}
        else:
            forced_agent = None
        new_events: List[Dict[str, Any]] = []
        insights: List[Dict[str, Any]] = []
        
        for task in tasks:
            # Determine how many agents for this task
            quantity = task.metadata.get("quantity", 1) if task.metadata else 1
            n_agents = min(quantity, len(self.agents))
            
            # Choose agents and capture candidate stats
            if forced_agent is not None:
                candidate_stats = self._score_candidates_for_task(task)
                selected_agents = [forced_agent]
            else:
                selected_agents, candidate_stats = self.choose_agents_for_task(task, n_agents)
            task_insight = {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "candidates": candidate_stats,
                "decisions": [],
            }
            candidate_map = {entry["agent_id"]: entry for entry in candidate_stats}
            
            # Execute task with each agent
            for agent in selected_agents:
                event = self.execute_task(agent, task)
                self.events.append(event)
                new_events.append(event)
                entry = candidate_map.get(agent.agent_id, {})
                task_insight["decisions"].append(
                    {
                        "agent_id": agent.agent_id,
                        "task_id": task.task_id,
                        "score": entry.get("score"),
                        "credit_mean": entry.get("credit_mean"),
                        "why": "Highest score and matching specialization.",
                        "risks": "Latency acceptable.",
                    }
                )
            insights.append(task_insight)

        return {"events": new_events, "insights": insights}

    def apply_feedback(self, task_id: str, positive: bool) -> Dict[str, Any]:
        """
        Adjust credit based on user feedback. If negative, penalize agent and reassign.
        """
        task_mem = self.task_memory.get(task_id)
        if not task_mem:
            return {"message": f"Task {task_id} not found.", "events": [], "insights": []}
        agent_id = task_mem.get("agent_id")
        task_type = task_mem.get("task_type", "unknown")
        agent = self.get_agent(agent_id) if agent_id is not None else None
        if agent is None:
            return {"message": f"Agent {agent_id} not found for task {task_id}.", "events": [], "insights": []}
        a, b = agent.credit.get(task_type, (5.0, 5.0))
        if positive:
            agent.credit[task_type] = (a + 1.0, b)
            return {"message": f"Positive feedback recorded for agent {agent_id}.", "events": [], "insights": []}

        # Negative feedback: penalize and rerun
        agent.credit[task_type] = (a, b + 1.5)
        metadata = {
            "question": task_mem.get("question"),
            "expression": task_mem.get("expression"),
        }
        rerun_task = Task(
            task_id=f"{task_id}-retry-{int(time.time() * 1000)}",
            task_type=task_type,
            required_skills={task_type},
            metadata=metadata,
        )
        exclude = {agent_id} if agent_id is not None else None
        selected_agents, candidates = self.choose_agents_for_task(rerun_task, 1, exclude_agents=exclude)
        events: List[Dict[str, Any]] = []
        decisions: List[Dict[str, Any]] = []
        if selected_agents:
            for replacement in selected_agents:
                event = self.execute_task(replacement, rerun_task)
                self.events.append(event)
                events.append(event)
                source_entry = next((c for c in candidates if c["agent_id"] == replacement.agent_id), {})
                decisions.append(
                    {
                        "agent_id": replacement.agent_id,
                        "task_id": rerun_task.task_id,
                        "score": source_entry.get("score"),
                        "credit_mean": source_entry.get("credit_mean"),
                        "why": "Replacement assignment after negative feedback.",
                        "risks": "Re-evaluating task.",
                    }
                )
        insight = {
            "task_id": rerun_task.task_id,
            "task_type": task_type,
            "candidates": candidates,
            "decisions": decisions,
        }
        return {
            "message": f"Negative feedback recorded. Reassigned task {task_id} to a new agent.",
            "events": events,
            "insights": [insight] if events else [],
        }

    def get_agent_history(self, agent_id: int) -> List[Dict[str, Any]]:
        """Return the memory list for a given agent_id."""
        agent = self.get_agent(agent_id)
        if agent is None:
            return []
        return agent.memory.copy()  # Return a copy to avoid external modification

    def get_task_memory(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Return the stored memory for a given task_id, or None."""
        return self.task_memory.get(task_id)

    def get_task_history_by_type(self, task_type: str) -> List[Dict[str, Any]]:
        """Return all task_memory entries with this task_type."""
        return [
            task_mem for task_mem in self.task_memory.values()
            if task_mem.get("task_type") == task_type
        ]
