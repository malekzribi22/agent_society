"""
Interactive runner for the flat society simulation.

Allows natural language commands to the SupervisorAgent (coach).
"""

from __future__ import annotations

import argparse
import os
import re

from flat_society import LLMClient, create_flat_society
from flat_society.config import (
    LLM_LOCAL_BASE_URL,
    LLM_LOCAL_MODEL,
    LLM_MAX_CONCURRENT,
    NUM_AGENTS_DEFAULT,
)
from flat_society.supervisor_agent import SupervisorAgent
from flat_society.visualizer import (
    print_agent_scoreboard,
    print_agent_history,
    print_candidate_table,
    print_event_log,
    print_execution_result_table,
    print_full_output,
    print_recent_events_from_task_memory,
    print_supervisor_decision,
    print_task_details,
    print_task_type_summary,
)


def format_reasoning_output(task_type: str, output_text: str | None) -> str:
    if not output_text:
        return "  No detailed response recorded."

    lines = [line.strip() for line in output_text.strip().splitlines() if line.strip()]
    if task_type == "multi_step_reasoning":
        bullets = []
        for line in lines:
            stripped = line.lstrip("- ").rstrip()
            # Normalize numbered steps into bullet format
            match = re.match(r"(\d+)[\.\)]\s*(.*)", stripped)
            if match:
                step_text = match.group(2).strip()
                bullets.append(f"- step {match.group(1)}: {step_text}")
            elif stripped.startswith("####"):
                bullets.append(f"  {stripped}")
            else:
                bullets.append(f"- {stripped}")
        return "\n".join(bullets)
    else:
        final_line = next((line for line in reversed(lines) if line.startswith("####")), None)
        reasoning = "\n".join(lines)
        if final_line:
            return f"{reasoning}\nFinal result: {final_line[4:].strip()}"
        return reasoning


def print_assignment_results(events: list[dict], task_memory: dict[str, dict[str, object]]) -> None:
    """Show which agents executed the current command and how they performed."""
    if not events:
        print("\nNo agents were assigned for this command.")
        return

    total = len(events)
    successes = sum(1 for event in events if event.get("success"))
    failures = total - successes
    agents = sorted({event.get("agent_id") for event in events if event.get("agent_id") is not None})
    task_types = sorted({event.get("task_type") for event in events if event.get("task_type")})

    print("\nCOMMAND SUMMARY")
    print("-" * 80)
    print(
        f"Assignments: {total} | Success: {successes} | Failure: {failures} | "
        f"Agents involved: {', '.join(map(str, agents)) if agents else 'n/a'} | "
        f"Task types: {', '.join(task_types) if task_types else 'n/a'}"
    )
    print("-" * 80)

    print("TASK EXECUTION DETAILS")
    print("-" * 80)
    for event in events:
        agent_id = event.get("agent_id", "?")
        position = event.get("position", "unknown")
        task_type = event.get("task_type", "?")
        task_id = event.get("task_id", "?")
        status = "SUCCESS" if event.get("success") else "FAILED"
        credit = event.get("new_credit_mean", 0.0)
        tool_used = event.get("tool_used")
        tool_note = f", tool={tool_used}" if tool_used else ""
        print(
            f"Task {task_id} ({task_type}) -> Agent {agent_id} ({position}) "
            f"{status} | credit={credit:.3f}{tool_note}"
        )
        task_mem = task_memory.get(task_id) if task_memory else None
        output_text = task_mem.get("output_text") if task_mem else None
        formatted = format_reasoning_output(task_type, output_text)
        print(f"(Agent {agent_id}) response:")
        print(formatted)
        print("-" * 80)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive flat society simulation with natural language commands"
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=NUM_AGENTS_DEFAULT,
        help=f"Number of agents (default: {NUM_AGENTS_DEFAULT})",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable LLM for natural language parsing (requires Ollama)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Flat Society - Interactive Coach Mode")
    print("=" * 80)
    print(f"Initializing {args.num_agents:,} agents...")

    # Create supervisor with agents
    supervisor_base = create_flat_society(args.num_agents, seed=args.seed)

    # Create LLM client if requested
    llm_client = None
    if args.use_llm:
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            raise SystemExit(
                "ERROR: --use-llm requires OPENAI_API_KEY to be set; no local fallback is used."
            )
        print("Initializing LLM client (OpenAI)...")
        llm_client = LLMClient(
            use_local=False,
            max_concurrent=LLM_MAX_CONCURRENT,
            openai_api_key=openai_key,
        )
        print("LLM client ready!")
    else:
        print("LLM disabled - using simple keyword parser")

    # Create SupervisorAgent
    supervisor = SupervisorAgent(
        supervisor_id=supervisor_base.supervisor_id,
        agents=supervisor_base.agents,
        llm_client=llm_client,
    )

    print("=" * 80)
    print("Ready! Type commands or 'help' for examples.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 80)
    print()

    pending_suggestion: dict | None = None

    # Main loop
    while True:
        try:
            cmd = input("Supervisor> ").strip()

            if not cmd:
                continue

            lower_cmd = cmd.lower()
            if lower_cmd in {"quit", "exit", "q"}:
                print("\nExiting...")
                break

            if pending_suggestion and lower_cmd in {"yes", "y", "please", "sure", "do it"}:
                suggestion = pending_suggestion
                pending_suggestion = None
                result = supervisor.run_user_command(
                    suggestion["task_text"], forced_agent_id=suggestion["agent_id"]
                )
                events = result.get("events", [])
                print_assignment_results(events, supervisor.task_memory)
                for insight in result.get("insights", []):
                    print_candidate_table(insight["candidates"], insight["task_type"])
                    print_supervisor_decision(insight["decisions"])
                print_execution_result_table(events, supervisor.task_memory)
                continue

            if pending_suggestion and lower_cmd in {"no", "n", "cancel"}:
                print("Okay, not reassigning that task.")
                pending_suggestion = None
                continue

            if cmd.lower() == "help":
                print("\nExample commands:")
                print("  Task commands:")
                print("    - task: send 50 agents to count people in section A")
                print("    - task: send 3 attackers to shoot at goal in zone B")
                print("    - task: run batch: 200 tasks of type people_count in section A")
                print("    - task: John's cow weighs 400 pounds... (math word problem)")
                print("    - task: explain in 5 steps how Alice can safely change a flat tire")
                print("  History commands:")
                print("    - history agent <agent_id>")
                print("    - history task <task_id>")
                print("    - output <task_id> (show full response)")
                print("    - feedback <task_id> good|bad")
                print("  View commands:")
                print("    - scoreboard")
                print("    - recent")
                print("    - summary")
                print("    - quit")
                print()
                continue

            # Handle task: prefix
            if cmd.lower().startswith("task:"):
                task_cmd = cmd[5:].strip()
                print(f"\nExecuting task: {task_cmd}")
                # Check for batch command
                if "run batch:" in task_cmd.lower():
                    # Parse batch command: "run batch: 200 tasks of type people_count in section A"
                    import re
                    match = re.search(r"(\d+)\s+tasks?\s+of\s+type\s+(\w+)", task_cmd.lower())
                    if match:
                        num_tasks = int(match.group(1))
                        task_type = match.group(2)
                        # Extract area if present
                        area_match = re.search(r"(section|zone)\s+([a-c])", task_cmd.lower())
                        area = area_match.group(2).upper() if area_match else None
                        
                        print(f"Running batch: {num_tasks} tasks of type {task_type}")
                        for i in range(num_tasks):
                            batch_cmd = f"send 1 agent to {task_type}"
                            if area:
                                batch_cmd += f" in {area_match.group(1)} {area}"
                            batch_result = supervisor.run_user_command(batch_cmd)
                            batch_events = batch_result["events"]
                            print_assignment_results(batch_events, supervisor.task_memory)
                            for insight in batch_result["insights"]:
                                print_candidate_table(insight["candidates"], insight["task_type"])
                                print_supervisor_decision(insight["decisions"])
                            print_execution_result_table(batch_events, supervisor.task_memory)
                            if (i + 1) % 50 == 0:
                                print(f"  Progress: {i + 1}/{num_tasks} tasks completed")
                        print(f"Batch complete: {num_tasks} tasks executed")
                    else:
                        result = supervisor.run_user_command(task_cmd)
                        events = result["events"]
                        print_assignment_results(events, supervisor.task_memory)
                        for insight in result["insights"]:
                            print_candidate_table(insight["candidates"], insight["task_type"])
                            print_supervisor_decision(insight["decisions"])
                        print_execution_result_table(events, supervisor.task_memory)
                else:
                    result = supervisor.run_user_command(task_cmd)
                    events = result["events"]
                    print_assignment_results(events, supervisor.task_memory)
                    for insight in result["insights"]:
                        print_candidate_table(insight["candidates"], insight["task_type"])
                        print_supervisor_decision(insight["decisions"])
                    print_execution_result_table(events, supervisor.task_memory)
                
                # Show recent events
                print_recent_events_from_task_memory(supervisor.task_memory, limit=20)
                print_agent_scoreboard(
                    supervisor.get_all_agents(),
                    task_memory=supervisor.task_memory,
                    top_n=10,
                )
                continue

            # Handle history agent command
            if cmd.lower().startswith("history agent"):
                parts = cmd.split()
                if len(parts) >= 3:
                    try:
                        agent_id = int(parts[2])
                        agent = supervisor.get_agent(agent_id)
                        if agent:
                            print_agent_history(agent, limit=50)
                        else:
                            print(f"Agent {agent_id} not found.")
                    except ValueError:
                        print("Invalid agent ID. Usage: history agent <agent_id>")
                else:
                    print("Usage: history agent <agent_id>")
                continue

            # Handle history task command
            if cmd.lower().startswith("history task"):
                parts = cmd.split()
                if len(parts) >= 3:
                    task_id = parts[2]
                    task_mem = supervisor.get_task_memory(task_id)
                    if task_mem:
                        print_task_details(task_mem)
                    else:
                        print(f"Task {task_id} not found.")
                else:
                    print("Usage: history task <task_id>")
                continue

            if cmd.lower().startswith("assign "):
                assign_str = cmd[7:].strip()
                if ":" in assign_str:
                    agent_part, task_text = assign_str.split(":", 1)
                else:
                    parts = assign_str.split(None, 1)
                    if len(parts) < 2:
                        print("Usage: assign <agent_id>: <task description>")
                        continue
                    agent_part, task_text = parts[0], parts[1]
                try:
                    forced_agent_id = int(agent_part.strip())
                except ValueError:
                    print("Invalid agent id for assign command.")
                    continue
                task_text = task_text.strip()
                if not task_text.lower().startswith("task:"):
                    task_text = "task: " + task_text
                result = supervisor.run_user_command(task_text, forced_agent_id=forced_agent_id)
                if result.get("error"):
                    print(result["error"])
                    continue
                events = result.get("events", [])
                forced_event = events[0] if events else None
                if forced_event and not forced_event.get("success"):
                    message = forced_event.get("response_text") or "Agent could not complete the task."
                    print(f"\nSupervisor: {message}")
                    suggested = forced_event.get("suggested_agent")
                    if suggested:
                        clean_task = task_text if task_text.lower().startswith("task:") else f"task: {task_text}"
                        pending_suggestion = {"agent_id": suggested, "task_text": clean_task}
                        print(f"May I assign agent {suggested}? (yes/no)")
                    else:
                        print("Please specify another agent with the 'assign' command.")
                    continue
                print_assignment_results(events, supervisor.task_memory)
                for insight in result.get("insights", []):
                    print_candidate_table(insight["candidates"], insight["task_type"])
                    print_supervisor_decision(insight["decisions"])
                print_execution_result_table(events, supervisor.task_memory)
                continue

            if cmd.lower().startswith("output"):
                parts = cmd.split()
                if len(parts) >= 2:
                    task_id = parts[1]
                    task_mem = supervisor.get_task_memory(task_id)
                    if task_mem:
                        print_full_output(task_mem)
                    else:
                        print(f"Task {task_id} not found.")
                else:
                    print("Usage: output <task_id>")
                continue

            if cmd.lower().startswith("feedback"):
                parts = cmd.split()
                if len(parts) >= 3:
                    task_id = parts[1]
                    sentiment = parts[2].lower()
                    if sentiment not in {"good", "positive", "👍", "bad", "negative", "👎"}:
                        print("Feedback sentiment must be 'good' or 'bad'.")
                        continue
                    positive = sentiment in {"good", "positive", "👍"}
                    result = supervisor.apply_feedback(task_id, positive=positive)
                    print(result.get("message", "Feedback recorded."))
                    for insight in result.get("insights", []):
                        print_candidate_table(insight["candidates"], insight["task_type"])
                        print_supervisor_decision(insight["decisions"])
                    new_events = result.get("events", [])
                    if new_events:
                        print_execution_result_table(new_events, supervisor.task_memory)
                    else:
                        print("No new executions triggered.")
                else:
                    print("Usage: feedback <task_id> good|bad")
                continue

            if cmd.lower() in {"show scoreboard", "scoreboard", "board"}:
                print_agent_scoreboard(
                    supervisor.get_all_agents(), 
                    task_memory=supervisor.task_memory,
                    top_n=30
                )
                continue

            if cmd.lower() in {"recent", "show recent"}:
                print_recent_events_from_task_memory(supervisor.task_memory, limit=50)
                continue

            if cmd.lower() in {"show events", "events", "log"}:
                print_event_log(supervisor.events, max_events=50)
                continue

            if cmd.lower() in {"show summary", "summary", "stats"}:
                print_task_type_summary(supervisor.events)
                continue

            # Execute command (legacy support - no "task:" prefix)
            print(f"\nExecuting: {cmd}")
            result = supervisor.run_user_command(cmd)
            events = result["events"]
            print_assignment_results(events, supervisor.task_memory)
            for insight in result["insights"]:
                print_candidate_table(insight["candidates"], insight["task_type"])
                print_supervisor_decision(insight["decisions"])
            print_execution_result_table(events, supervisor.task_memory)

            # Show recent events
            print_recent_events_from_task_memory(supervisor.task_memory, limit=20)

            # Show updated scoreboard (top performers)
            print_agent_scoreboard(
                supervisor.get_all_agents(), 
                task_memory=supervisor.task_memory,
                top_n=20
            )

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total events: {len(supervisor.events)}")
    print(f"Total tasks in memory: {len(supervisor.task_memory)}")
    print_task_type_summary(supervisor.events)
    print_agent_scoreboard(
        supervisor.get_all_agents(), 
        task_memory=supervisor.task_memory,
        top_n=30
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
