"""
Visualization utilities for the flat society simulation.

Provides scoreboards, event logs, and optional plotting.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.json import JSON as RichJSON
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .models import Agent

console = Console()


def print_recent_events_from_task_memory(
    task_memory: Dict[str, Dict[str, Any]], limit: int = 50
) -> None:
    """
    Print the last `limit` tasks in chronological order with key information.
    """
    import time

    if not task_memory:
        print("\nNo recorded tasks yet.")
        return

    # Take the most recent `limit` events, but display oldest->newest for readability
    sorted_events = sorted(
        task_memory.values(), key=lambda x: x.get("timestamp", 0), reverse=True
    )[:limit]
    sorted_events.reverse()

    print("\n" + "=" * 110)
    print(f"RECENT TASK LOG (last {len(sorted_events)} entries)")
    print("=" * 110)
    print(
        f"{'Time':<12} {'Task ID':<18} {'Type':<13} {'Agent':<7} {'Pos':<10} "
        f"{'Result':<8} {'Credit':<15} {'Skill':<10} {'Tool':<12}"
    )
    print("-" * 110)

    for event in sorted_events:
        timestamp = event.get("timestamp", 0)
        time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
        task_id = event.get("task_id", "?")[:16]
        task_type = event.get("task_type", "?")[:11]
        agent_id = event.get("agent_id", "?")
        position = (event.get("position") or "unknown")[:10]
        success = event.get("success", False)
        status = "SUCCESS" if success else "FAILED"
        credit_before = event.get("credit_before", event.get("credit_after", 0.0))
        credit_after = event.get("credit_after", 0.0)
        skill_used = (event.get("skill_used") or "unknown")[:10]
        tool_used = (event.get("tool_used") or "-")[:12]
        credit_str = f"{credit_before:.2f}->{credit_after:.2f}"

        print(
            f"{time_str:<12} {task_id:<18} {task_type:<13} {agent_id!s:<7} {position:<10} "
            f"{status:<8} {credit_str:<15} {skill_used:<10} {tool_used:<12}"
        )

    print("=" * 110 + "\n")


def print_recent_events(
    task_memory: Dict[str, Dict[str, Any]], limit: int = 50
) -> None:
    """
    Backwards-compatible wrapper that prints recent events using task memory.
    """
    print_recent_events_from_task_memory(task_memory, limit=limit)


def print_event_log(events: List[Dict], max_events: int = 100) -> None:
    """
    Print each event as a readable line.
    
    Example:
    Agent 7 (attacker) did task 'shoot' in zone B -> SUCCESS, credit now 0.82
    """
    print("\n" + "=" * 80)
    print("EVENT LOG (last {} events)".format(min(max_events, len(events))))
    print("=" * 80)
    
    for event in events[-max_events:]:
        agent_id = event.get("agent_id", "?")
        position = event.get("position", "unknown")
        task_type = event.get("task_type", "?")
        success = event.get("success", False)
        credit = event.get("new_credit_mean", 0.0)
        target_area = ""
        
        # Extract target area from metadata if available
        if "target_area" in str(event):
            # Try to get from task metadata if we had it
            pass  # Would need task object, simplified for now
        
        status = "SUCCESS" if success else "FAILED"
        print(
            f"Agent {agent_id:4d} ({position:10s}) did task '{task_type:15s}' "
            f"-> {status:7s}, credit now {credit:.2f}"
        )
    print("=" * 80 + "\n")


def print_candidate_table(candidate_stats: List[Dict[str, Any]], task_type: str, top_k: int = 30) -> None:
    """Render a Rich table showing the top candidates."""
    if not candidate_stats:
        return
    table = Table(title=f"Society top-{min(top_k, len(candidate_stats))} ({task_type})")
    table.add_column("agent_id")
    table.add_column("score")
    table.add_column("credit_mean")
    table.add_column("skill")
    table.add_column("min_latency")
    table.add_column("tools")
    for entry in candidate_stats[:top_k]:
        table.add_row(
            str(entry.get("agent_id")),
            f"{entry.get('score', 0.0):.3f}",
            f"{entry.get('credit_mean', 0.0):.3f}",
            f"{entry.get('skill', 0.0):.2f}",
            str(entry.get("min_latency", "-")),
            entry.get("tools", ""),
        )
    console.print(table)


def print_supervisor_decision(decisions: List[Dict[str, Any]]) -> None:
    """Render supervisor decision JSON similar to autogen demo."""
    if not decisions:
        return
    data = decisions if len(decisions) > 1 else decisions[0]
    console.print(
        Panel(
            RichJSON.from_data(data, indent=2),
            title="Supervisor decision (JSON)",
            border_style="cyan",
        )
    )


def _short_text(text: str, limit: int = 200) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def print_execution_result_table(
    events: List[Dict[str, Any]], task_memory: Dict[str, Dict[str, Any]]
) -> None:
    """Render execution results with outputs."""
    if not events:
        return
    table = Table(title="Execution result")
    table.add_column("agent_id")
    table.add_column("task_id")
    table.add_column("task_type")
    table.add_column("latency_ms")
    table.add_column("outcome")
    table.add_column("output")
    for event in events:
        task_id = event.get("task_id")
        record = task_memory.get(task_id, {})
        output = record.get("output_text") or event.get("response_text") or ""
        table.add_row(
            str(event.get("agent_id")),
            task_id or "?",
            event.get("task_type", "?"),
            str(event.get("latency_ms", "-")),
            "SUCCESS" if event.get("success") else "FAILED",
            _short_text(output),
        )
    console.print(table)


def print_full_output(task_mem: Dict[str, Any]) -> None:
    """Print the complete output text for a task."""
    output = task_mem.get("output_text")
    if not output:
        console.print("No output recorded for this task.")
        return
    console.print(Panel(Text(output), title=f"Full Output - {task_mem.get('task_id', '?')}"))


def print_agent_scoreboard(
    agents: List[Agent], 
    task_memory: Optional[Dict[str, Dict[str, Any]]] = None,
    top_n: Optional[int] = None
) -> None:
    """
    Print a scoreboard showing:
    - Total tasks per agent
    - Success rate (overall and per task_type)
    - Main position & skills
    
    Shows starters (top performers) prominently, but includes all agents.
    """
    agent_lookup = {agent.agent_id: agent for agent in agents}

    # Aggregate statistics per agent
    agent_stats: Dict[int, Dict] = defaultdict(
        lambda: {
            "total_tasks": 0,
            "successful": 0,
            "failed": 0,
            "by_task_type": defaultdict(lambda: {"success": 0, "total": 0}),
            "position": "",
            "agent_id": 0,
        }
    )
    
    # Initialize with agent info
    for agent in agents:
        agent_stats[agent.agent_id]["position"] = agent.position
        agent_stats[agent.agent_id]["agent_id"] = agent.agent_id
    
    # Process task_memory if provided, otherwise use agent.memory
    if task_memory:
        for task_mem in task_memory.values():
            agent_id = task_mem.get("agent_id")
            if agent_id is None:
                continue
            
            stats = agent_stats[agent_id]
            stats["total_tasks"] += 1
            if task_mem.get("success", False):
                stats["successful"] += 1
            else:
                stats["failed"] += 1
            
            task_type = task_mem.get("task_type", "unknown")
            task_stats = stats["by_task_type"][task_type]
            task_stats["total"] += 1
            if task_mem.get("success", False):
                task_stats["success"] += 1
    else:
        # Fallback: use agent.memory
        for agent in agents:
            for mem_entry in agent.memory:
                stats = agent_stats[agent.agent_id]
                stats["total_tasks"] += 1
                if mem_entry.get("success", False):
                    stats["successful"] += 1
                else:
                    stats["failed"] += 1
                
                task_type = mem_entry.get("task_type", "unknown")
                task_stats = stats["by_task_type"][task_type]
                task_stats["total"] += 1
                if mem_entry.get("success", False):
                    task_stats["success"] += 1
    
    # Convert to list and sort by success rate
    stats_list = []
    for agent_id, stats in agent_stats.items():
        if stats["total_tasks"] > 0:
            success_rate = stats["successful"] / stats["total_tasks"]
        else:
            success_rate = 0.0
        stats["overall_success_rate"] = success_rate
        stats_list.append(stats)
    
    # Sort by success rate (descending)
    stats_list.sort(key=lambda x: x["overall_success_rate"], reverse=True)
    
    # Filter to top N if specified
    if top_n is not None:
        stats_list = stats_list[:top_n]
    
    # Print header
    print("\n" + "=" * 140)
    print("AGENT SCOREBOARD" + (f" (Top {top_n})" if top_n else ""))
    print("=" * 140)
    print(
        f"{'Agent ID':<8} {'Position':<12} {'Total':<8} {'Success':<8} {'Failed':<8} "
        f"{'Success %':<10} {'Math %':<8} {'Reason %':<10} {'Best Tasks':<40} {'Top Credit Means':<30}"
    )
    print("-" * 140)
    
    # Print each agent
    for stats in stats_list:
        agent_id = stats["agent_id"]
        position = stats["position"]
        total = stats["total_tasks"]
        successful = stats["successful"]
        failed = stats["failed"]
        success_rate = stats["overall_success_rate"]

        # Find top task types by success rate
        task_perf = []
        for task_type, task_stats in stats["by_task_type"].items():
            if task_stats["total"] > 0:
                rate = task_stats["success"] / task_stats["total"]
                task_perf.append((rate, task_stats["total"], task_type))
        task_perf.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best_tasks = ", ".join(
            f"{task_type} ({rate*100:>4.0f}%/{total_t})"
            for rate, total_t, task_type in task_perf[:3]
        ) or "n/a"

        # Credit mean summary
        credit_summary = "n/a"
        agent = agent_lookup.get(agent_id)
        if agent and agent.credit:
            credit_perf = []
            for task_type, (a_val, b_val) in agent.credit.items():
                total_prior = a_val + b_val
                if total_prior <= 0:
                    continue
                mean = a_val / total_prior
                credit_perf.append((mean, total_prior, task_type))
            credit_perf.sort(key=lambda x: (x[0], x[1]), reverse=True)
            credit_summary = ", ".join(
                f"{task_type}:{mean:.2f}"
                for mean, _, task_type in credit_perf[:3]
            ) or "n/a"
        
        # Mark starters (top performers)
        marker = "⭐" if success_rate >= 0.7 and total >= 5 else "  "

        # Specialist rates
        math_stats = stats["by_task_type"].get("math_word")
        math_rate = (
            math_stats["success"] / math_stats["total"]
            if math_stats and math_stats["total"] > 0
            else None
        )
        reasoning_stats = stats["by_task_type"].get("multi_step_reasoning")
        reasoning_rate = (
            reasoning_stats["success"] / reasoning_stats["total"]
            if reasoning_stats and reasoning_stats["total"] > 0
            else None
        )
        math_display = f"{math_rate*100:>6.1f}%" if math_rate is not None else "   n/a"
        reasoning_display = f"{reasoning_rate*100:>6.1f}%" if reasoning_rate is not None else "   n/a"

        print(
            f"{marker} {agent_id:<8} {position:<12} {total:<8} {successful:<8} {failed:<8} "
            f"{success_rate*100:>6.1f}%    {math_display:<8} {reasoning_display:<10} {best_tasks:<40} {credit_summary:<30}"
        )
    
    print("=" * 140)
    print("⭐ = Starter (high success rate, multiple tasks)")
    print("=" * 140 + "\n")


def print_task_type_summary(events: List[Dict]) -> None:
    """Print summary statistics per task type."""
    task_stats: Dict[str, Dict] = defaultdict(
        lambda: {"total": 0, "success": 0, "failed": 0}
    )
    
    for event in events:
        task_type = event.get("task_type", "unknown")
        task_stats[task_type]["total"] += 1
        if event.get("success", False):
            task_stats[task_type]["success"] += 1
        else:
            task_stats[task_type]["failed"] += 1
    
    print("\n" + "=" * 80)
    print("TASK TYPE SUMMARY")
    print("=" * 80)
    print(f"{'Task Type':<20} {'Total':<10} {'Success':<10} {'Failed':<10} {'Success %':<10}")
    print("-" * 80)
    
    for task_type, stats in sorted(task_stats.items()):
        total = stats["total"]
        success = stats["success"]
        failed = stats["failed"]
        rate = (success / total * 100) if total > 0 else 0.0
        print(
            f"{task_type:<20} {total:<10} {success:<10} {failed:<10} {rate:>6.1f}%"
        )
    
    print("=" * 80 + "\n")


def plot_agent_performance(
    agents: List[Agent], events: List[Dict], top_n: int = 20
) -> None:
    """
    Optional Matplotlib plotting of top N agents by success rate.
    Requires matplotlib to be installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available, skipping plot.")
        return
    
    # Aggregate stats
    agent_stats: Dict[int, float] = defaultdict(float)
    agent_totals: Dict[int, int] = defaultdict(int)
    
    for event in events:
        agent_id = event.get("agent_id")
        if agent_id is None:
            continue
        agent_totals[agent_id] += 1
        if event.get("success", False):
            agent_stats[agent_id] += 1
    
    # Calculate success rates
    success_rates = []
    agent_ids = []
    for agent_id, total in sorted(agent_totals.items(), key=lambda x: x[1], reverse=True)[:top_n]:
        if total > 0:
            rate = agent_stats[agent_id] / total
            success_rates.append(rate * 100)
            agent_ids.append(str(agent_id))
    
    if not success_rates:
        print("No data to plot.")
        return
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(agent_ids)), success_rates)
    plt.xlabel("Agent ID")
    plt.ylabel("Success Rate (%)")
    plt.title(f"Top {top_n} Agents by Success Rate")
    plt.xticks(range(len(agent_ids)), agent_ids, rotation=45)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_agent_history(agent: Agent, limit: int = 50) -> None:
    """Print the history/memory of a specific agent."""
    import time

    total_tasks = len(agent.memory)
    successes = sum(1 for entry in agent.memory if entry.get("success"))
    success_rate = (successes / total_tasks * 100) if total_tasks else 0.0
    recent_entries = agent.memory[-limit:]
    
    print("\n" + "=" * 100)
    print(f"AGENT {agent.agent_id} HISTORY ({agent.position})")
    print("=" * 100)
    print(f"Position: {agent.position}")
    print(f"Total tasks: {total_tasks}")
    print(f"Success rate: {success_rate:.1f}% ({successes}/{total_tasks})")
    print(f"Current credit means:")
    for task_type, (a, b) in agent.credit.items():
        mean = a / (a + b) if (a + b) > 0 else 0.5
        print(f"  {task_type}: {mean:.3f}")
    print()

    print(
        f"{'Task ID':<18} {'Type':<12} {'Result':<8} {'Credit':<15} "
        f"{'Skill':<10} {'Tool':<12} {'Time':<20}"
    )
    print("-" * 100)
    
    for entry in recent_entries:
        task_id = entry.get("task_id", "?")[:16]
        task_type = entry.get("task_type", "?")[:10]
        success = "YES" if entry.get("success", False) else "NO"
        credit_before = entry.get("credit_mean_before", entry.get("credit_mean_after", 0.0))
        credit_after = entry.get("credit_mean_after", 0.0)
        credit_str = f"{credit_before:.2f}->{credit_after:.2f}"
        skill_used = (entry.get("skill_used") or "unknown")[:10]
        tool_used = (entry.get("tool_used") or "-")[:12]
        timestamp = entry.get("timestamp", 0)
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        
        print(
            f"{task_id:<18} {task_type:<12} {success:<8} {credit_str:<15} "
            f"{skill_used:<10} {tool_used:<12} {time_str:<20}"
        )
    
    print("=" * 100 + "\n")


def print_task_details(task_mem: Dict[str, Any]) -> None:
    """Print detailed information about a specific task."""
    print("\n" + "=" * 80)
    print(f"TASK DETAILS: {task_mem.get('task_id', '?')}")
    print("=" * 80)
    print(f"Task Type: {task_mem.get('task_type', '?')}")
    print(f"Agent ID: {task_mem.get('agent_id', '?')}")
    print(f"Agent Position: {task_mem.get('position', 'unknown')}")
    print(f"Success: {'YES' if task_mem.get('success', False) else 'NO'}")
    print(f"Score Used: {task_mem.get('score_used', 0.0):.3f}")
    print(f"Credit Before: {task_mem.get('credit_before', 0.0):.3f}")
    print(f"Credit After: {task_mem.get('credit_after', 0.0):.3f}")
    print(f"Skill Used: {task_mem.get('skill_used', '?')}")
    print(f"Tool Used: {task_mem.get('tool_used', '?')}")
    if task_mem.get("question"):
        print(f"Question: {task_mem.get('question')}")
    timestamp = task_mem.get("timestamp", 0)
    import time
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    print(f"Timestamp: {time_str}")
    output_text = task_mem.get("output_text")
    if output_text:
        print("\nOutput Preview:")
        lines = output_text.strip().splitlines()
        preview_lines = lines[:8]
        for line in preview_lines:
            print(f"  {line}")
        if len(lines) > len(preview_lines):
            print("  ...")
    if task_mem.get("llm_reasoning"):
        print(f"LLM Reasoning: {task_mem.get('llm_reasoning')}")
    print("=" * 80 + "\n")


def plot_task_success_by_type(task_memory: Dict[str, Dict[str, Any]]) -> None:
    """Bar chart of success rate per task_type."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available, skipping plot.")
        return
    
    task_stats: Dict[str, Dict] = defaultdict(
        lambda: {"total": 0, "success": 0}
    )
    
    for task_mem in task_memory.values():
        task_type = task_mem.get("task_type", "unknown")
        task_stats[task_type]["total"] += 1
        if task_mem.get("success", False):
            task_stats[task_type]["success"] += 1
    
    task_types = list(task_stats.keys())
    success_rates = [
        (task_stats[t]["success"] / task_stats[t]["total"] * 100)
        if task_stats[t]["total"] > 0
        else 0.0
        for t in task_types
    ]
    
    plt.figure(figsize=(10, 6))
    plt.bar(task_types, success_rates)
    plt.xlabel("Task Type")
    plt.ylabel("Success Rate (%)")
    plt.title("Success Rate by Task Type")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
