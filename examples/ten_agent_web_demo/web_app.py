"""Mini 10-agent web demo.

Usage:
    pip install flask
    python -m ten_agent_web_demo.web_app

Then open http://127.0.0.1:5000 in your browser.
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, abort, redirect, render_template_string, request, send_file, url_for

from .config import ARCH_MODE_FLAT, ARCH_MODE_HIER
from .models import TASK_TYPE_MATH, TASK_TYPE_REASONING, TASK_TYPE_UAV
from .simulation import (
    SimulationState,
    apply_feedback,
    create_default_state,
    execute_task,
    execute_random_uav_task,
    compute_selection_scores,
    load_uav_tasks,
    run_uav_batch,
    split_into_subtasks,
    run_batch,
    process_user_line,

)

app = Flask(__name__)
STATE: SimulationState = create_default_state()
CURRENT_TASK_MODE = TASK_TYPE_MATH
EXAMPLES_ROOT = Path(__file__).resolve().parent.parent
UAV_DATA_ROOT = (EXAMPLES_ROOT / "uav_roundabouts").resolve()


def _resolve_uav_image_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = (EXAMPLES_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    try:
        candidate.relative_to(UAV_DATA_ROOT)
    except ValueError:
        return None
    if not candidate.exists():
        return None
    return candidate


def _relative_uav_image_path(raw_path: str | None) -> str | None:
    resolved = _resolve_uav_image_path(raw_path)
    if not resolved:
        return None
    return str(resolved.relative_to(UAV_DATA_ROOT))


def _aggregate_agent_type_stats():
    stats: dict[int, dict[str, dict[str, int]]] = {}
    for task in STATE.tasks:
        agent_id = task.assigned_agent_id
        if not agent_id:
            continue
        agent_stats = stats.setdefault(agent_id, {})
        entry = agent_stats.setdefault(task.task_type, {"successes": 0, "failures": 0})
        if task.success is True:
            entry["successes"] += 1
        elif task.success is False:
            entry["failures"] += 1
    return stats


def _build_agent_scoreboards(state: SimulationState):
    stats_lookup = _aggregate_agent_type_stats()
    boards = []
    label_map = [
        (TASK_TYPE_MATH, "Math"),
        (TASK_TYPE_REASONING, "Reasoning"),
        (TASK_TYPE_UAV, "UAV Car Count"),
    ]
    for task_type, label in label_map:
        scored = compute_selection_scores(state, task_type)
        rows = []
        for agent, total, credit, exploration, fairness in scored:
            agent_stats = stats_lookup.get(agent.agent_id, {}).get(
                task_type, {"successes": 0, "failures": 0}
            )
            rows.append(
                {
                    "agent_id": agent.agent_id,
                    "name": agent.name,
                    "role": agent.role,
                    "credit": credit,
                    "explore": exploration,
                    "fairness": fairness,
                    "score": total,
                    "exposure": agent.exposure.get(task_type, 0),
                    "successes": agent_stats.get("successes", 0),
                    "failures": agent_stats.get("failures", 0),
                }
            )
        boards.append({"task_type": task_type, "label": label, "rows": rows})
    return boards


def _build_uav_preview(limit: int = 12):
    try:
        tasks = load_uav_tasks(limit=limit, random_subset=False)
    except Exception as exc:  # pragma: no cover - defensive for demo use
        return [], str(exc)
    preview_rows = []
    for task in tasks:
        preview_rows.append(
            {
                "image_name": task.image_name,
                "num_cars": task.num_cars,
                "image_path": task.image_path,
                "image_rel": _relative_uav_image_path(task.image_path),
                "question": f"How many cars are in {getattr(task, 'image_name', 'this image')}?",
            }
        )
    return preview_rows, None


def _coerce_int(value: str | None):
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _uav_task_rows(state: SimulationState, limit: int = 25):
    rows = [task for task in state.tasks if task.task_type == TASK_TYPE_UAV]
    rows = rows[-limit:]
    rows.reverse()
    formatted = []
    for task in rows:
        gt = _coerce_int(task.ground_truth)
        pred = _coerce_int(task.agent_answer or task.output)
        formatted.append(
            {
                "task_id": task.task_id,
                "image_name": task.image_name or (task.input_text[:40] if task.input_text else "n/a"),
                "image_rel": _relative_uav_image_path(task.image_path),
                "ground_truth": gt,
                "prediction": pred,
                "abs_error": task.abs_error,
                "reward": task.numeric_reward,
                "success": task.success,
                "agent": task.assigned_agent_name or task.assigned_agent_id,
                "leader": task.leader_name,
            }
        )
    return formatted

TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>10-Agent Mini Society</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
      body { font-family: Arial, sans-serif; margin: 2rem; background-color: #111; color: #eee; }
      h1, h2, h3 { color: #6cf; }
      table { width: 100%; border-collapse: collapse; margin-bottom: 2rem; }
      th, td { border: 1px solid #555; padding: 0.5rem; }
      th { background-color: #222; }
      tr.success { background-color: #123c18; }
      tr.failure { background-color: #3c1212; }
      tr.pending { background-color: #333; }
      .feedback-btn { border:none; background:transparent; cursor:pointer; font-size:1rem; }
      .feedback-btn.good { color:#3cff7d; }
      .feedback-btn.bad { color:#ff5c5c; }
      .form-panel { margin-bottom: 2rem; background:#1d1d1d; padding:1rem; border-radius:8px; }
      textarea { width: 100%; height: 120px; }
      button { padding: 0.5rem 1rem; background:#6cf; border:none; color:#000; font-weight:bold; cursor:pointer; }
      .hint { font-size:0.9rem; color:#bbb; margin-top:0.5rem; }
      .uav-preview-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:0.75rem; margin-bottom:0.75rem; }
      .uav-card { display:block; border:1px solid #333; padding:0.5rem; border-radius:8px; background:#181818; }
      .uav-card input { margin-right:0.35rem; }
      .uav-thumb { height:120px; background:#000; display:flex; align-items:center; justify-content:center; margin-bottom:0.5rem; }
      .uav-thumb img { max-height:120px; max-width:100%; object-fit:cover; }
      .thumb-missing { font-size:0.8rem; color:#777; }
      .error { color:#ff8080; }
      /* Modal styles */
      .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.8); }
      .modal-content { margin: 5% auto; display: block; max-width: 80%; max-height: 80vh; border: 2px solid #6cf; border-radius: 8px; }
      .modal-caption { margin: auto; display: block; width: 80%; text-align: center; color: #ccc; padding: 10px 0; font-size: 1.2rem; }
      .close-btn { position: absolute; top: 15px; right: 35px; color: #f1f1f1; font-size: 40px; font-weight: bold; cursor: pointer; transition: 0.3s; }
      .close-btn:hover, .close-btn:focus { color: #bbb; text-decoration: none; cursor: pointer; }
      #uavSearch { width: 100%; padding: 0.5rem; margin-bottom: 1rem; background: #333; border: 1px solid #555; color: #eee; border-radius: 4px; }
    </style>
    <script>
      function filterUAVs() {
        const input = document.getElementById('uavSearch');
        const filter = input.value.toLowerCase();
        const cards = document.getElementsByClassName('uav-card');
        for (let i = 0; i < cards.length; i++) {
          const text = cards[i].textContent || cards[i].innerText;
          if (text.toLowerCase().indexOf(filter) > -1) {
            cards[i].style.display = "";
          } else {
            cards[i].style.display = "none";
          }
        }
      }

      function openModal(imgSrc, imgName, gtCars) {
        const modal = document.getElementById('uavModal');
        const modalImg = document.getElementById('modalImage');
        const caption = document.getElementById('modalCaption');
        modal.style.display = "block";
        // Use the full-res route if available, or just the src
        // We will construct the full-res URL from the name if possible, or just use src
        // Actually, the user asked for a dedicated route /uav-image-full/<path>
        // We can infer the path from the image name or just pass it.
        // Let's assume imgSrc is the thumbnail URL. We can try to replace 'uav-image' with 'uav-image-full'
        // if the URL structure matches.
        let fullSrc = imgSrc.replace("/uav-image/", "/uav-image-full/");
        modalImg.src = fullSrc;
        caption.innerHTML = `<strong>${imgName}</strong><br>GT: ${gtCars} cars`;
      }

      function closeModal() {
        document.getElementById('uavModal').style.display = "none";
      }

      // Close modal when clicking outside image
      window.onclick = function(event) {
        const modal = document.getElementById('uavModal');
        if (event.target == modal) {
          modal.style.display = "none";
        }
      }
    </script>
  </head>
  <body>
    <h1>10-Agent Mini Society</h1>
    <div class="form-panel">
      <h3>Configuration</h3>
      <form method="post" action="{{ url_for('update_config') }}">
        
        <!-- Architecture -->
        <div style="margin-bottom:1rem;">
          <label><strong>Architecture:</strong></label>
          <label style="margin-left:0.5rem;">
            <input type="radio" name="architecture_mode" value="flat" {% if config.architecture_mode == 'flat' %}checked{% endif %}>
            Flat
          </label>
          <label style="margin-left:1rem;">
            <input type="radio" name="architecture_mode" value="hierarchical" {% if config.architecture_mode == 'hierarchical' %}checked{% endif %}>
            Hierarchical
          </label>
        </div>

        <!-- Selection Policy -->
        <div style="margin-bottom:1rem;">
          <label><strong>Selection Policy:</strong></label>
          <label style="margin-left:0.5rem;">
            <input type="checkbox" name="use_softmax" value="true" {% if config.use_softmax %}checked{% endif %} onchange="document.getElementById('temp_control').style.display = this.checked ? 'block' : 'none'">
            Use Softmax (vs Greedy)
          </label>
          <div id="temp_control" style="margin-left:1.5rem; margin-top:0.5rem; display: {% if config.use_softmax %}block{% else %}none{% endif %};">
            <label>Temperature (&tau;): <span id="temp_val">{{ config.softmax_temperature }}</span></label>
            <br>
            <input type="range" name="softmax_temperature" min="0.1" max="3.0" step="0.1" value="{{ config.softmax_temperature }}" oninput="document.getElementById('temp_val').innerText = this.value">
            <p class="hint">Low &tau; = Greedy, High &tau; = Random</p>
          </div>
        </div>

        <!-- Exploration -->
        <div style="margin-bottom:1rem;">
          <label><strong>Exploration (UCB):</strong></label>
          <div style="margin-left:1.5rem;">
            <label>Coefficient (c): <span id="ucb_val">{{ config.exploration_coefficient }}</span></label>
            <br>
            <input type="range" name="exploration_coefficient" min="0.0" max="2.0" step="0.1" value="{{ config.exploration_coefficient }}" oninput="document.getElementById('ucb_val').innerText = this.value">
            <p class="hint">c=0: No UCB. c>0: Explore less visited agents.</p>
          </div>
        </div>

        <!-- Fairness -->
        <div style="margin-bottom:1rem;">
          <label><strong>Fairness:</strong></label>
          <label style="margin-left:0.5rem;">
            <input type="checkbox" name="enable_fairness" value="true" {% if config.enable_fairness %}checked{% endif %} onchange="document.getElementById('fair_control').style.display = this.checked ? 'block' : 'none'">
            Enable Fairness Term
          </label>
          <div id="fair_control" style="margin-left:1.5rem; margin-top:0.5rem; display: {% if config.enable_fairness %}block{% else %}none{% endif %};">
            <label>Strength (&lambda;): <span id="fair_val">{{ config.fairness_lambda }}</span></label>
            <br>
            <input type="range" name="fairness_lambda" min="0.0" max="2.0" step="0.1" value="{{ config.fairness_lambda }}" oninput="document.getElementById('fair_val').innerText = this.value">
          </div>
        </div>

        <button type="submit">Update Configuration</button>
      </form>
    </div>



    <div class="form-panel">
      <form method="post" action="{{ url_for('ask') }}">
        <label for="task_input"><strong>Enter tasks (one per line):</strong></label>
        <textarea id="task_input" name="question" placeholder="Example:
1+1
how to change a tire
count cars in original/imgs/00001_frame000001_original.jpg
Use 2 drones to survey area A at 5m"></textarea>
        <p class="hint">
          Each non-empty line becomes its own task. Task type is inferred automatically.
          For UAV tasks, include 'count cars' and the image filename.
          For UAV missions, describe the mission (e.g. "survey", "drones").
        </p>
        <div style="margin-top:0.5rem;">
          <button type="submit">Submit Tasks</button>
        </div>
      </form>
      <h3 style="margin-top:1.5rem;">Batch experiments</h3>
      <form method="post" action="{{ url_for('run_batch_route') }}">
        <label for="batch_kind">Mode:&nbsp;</label>
        <select name="batch_kind" id="batch_kind">
          <option value="math">Math</option>
          <option value="reasoning">Reasoning</option>
          <option value="mixed">Mixed</option>
          <option value="uav_cars">UAV car count</option>
        </select>
        &nbsp;&nbsp;
        <label for="batch_n">Tasks:&nbsp;</label>
        <input type="number" name="batch_n" id="batch_n" min="1" max="200" value="25" style="width:80px;">
        &nbsp;&nbsp;
        <button type="submit">Run batch</button>
        <p class="hint">Choosing "UAV car count" samples N random images from the roundabout dataset.</p>
      </form>
      <h3 style="margin-top:1rem;">Select specific UAV images</h3>
      {% if uav_loader_error %}
        <p class="error">{{ uav_loader_error }}</p>
      {% elif uav_preview_tasks %}
        <input type="text" id="uavSearch" onkeyup="filterUAVs()" placeholder="Filter images by name or car count...">
        <form method="post" action="{{ url_for('run_uav_selection') }}">
          <div class="uav-preview-grid">
            {% for preview in uav_preview_tasks %}
              <label class="uav-card">
                <div>
                  <input type="checkbox" name="image_name" value="{{ preview.image_name }}">
                  <strong>{{ preview.image_name }}</strong>
                </div>
                <div class="uav-thumb">
                  {% if preview.image_rel %}
                    <img src="{{ url_for('uav_image', rel_path=preview.image_rel) }}" 
                         alt="{{ preview.image_name }}"
                         onclick="event.preventDefault(); openModal(this.src, '{{ preview.image_name }}', '{{ preview.num_cars }}')"
                         style="cursor: pointer;">
                  {% else %}
                    <div class="thumb-missing">image unavailable</div>
                  {% endif %}
                </div>
                <div class="uav-meta">{{ preview.num_cars }} cars</div>
              </label>
            {% endfor %}
          </div>
          <button type="submit">Run selected UAV tasks</button>
          <p class="hint">Check one or more images above before submitting.</p>
        </form>
      {% else %}
        <p class="hint">No preview rows found in uav_roundabout_tasks.csv.</p>
      {% endif %}
    </div>

    <h3>Agents</h3>
    <div style="margin-bottom: 10px; font-size: 0.9em; color: #aaa;">
      <strong>Current Policy:</strong> 
      {% if state.use_softmax %}Softmax (&tau;={{ "%.2f"|format(state.softmax_temperature) }}){% else %}Greedy{% endif %} |
      <strong>Exploration (UCB):</strong> c={{ "%.2f"|format(state.exploration_coefficient) }} |
      <strong>Fairness:</strong> {% if state.enable_fairness %}On (&lambda;={{ "%.2f"|format(state.fairness_lambda) }}){% else %}Off{% endif %}
    </div>
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Name</th>
          <th>Role</th>
          <th>Leader</th>
          <th>Tasks</th>
          <th>Successes</th>
          <th>Failures</th>
          <th>Success Rate</th>
          <th>Credit (role)</th>
          <th>Suspicious</th>
        </tr>
      </thead>
      <tbody>
        {% for agent in agents %}
        <tr>
          <td>{{ agent.agent_id }}</td>
          <td>{{ agent.name }}</td>
          <td>{{ agent.role }}</td>
          <td>{{ agent.leader_name }}</td>
          <td>{{ agent.tasks_done }}</td>
          <td>{{ agent.successes }}</td>
          <td>{{ agent.failures }}</td>
          <td>{{ agent.success_rate }}</td>
          <td>{{ agent.credit_means }}</td>
          <td>{{ agent.suspicious_count }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>

    <h2>Agent Scoreboards</h2>
    {% for board in agent_scoreboards %}
      {% if board.rows %}
        <h3>{{ board.label }}</h3>
        <table>
          <thead>
            <tr>
              <th>Agent</th>
              <th>Role</th>
              <th>Exposure</th>
              <th>Successes</th>
              <th>Failures</th>
              <th>Credit</th>
              <th>Explore</th>
              <th>Fairness</th>
              <th>Score</th>
            </tr>
          </thead>
          <tbody>
            {% for row in board.rows %}
            <tr>
              <td>{{ row.agent_id }} – {{ row.name }}</td>
              <td>{{ row.role }}</td>
              <td>{{ row.exposure }}</td>
              <td>{{ row.successes }}</td>
              <td>{{ row.failures }}</td>
              <td>{{ '%.3f'|format(row.credit) }}</td>
              <td>{{ '%.3f'|format(row.explore) }}</td>
              <td>{{ '%.3f'|format(row.fairness) }}</td>
              <td>{{ '%.3f'|format(row.score) }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      {% endif %}
    {% endfor %}

    {% if leaders %}
      <h2>Leaders</h2>
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Name</th>
            <th>Role</th>
            <th>Workers</th>
            <th>Metrics</th>
          </tr>
        </thead>
        <tbody>
          {% for leader in leaders %}
          <tr>
            <td>{{ leader.leader_id }}</td>
            <td>{{ leader.name }}</td>
            <td>{{ leader.role }}</td>
            <td>{{ leader.workers_display }}</td>
            <td>
              {% for metric in leader.metrics %}
                <div><strong>{{ metric.label }}</strong>: tasks={{ metric.tasks }}, success={{ metric.success_rate }}, credit={{ metric.credit_mean }}, exposure={{ metric.exposure }}, suspicious={{ metric.suspicious }}</div>
              {% endfor %}
            </td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    {% endif %}

    {% if highlight_scores %}
      <h2>Selection scores for task #{{ highlight_scores['task_id'] }}</h2>
      <canvas id="scoreChart" height="120"></canvas>
      <script>
        const ctx = document.getElementById('scoreChart').getContext('2d');
        const labels = {{ highlight_scores['labels'] | tojson }};
        const values = {{ highlight_scores['values'] | tojson }};
        new Chart(ctx, {
          type: 'bar',
          data: { labels: labels, datasets: [{ label: 'Selection score', data: values }] },
          options: { responsive: true, scales: { y: { beginAtZero: true } } }
        });
      </script>
            {% if highlight_scores.agent_components %}
            <div style="margin-top:1rem; padding:10px; background:#222; border:1px solid #444;">
                <strong>Selected Agent: {{ highlight_scores.agent_name }}</strong><br>
                Credit: {{ "%.3f"|format(highlight_scores.agent_components.get('credit_mean', 0)) }}<br>
                Exploration: {{ "%.3f"|format(highlight_scores.agent_components.get('exploration_term', 0)) }}<br>
                Fairness: {{ "%.3f"|format(highlight_scores.agent_components.get('fairness_penalty', 0)) }}<br>
                <strong>Total Score: {{ "%.3f"|format(highlight_scores.agent_components.get('total_score', 0)) }}</strong>
            </div>
            {% endif %}

            {% if highlight_scores['leader_components'] %}
        {% set lc = highlight_scores['leader_components'] %}
        <p>Leader score ({{ highlight_scores['leader_name'] or 'n/a' }}): credit={{ '%.3f'|format(lc.get('credit_mean', 0.0)) }}, explore={{ '%.3f'|format(lc.get('exploration_term', 0.0)) }}, fair={{ '%.3f'|format(lc.get('fairness_penalty', 0.0)) }}, total={{ '%.3f'|format(lc.get('total_score', 0.0)) }}</p>
      {% endif %}
    {% endif %}

    {% if uav_task_rows %}
      <h2>UAV Car-Count Results</h2>
      <table>
        <thead>
          <tr>
            <th>Task</th>
            <th>Image</th>
            <th>GT cars</th>
            <th>Predicted</th>
            <th>Abs error</th>
            <th>Reward</th>
            <th>Success</th>
            <th>Agent</th>
            <th>Leader</th>
          </tr>
        </thead>
        <tbody>
          {% for row in uav_task_rows %}
          <tr class="{% if row.success %}success{% else %}failure{% endif %}">
            <td>{{ row.task_id }}</td>
            <td>
              {{ row.image_name }}<br>
                {% if row.image_rel %}
                  <img src="{{ url_for('uav_image', rel_path=row.image_rel) }}" 
                       alt="{{ row.image_name }}" 
                       style="max-height:100px; margin-top:0.25rem; cursor: pointer;"
                       onclick="openModal(this.src, '{{ row.image_name }}', '{{ row.ground_truth if row.ground_truth is not none else 'n/a' }}')">
                {% endif %}
            </td>
            <td>{{ row.ground_truth if row.ground_truth is not none else 'n/a' }}</td>
            <td>{{ row.prediction if row.prediction is not none else 'n/a' }}</td>
            <td>{{ row.abs_error if row.abs_error is not none else 'n/a' }}</td>
            <td>{{ '%.2f'|format(row.reward or 0.0) }}</td>
            <td>{{ row.success }}</td>
            <td>{{ row.agent }}</td>
            <td>{{ row.leader or '—' }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
      {% if uav_reward_values %}
        <h3>UAV numeric reward trend</h3>
        <canvas id="uavRewardChart" height="100"></canvas>
        <script>
          const uavCtx = document.getElementById('uavRewardChart').getContext('2d');
          new Chart(uavCtx, {
            type: 'line',
            data: { labels: {{ uav_reward_labels | tojson }}, datasets: [{ label: 'Reward', data: {{ uav_reward_values | tojson }}, fill:false, borderColor:'#6cf' }] },
            options: { responsive: true, scales: { y: { beginAtZero: true, max: 1 } } }
          });
        </script>
      {% endif %}
    {% endif %}

    <h2>Task History</h2>
    <table>
      <thead>
        <tr>
          <th>Task ID</th>
          <th>Leader</th>
          <th>Agent</th>
          <th>Type</th>
          <th>Input</th>
          <th>Output</th>
          <th>Agent Score</th>
          <th>Leader Score</th>
          <th>Evaluation</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>
        {% for task in tasks %}
        {% set sc = task.selection_components or {} %}
        <tr class="{% if task.success is none %}pending{% elif task.success %}success{% else %}failure{% endif %}">
          <td>{{ task.task_id }}</td>
          <td>{{ task.leader_name or '—' }}</td>
          <td>{{ task.assigned_agent_name or task.assigned_agent_id }}</td>
          <td>{{ task.task_type }}</td>
          <td>
            {% if task.task_type == 'uav_car_count' %}
              {{ task.input_text }}<br>
              {% if task.image_path %}
                <small>{{ task.image_path }}</small>
              {% endif %}
            {% elif task.task_type == 'uav_mission' %}
              <strong>Mission:</strong> {{ task.input_text }}
            {% else %}
              {{ task.input_text }}
            {% endif %}
          </td>
          <td>
            {% if task.task_type == 'uav_mission' %}
              <pre style="white-space: pre-wrap; font-size: 0.9rem;">{{ task.output }}</pre>
            {% elif task.task_type == 'uav_car_count' and '\n' in task.output %}
               <pre style="white-space: pre-wrap; font-size: 0.9rem;">{{ task.output }}</pre>
            {% else %}
              {{ task.output }}
            {% endif %}
          </td>
          <td>
            {% if sc %}
              total={{ '%.3f'|format(sc.get('total_score', 0.0)) }},
              credit={{ '%.3f'|format(sc.get('credit_mean', 0.0)) }},
              explore={{ '%.3f'|format(sc.get('exploration_term', 0.0)) }},
              fair={{ '%.3f'|format(sc.get('fairness_penalty', 0.0)) }}
            {% else %}
              n/a
            {% endif %}
          </td>
          <td>
            {% set lsc = task.leader_selection_components or {} %}
            {% if lsc %}
              total={{ '%.3f'|format(lsc.get('total_score', 0.0)) }},
              credit={{ '%.3f'|format(lsc.get('credit_mean', 0.0)) }},
              explore={{ '%.3f'|format(lsc.get('exploration_term', 0.0)) }},
              fair={{ '%.3f'|format(lsc.get('fairness_penalty', 0.0)) }}
            {% else %}
              n/a
            {% endif %}
          </td>
          <td>
            {% if task.success is none %}
              Pending – {{ task.evaluation_reason or 'awaiting human feedback' }}
            {% elif task.success %}
              Success – {{ task.evaluation_reason }}
            {% else %}
              Failure – {{ task.evaluation_reason }}
            {% endif %}
            {% if task.numeric_match is not none %}
              (numeric={{ task.numeric_match }})
            {% endif %}
            {% if task.verdict_match is not none %}
              (verdict={{ task.verdict_match }})
            {% endif %}
            {% if task.partial_credit %}
              (partial credit)
            {% endif %}
            {% if task.ground_truth %}
              (expected={{ task.ground_truth }})
            {% endif %}
            {% if task.used_llm %}
              [LLM]
            {% endif %}
          </td>
          <td>
            <a href="{{ url_for('index', highlight_task=task.task_id) }}">Details</a>
            <form style="display:inline;" method="post" action="{{ url_for('feedback', task_id=task.task_id, direction='up') }}">
              <button class="feedback-btn good" title="Thumbs up">👍</button>
            </form>
            <form style="display:inline;" method="post" action="{{ url_for('feedback', task_id=task.task_id, direction='down') }}">
              <button class="feedback-btn bad" title="Thumbs down">👎</button>
            </form>
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    
    <!-- The Modal -->
    <div id="uavModal" class="modal">
      <span class="close-btn" onclick="closeModal()">&times;</span>
      <img class="modal-content" id="modalImage">
      <div id="modalCaption" class="modal-caption"></div>
    </div>
  </body>
</html>
"""


@app.get("/")
def index():
    agents_context = []
    for agent in STATE.agents:
        completed = agent.successes + agent.failures
        success_rate = f"{(agent.successes / completed * 100):.1f}%" if completed else "N/A"
        credit = agent.credits.get(agent.role, {})
        leader = STATE.agent_to_leader.get(agent.agent_id)
        agents_context.append(
            {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "role": agent.role,
                "leader_name": leader.name if leader else "—",
                "tasks_done": agent.tasks_done,
                "successes": agent.successes,
                "failures": agent.failures,
                "success_rate": success_rate,
                "credit_means": f"{credit.get('credit_mean', 0.5):.2f}",
                "suspicious_count": agent.suspicious_count,
            }
        )

    agent_scoreboards = _build_agent_scoreboards(STATE)

    agent_lookup = {agent.agent_id: agent for agent in STATE.agents}
    leaders_context = []
    for leader in STATE.leaders:
        worker_names = [
            agent_lookup[agent_id].name
            for agent_id in leader.worker_ids
            if agent_id in agent_lookup
        ]
        metrics_rows = []
        for task_type, label in [
            (TASK_TYPE_MATH, "Math"),
            (TASK_TYPE_REASONING, "Reasoning"),
            (TASK_TYPE_UAV, "UAV Car Count"),
        ]:
            metrics = leader.group_metrics.get(task_type)
            successes = metrics.successes if metrics else 0
            failures = metrics.failures if metrics else 0
            total = successes + failures
            success_rate = f"{(successes / total * 100):.1f}%" if total else "0.0%"
            credit_mean = metrics.credit_mean if metrics else 0.5
            metrics_rows.append(
                {
                    "task_type": task_type,
                    "label": label,
                    "tasks": metrics.n if metrics else 0,
                    "success_rate": success_rate,
                    "credit_mean": f"{credit_mean:.2f}",
                    "exposure": metrics.exposure_count if metrics else 0,
                    "suspicious": metrics.suspicious_events if metrics else 0,
                }
            )
        leaders_context.append(
            {
                "leader_id": leader.leader_id,
                "name": leader.name,
                "role": leader.role,
                "workers_display": ", ".join(worker_names) if worker_names else "—",
                "metrics": metrics_rows,
            }
        )

    highlight_scores = None
    highlight_id = request.args.get("highlight_task")
    if highlight_id:
        try:
            tid = int(highlight_id)
            highlight = next((t for t in STATE.tasks if t.task_id == tid), None)
            if highlight and highlight.selection_scores:
                labels = [agent.name for agent in STATE.agents]
                values = [highlight.selection_scores.get(agent.agent_id, 0.0) for agent in STATE.agents]
                highlight_scores = {
                    "task_id": highlight.task_id,
                    "labels": labels,
                    "values": values,
                    "leader_components": highlight.leader_selection_components or {},
                    "leader_name": highlight.leader_name,
                    "agent_components": highlight.selection_components or {},
                    "agent_name": highlight.assigned_agent_name,
                }
            else:
                highlight_scores = None
        except ValueError:
            highlight_scores = None

    uav_preview_tasks, uav_loader_error = _build_uav_preview()
    uav_task_rows = _uav_task_rows(STATE)
    uav_reward_labels: list[str] = []
    uav_reward_values: list[float] = []
    for task in STATE.tasks:
        if task.task_type != TASK_TYPE_UAV or task.numeric_reward is None:
            continue
        uav_reward_labels.append(str(task.task_id))
        uav_reward_values.append(round(float(task.numeric_reward), 3))

    return render_template_string(
        TEMPLATE,
        agents=agents_context,
        leaders=leaders_context,
        tasks=list(reversed(STATE.tasks[-50:])),
        highlight_scores=highlight_scores,
        architecture_mode=STATE.architecture_mode,
        state=STATE,
        agent_scoreboards=agent_scoreboards,
        uav_preview_tasks=uav_preview_tasks,
        uav_loader_error=uav_loader_error,
        config=STATE,  # Pass the full state object to access config fields
        uav_task_rows=uav_task_rows,
        uav_reward_labels=uav_reward_labels,
        uav_reward_values=uav_reward_values,
        current_task_mode=CURRENT_TASK_MODE,
    )


@app.post("/set-mode")
def set_mode():
    mode = request.form.get("mode", ARCH_MODE_HIER)
    if mode not in (ARCH_MODE_FLAT, ARCH_MODE_HIER):
        mode = ARCH_MODE_HIER
    STATE.architecture_mode = mode
    return redirect(url_for("index"))


@app.post("/update_config")
def update_config():
    # Architecture
    mode = request.form.get("architecture_mode")
    if mode in [ARCH_MODE_FLAT, ARCH_MODE_HIER]:
        STATE.architecture_mode = mode
        
    # Softmax
    STATE.use_softmax = (request.form.get("use_softmax") == "true")
    try:
        STATE.softmax_temperature = float(request.form.get("softmax_temperature", 0.8))
    except ValueError:
        pass
        
    # Exploration
    try:
        STATE.exploration_coefficient = float(request.form.get("exploration_coefficient", 0.0))
    except ValueError:
        pass
        
    # Fairness
    STATE.enable_fairness = (request.form.get("enable_fairness") == "true")
    try:
        STATE.fairness_lambda = float(request.form.get("fairness_lambda", 0.0))
    except ValueError:
        pass
        
    return redirect(url_for("index"))





@app.post("/ask")
def ask():
    raw = request.form.get("question", "")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    
    for line in lines:
        process_user_line(STATE, line)
        
    return redirect(url_for("index"))


@app.post("/run-batch")
def run_batch_route():
    global CURRENT_TASK_MODE
    kind = request.form.get("batch_kind", "math")
    try:
        n = int(request.form.get("batch_n", "50"))
    except ValueError:
        n = 50
    n = max(1, min(n, 200))
    if kind == "uav_cars":
        use_leaders = STATE.architecture_mode == ARCH_MODE_HIER
        run_uav_batch(STATE, n, use_leaders=use_leaders)
        CURRENT_TASK_MODE = TASK_TYPE_UAV
    else:
        run_batch(STATE, kind, n)
    return redirect(url_for("index"))


@app.post("/feedback/<int:task_id>/<direction>")
def feedback(task_id: int, direction: str):
    positive = direction == "up"
    apply_feedback(STATE, task_id, positive)
    return redirect(url_for("index"))


@app.post("/run-uav-selection")
def run_uav_selection():
    global CURRENT_TASK_MODE
    selected = request.form.getlist("image_name")
    if selected:
        use_leaders = STATE.architecture_mode == ARCH_MODE_HIER
        run_uav_batch(
            STATE,
            n_tasks=len(selected),
            use_leaders=use_leaders,
            selected_image_names=selected,
        )
        CURRENT_TASK_MODE = TASK_TYPE_UAV
    return redirect(url_for("index"))


@app.get("/uav-image/<path:rel_path>")
def uav_image(rel_path: str):
    safe_path = (UAV_DATA_ROOT / Path(rel_path)).resolve()
    try:
        safe_path.relative_to(UAV_DATA_ROOT)
    except ValueError:
        abort(403)
    if not safe_path.exists():
        abort(404)
    return send_file(safe_path)


@app.get("/uav-image-full/<path:rel_path>")
def uav_image_full(rel_path: str):
    # Serve the same file, but this route allows the frontend to distinguish
    # and potentially we could serve a higher res version if we had it.
    # For now, it maps to the same file.
    safe_path = (UAV_DATA_ROOT / Path(rel_path)).resolve()
    try:
        safe_path.relative_to(UAV_DATA_ROOT)
    except ValueError:
        abort(403)
    if not safe_path.exists():
        abort(404)
    return send_file(safe_path)


if __name__ == "__main__":
    app.run(debug=True)
