
from ten_agent_web_demo.simulation import create_default_state, run_batch, execute_task

def test_simulation():
    print("Creating state...")
    state = create_default_state()
    print(f"Initial tasks: {len(state.tasks)}")
    
    print("Running batch (10 tasks)...")
    run_batch(state, "math", 10)
    print(f"Tasks after batch: {len(state.tasks)}")
    
    if len(state.tasks) > 0:
        print("Last task output:", state.tasks[-1].output)
        print("Last task success:", state.tasks[-1].success)
    else:
        print("ERROR: No tasks generated!")

if __name__ == "__main__":
    test_simulation()
