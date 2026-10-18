"""Explicit GUI smoke check; run separately from headless environment tests."""

from collections import deque
import copy
import json
from pathlib import Path
import sys
import tempfile
import tkinter as tk
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import MazeApp
from environment import Action, DELTAS
from qlearning import QLearningAgent, evaluate_greedy, train_one
from curriculum import train_budget


def finish_demo(app):
    """Run the actual UI callback without waiting for the animation clock."""
    for _ in range(app.env.max_steps + 1):
        if app.env.done:
            break
        if app.demo_timer is not None:
            app.root.after_cancel(app.demo_timer)
            app.demo_timer = None
        app.demo_tick()
    assert app.env.done


def main():
    temp = tempfile.TemporaryDirectory()
    root = tk.Tk()
    try:
        app = MazeApp(root, results_dir=temp.name)
        root.update()
        root.focus_force()
        root.update()
        root.event_generate("<KeyPress-Up>")
        root.update()
        assert (app.env.steps, app.env.total_reward, app.env.wall_hits) == (1, -5, 1)
        root.event_generate("<KeyPress-Right>")
        root.update()
        assert (app.env.position, app.env.total_reward) == ((1, 2), -6)
        root.event_generate("<KeyPress>", keycode=82)
        root.update()
        assert (app.env.steps, app.env.position) == (0, app.env.start)
        app.move_buttons[2].invoke()  # Down is blocked at the start.
        assert app.env.wall_hits == 1
        app.reset_button.invoke()
        assert app.env.steps == 0

        # Seed input must not be interpreted as movement keys by the window.
        app.demo_seed_entry.focus_set()
        root.update()
        app.demo_seed_entry.event_generate("<KeyPress-Right>")
        root.update()
        assert app.env.steps == 0
        app.demo_seed.set("invalid")
        app.start_demo()
        assert app.demo_agent is None and app.env.steps == 0
        app.demo_seed.set("0")
        app.demo_speed.set("ช้า")
        app.start_demo()
        assert app.demo_running and app.demo_timer is not None
        assert all(str(button.cget("state")) == "disabled" for button in app.move_buttons)
        app.move(Action.RIGHT)
        assert app.env.steps == 0  # No manual interference while the demo is active.
        app.toggle_demo_pause()
        assert not app.demo_running and app.demo_timer is None
        paused_steps = app.env.steps
        app.demo_tick()
        assert app.env.steps == paused_steps
        app.toggle_demo_pause()
        assert app.demo_running and app.demo_timer is not None
        # Drive the scheduled callback without real-time waits; cancel each pending
        # timer before advancing to avoid duplicate callbacks in this smoke check.
        finish_demo(app)
        assert not app.demo_running and app.demo_timer is None
        demo_stats = (app.env.steps, app.env.total_reward, app.env.wall_hits)
        from agents import RandomAgent, episode_seed
        from baseline import run_episode
        from environment import MazeEnv
        record = run_episode(MazeEnv(), RandomAgent(episode_seed(0, 1)), 0, 1)
        assert demo_stats == (record["steps"], record["env_reward"], record["wall_hits"])
        app.start_demo()
        assert app.demo_timer is not None
        app.reset_button.invoke()
        assert app.demo_agent is None and app.demo_timer is None and app.env.steps == 0

        app.demo_seed.set("42")
        app.start_demo()
        app.reset()
        assert app.demo_agent is None and not app.demo_running

        app.start_q_demo()
        assert app.demo_agent is None
        assert "train.cmd" in app.feedback.cget("text")
        fixtures = Path(temp.name) / "qlearning-20200101"
        models = fixtures / "models"
        models.mkdir(parents=True)
        untrained_path, trained_path = models / "untrained.json", models / "trained.json"
        QLearningAgent(app.env.rows, app.env.cols).save(untrained_path, app.env, metadata={"episodes_completed": 0})
        trained, _, _ = train_one(0, episodes=150)
        trained.save(trained_path, app.env, metadata={"episodes_completed": 150})
        (fixtures / "summary.json").write_text(json.dumps({
            "status": "complete", "demo_model": "models/trained.json"}), encoding="utf-8")
        app.q_start_button.invoke()  # Automatically chooses the completed run's declared demo.
        assert app.demo_kind == "Q-learning" and app.demo_running
        before_q = copy.deepcopy(app.q_agent.q)
        before_updates, before_rng = app.q_agent.updates, app.q_agent.rng.getstate()
        app.toggle_demo_pause()
        assert app.demo_timer is None
        app.toggle_demo_pause()
        finish_demo(app)
        assert app.env.terminated and app.env.steps == 18
        stats, _ = evaluate_greedy(app.q_agent, MazeEnv())
        assert (app.env.steps, app.env.total_reward, app.env.wall_hits) == (stats["steps"], stats["env_reward"], stats["wall_hits"])
        assert app.q_agent.q == before_q and app.q_agent.updates == before_updates
        assert app.q_agent.rng.getstate() == before_rng
        with patch("app.filedialog.askopenfilename", return_value=str(untrained_path)):
            app.q_load_button.invoke()
        assert app.q_agent.metadata["episodes_completed"] == 0
        app.start_q_demo()
        finish_demo(app)
        assert app.env.truncated and app.env.wall_hits == 300
        assert app.q_agent.updates == 0
        wrong = models / "wrong.json"
        QLearningAgent(1, 2).save(wrong, MazeEnv(("SG",)))
        previous = (app.env.position, app.env.steps, app.q_model_path)
        assert app.load_q_model(wrong) is False
        assert (app.env.position, app.env.steps, app.q_model_path) == previous
        with patch("app.filedialog.askopenfilename", return_value=""):
            app.q_load_button.invoke()
        assert app.q_model_path == previous[2]
        app.load_q_model(trained_path)
        app.reset()

        # A matched phase-4 run should be selected by the method picker, with
        # the old phase-3 checkpoint still available through the file picker.
        comparison = Path(temp.name) / "exploration-20200101"
        (comparison / "models").mkdir(parents=True)
        demo_models = {}
        for method in ("plain", "curriculum", "curriculum_bonus"):
            matched, _, _, _ = train_budget(0, method=method, budget=15000)
            model_path = comparison / "models" / f"{method}.json"
            matched.save(model_path, MazeEnv(), metadata={
                "method": method, "training_budget_actions": 15000,
                "training_env_steps": matched.updates})
            demo_models[method] = f"models/{method}.json"
        (comparison / "summary.json").write_text(json.dumps({
            "status": "complete", "demo_models": demo_models}), encoding="utf-8")
        for label, method in (("Curriculum", "curriculum"), ("Q-learning", "plain"),
                              ("Curric + Bonus", "curriculum_bonus")):
            app.q_method.set(label)
            app.q_method_box.event_generate("<<ComboboxSelected>>")
            root.update()
            assert app.q_agent is None and app.demo_timer is None
            app.q_start_button.invoke()
            assert app.demo_kind == label
            assert app.q_model_path == (comparison / demo_models[method]).resolve()
            initial = (copy.deepcopy(app.q_agent.q), app.q_agent.updates, app.q_agent.rng.getstate())
            finish_demo(app)
            assert app.env.terminated and app.env.steps == 18
            assert initial == (app.q_agent.q, app.q_agent.updates, app.q_agent.rng.getstate())
            app.reset()
        app.q_method.set("Curriculum")
        app.change_q_method()
        assert app.load_q_model(trained_path)
        assert app.q_method.get() == "Q-learning"
        app.reset()

        queue = deque([(app.env.start, [])])
        seen = {app.env.start}
        route = None
        while queue:
            position, path = queue.popleft()
            if position == app.env.goal:
                route = path
                break
            for action, (dr, dc) in DELTAS.items():
                candidate = position[0] + dr, position[1] + dc
                if candidate not in seen and app.env.is_walkable(candidate):
                    seen.add(candidate)
                    queue.append((candidate, path + [action]))
        assert route
        for action in route:
            app.move(action)
        root.update()
        assert app.env.terminated and not app.env.truncated
        assert all(str(button.cget("state")) == "disabled" for button in app.move_buttons)
        steps = app.env.steps
        root.event_generate("<KeyPress-Left>")
        root.update()
        assert app.env.steps == steps
        app.reset()
        for _ in range(app.env.max_steps):
            app.move(Action.UP)
        root.update()
        assert app.env.truncated and not app.env.terminated
        assert all(str(button.cget("state")) == "disabled" for button in app.move_buttons)
        app.reset()
        for width, height in [(1080, 790), (880, 720)]:
            root.geometry(f"{width}x{height}")
            root.update()
            assert app.canvas.winfo_width() > 300
            assert app.canvas.winfo_height() > 300
            assert app.reset_button.winfo_rooty() + app.reset_button.winfo_height() <= root.winfo_rooty() + root.winfo_height()
            assert app.feedback.winfo_rooty() + app.feedback.winfo_height() <= root.winfo_rooty() + root.winfo_height()
            assert app.demo_start_button.winfo_ismapped()
            assert app.demo_speed_box.winfo_rootx() + app.demo_speed_box.winfo_width() <= root.winfo_rootx() + root.winfo_width()
            assert app.q_start_button.winfo_ismapped()
            assert app.q_model_label.winfo_rootx() + app.q_model_label.winfo_width() <= root.winfo_rootx() + root.winfo_width()
            assert app.q_model_label.winfo_width() >= app.q_model_label.winfo_reqwidth()
        app.start_q_demo()
        assert app.demo_timer is not None  # close() below must cancel this timer.
        print("GUI checks passed: manual/random, legacy Q, matched three-method picker, frozen demos, pause/reset/close, dimensions")
    finally:
        app.close()
        temp.cleanup()


if __name__ == "__main__":
    main()
