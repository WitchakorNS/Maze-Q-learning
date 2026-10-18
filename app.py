"""Manual play, random baseline, and frozen Q-learning demonstrations."""

import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

from environment import Action, MazeEnv
from agents import RandomAgent, episode_seed
from qlearning import QLearningAgent, latest_demo_model
from curriculum import latest_comparison_model


BG = "#0c1421"
PANEL = "#142033"
TEXT = "#e5edf7"
MUTED = "#9cacc3"
TEAL = "#5ee0bc"
GOLD = "#f2c66d"
RED = "#ff9b9b"
FONT = "Leelawadee UI"
Q_METHODS = {"Q-learning": "plain", "Curriculum": "curriculum", "Curric + Bonus": "curriculum_bonus"}


class MazeApp:
    def __init__(self, root, *, results_dir=None):
        self.root = root
        self.env = MazeEnv()
        self.message = "พาตัวละครจาก S ไปยัง G โดยใช้จำนวนก้าวให้น้อยที่สุด"
        self.message_color = MUTED
        self.last_action = "—"
        self.last_event = "reset"
        self.demo_agent = None
        self.demo_running = False
        self.demo_timer = None
        self.demo_kind = "Random"
        self.q_agent = None
        self.q_model_path = None
        self.results_dir = Path(results_dir) if results_dir is not None else Path(__file__).resolve().parent / "results"
        root.title("Maze RL · Phase 5 · Exploration Bonus")
        root.geometry("1080x790")
        root.minsize(880, 720)
        root.configure(bg=BG)
        root.option_add("*Font", (FONT, 11))

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("Maze.TButton", font=(FONT, 11), background="#25364e",
                        foreground=TEXT, padding=(12, 9), borderwidth=0)
        style.map("Maze.TButton", background=[("active", "#354d6b")],
                  foreground=[("disabled", "#68798f")])
        style.configure("Reset.TButton", font=(FONT, 11, "bold"), background=TEAL,
                        foreground=BG, padding=(12, 10), borderwidth=0)
        style.map("Reset.TButton", background=[("active", "#87ebd0")])

        main = tk.Frame(root, bg=BG, padx=28, pady=20)
        main.pack(fill="both", expand=True)
        self.label(main, "MAZE LAB   /   REINFORCEMENT LEARNING", 10, TEAL).pack(anchor="w")
        self.label(main, "เขาวงกต: เรียนรู้และสำรวจด้วย AI", 22, TEXT, True).pack(anchor="w", pady=(5, 3))
        self.label(main, "ระยะที่ 5  ·  ฝึกเทียบด้วย exploration.cmd  ·  เดโมจาก S จริงด้วย ε = 0 ไม่มีโบนัส", 11, MUTED).pack(anchor="w")

        content = tk.Frame(main, bg=BG)
        content.pack(fill="both", expand=True, pady=(20, 12))
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=0, minsize=285)
        content.rowconfigure(0, weight=1)
        map_area = tk.Frame(content, bg=BG)
        map_area.grid(row=0, column=0, sticky="nsew", padx=(0, 24))
        self.canvas = tk.Canvas(map_area, bg=BG, highlightthickness=0,
                                width=610, height=560)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda event: self.render_maze())
        self.label(map_area, "S จุดเริ่ม     G เป้าหมาย     ● ตัวละคร     · ช่องที่เคยเดิน", 10, MUTED).pack(anchor="w", pady=(6, 0))

        sidebar = tk.Frame(content, bg=PANEL, padx=16, pady=12)
        sidebar.grid(row=0, column=1, sticky="nsew")
        self.label(sidebar, "รอบการเล่น", 14, TEXT, True, PANEL).pack(anchor="w")
        self.status = self.label(sidebar, "กำลังเล่น", 11, TEAL, bg=PANEL)
        self.status.pack(anchor="w", pady=(0, 6))
        self.values = {}
        for key, caption in [("steps", "จำนวนก้าว"), ("reward", "รางวัลสะสม"),
                             ("walls", "ชนกำแพง"), ("position", "แถว / คอลัมน์")]:
            row = tk.Frame(sidebar, bg=PANEL)
            row.pack(fill="x", pady=3)
            self.label(row, caption, 10, MUTED, bg=PANEL).pack(side="left")
            self.values[key] = self.label(row, "", 15, TEXT, True, PANEL)
            self.values[key].pack(side="right")

        self.label(sidebar, "รางวัลต่อการกระทำ", 11, TEXT, True, PANEL).pack(anchor="w", pady=(10, 4))
        for text, color in [("เดิน −1   ·   ชนกำแพง −5", MUTED), ("ถึงเป้าหมาย   +100", GOLD)]:
            self.label(sidebar, text, 10, color, bg=PANEL).pack(anchor="w", pady=2)
        self.label(sidebar, "300 ก้าวต่อรอบ · พิกัดเริ่มจาก 0", 10, MUTED, bg=PANEL).pack(anchor="w", pady=(6, 6))
        self.reset_button = ttk.Button(sidebar, text="เริ่มรอบใหม่  ·  R", style="Reset.TButton", command=self.reset)
        self.reset_button.pack(fill="x", pady=(4, 0))

        controls = tk.Frame(main, bg=BG)
        controls.pack(side="bottom", fill="x", before=content)
        arrows = tk.Frame(controls, bg=BG)
        arrows.pack(side="right")
        self.move_buttons = []
        for label, action, row, col in [("↑", Action.UP, 0, 1), ("←", Action.LEFT, 1, 0),
                                        ("↓", Action.DOWN, 1, 1), ("→", Action.RIGHT, 1, 2)]:
            button = ttk.Button(arrows, text=label, width=4, style="Maze.TButton",
                                command=lambda a=action: self.move(a))
            button.grid(row=row, column=col, padx=2, pady=2)
            self.move_buttons.append(button)
        details = tk.Frame(controls, bg=BG)
        details.pack(side="left", fill="x", expand=True, padx=(0, 16))
        demo_controls = tk.Frame(details, bg=BG)
        demo_controls.pack(anchor="w", pady=(0, 4))
        self.demo_start_button = ttk.Button(demo_controls, text="สุ่มเดิน 1 รอบ", style="Maze.TButton", command=self.start_demo)
        self.demo_start_button.pack(side="left", padx=(0, 6))
        self.demo_pause_button = ttk.Button(demo_controls, text="พัก", width=6, style="Maze.TButton", command=self.toggle_demo_pause)
        self.demo_pause_button.pack(side="left", padx=(0, 8))
        self.label(demo_controls, "Seed", 10, MUTED).pack(side="left", padx=(0, 4))
        self.demo_seed = tk.StringVar(value="0")
        self.demo_seed_entry = ttk.Entry(demo_controls, textvariable=self.demo_seed, width=6)
        self.demo_seed_entry.pack(side="left", padx=(0, 8))
        self.demo_speed = tk.StringVar(value="ปกติ")
        self.demo_speed_box = ttk.Combobox(demo_controls, textvariable=self.demo_speed,
                                          values=("ช้า", "ปกติ", "เร็ว"), state="readonly", width=6)
        self.demo_speed_box.pack(side="left")
        q_controls = tk.Frame(details, bg=BG)
        q_controls.pack(anchor="w", pady=(0, 4))
        self.q_start_button = ttk.Button(q_controls, text="ให้ Q เดิน", style="Maze.TButton", command=self.start_q_demo)
        self.q_start_button.pack(side="left", padx=(0, 6))
        self.q_load_button = ttk.Button(q_controls, text="โหลด Q-table", style="Maze.TButton", command=self.load_q_model)
        self.q_load_button.pack(side="left", padx=(0, 8))
        self.q_method = tk.StringVar(value="Q-learning")
        self.q_method_box = ttk.Combobox(q_controls, textvariable=self.q_method,
                                        values=tuple(Q_METHODS), state="readonly", width=14)
        self.q_method_box.pack(side="left", padx=(0, 8))
        self.q_method_box.bind("<<ComboboxSelected>>", self.change_q_method)
        self.q_model_label = self.label(q_controls, "ยังไม่โหลดโมเดล", 10, MUTED)
        self.q_model_label.pack(side="left")
        self.label(details, "เล่นเอง: ลูกศร / WASD  ·  R เริ่มใหม่  ·  เก็บ baseline ด้วย baseline.cmd", 10, TEXT).pack(anchor="w")
        self.feedback = self.label(details, self.message, 10, MUTED)
        self.feedback.configure(wraplength=630, justify="left")
        self.feedback.pack(anchor="w", pady=(5, 0))
        self.transition = self.label(details, "", 10, MUTED)
        self.transition.pack(anchor="w", pady=(5, 0))

        # Bind on the window so arrow keys also work after a button has focus.
        root.bind("<KeyPress>", self.on_key)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh()
        root.after_idle(root.focus_set)

    @staticmethod
    def label(parent, text, size=11, color=TEXT, bold=False, bg=BG):
        return tk.Label(parent, text=text, bg=bg, fg=color,
                        font=(FONT, size, "bold" if bold else "normal"), anchor="w")

    def on_key(self, event):
        # Let text entry and the speed picker receive their own keyboard input.
        if isinstance(event.widget, (ttk.Entry, ttk.Combobox)):
            return
        key = event.keysym.lower()
        # Windows virtual keys keep WASD/R usable when Thai input is active.
        if sys.platform == "win32":
            key = {87: "w", 65: "a", 83: "s", 68: "d", 82: "r"}.get(event.keycode, key)
        mapping = {"up": Action.UP, "w": Action.UP, "right": Action.RIGHT, "d": Action.RIGHT,
                   "down": Action.DOWN, "s": Action.DOWN, "left": Action.LEFT, "a": Action.LEFT}
        if key in mapping:
            self.move(mapping[key])
            return "break"
        if key == "r":
            self.reset()
            return "break"

    def move(self, action, *, source="manual"):
        if self.env.done or (self.demo_agent is not None and source != "agent"):
            return
        previous = self.env.position
        state, reward, terminated, truncated, info = self.env.step(action)
        self.last_action = {Action.UP: "ขึ้น", Action.RIGHT: "ขวา", Action.DOWN: "ลง", Action.LEFT: "ซ้าย"}[action]
        self.last_event = info["event"]
        if terminated:
            self.message = f"ถึงเป้าหมาย! ใช้ {self.env.steps} ก้าว · กด R เพื่อเล่นเองอีกครั้ง"
            self.message_color = TEAL
        elif truncated:
            self.message = "ครบจำนวนก้าวแล้ว · กด R เพื่อเริ่มรอบใหม่"
            self.message_color = GOLD
        elif info["event"] == "wall":
            self.message = "ชนกำแพง / ขอบแผนที่: อยู่ช่องเดิม และเสียรางวัล 5 คะแนน"
            self.message_color = RED
        else:
            self.message = "เดินสำเร็จ: เสียรางวัล 1 คะแนนต่อก้าว"
            self.message_color = MUTED
        value = ""
        if isinstance(self.demo_agent, QLearningAgent):
            value = f"    Q={self.demo_agent.values(previous)[action]:.2f}"
        self.transition.configure(text=f"{previous}  →  {self.last_action}  →  {state}    รางวัล {reward:+d}{value}")
        self.refresh()

    def start_demo(self):
        try:
            seed = int(self.demo_seed.get())
        except ValueError:
            self.feedback.configure(text="Seed ต้องเป็นจำนวนเต็ม เช่น 0, 1 หรือ 42", fg=RED)
            return
        self.cancel_demo()
        self.env.reset()
        self.demo_agent = RandomAgent(episode_seed(seed, 1))
        self.demo_kind = "Random"
        self.demo_running = True
        self.last_event = "reset"
        self.message = f"Random Agent · seed {seed} · สุ่ม 1 รอบโดยไม่มีการเรียนรู้"
        self.message_color = MUTED
        self.transition.configure(text="")
        self.refresh()
        self.root.focus_set()
        self.schedule_demo()

    def load_q_model(self, path=None):
        if path is None:
            path = filedialog.askopenfilename(parent=self.root, title="เลือก Q-table ก่อน/ระหว่าง/หลังฝึก",
                                              initialdir=str(self.results_dir),
                                              filetypes=[("Q-table JSON", "*.json")])
        if not path:
            return False
        try:
            agent = QLearningAgent.load(path, self.env)
        except (OSError, ValueError) as exc:
            self.feedback.configure(text=f"โหลดโมเดลไม่ได้: {exc}", fg=RED)
            return False
        self.cancel_demo()
        self.q_agent, self.q_model_path = agent, Path(path)
        method = next((label for label, key in Q_METHODS.items() if key == agent.metadata.get("method")), "Q-learning")
        self.q_method.set(method)
        if "training_budget_actions" in agent.metadata:
            progress = f"{agent.updates:,} ก้าว"
        else:
            progress = f"ep {agent.metadata.get('episodes_completed', '?')}"
        self.q_model_label.configure(text=f"seed {agent.seed} · {progress}", fg=TEAL)
        self.env.reset()
        self.last_event = "reset"
        self.message = "โหลด Q-table แล้ว · กด ให้ Q เดิน เพื่อทดสอบโดยไม่เรียนรู้เพิ่ม"
        self.message_color = TEAL
        self.transition.configure(text="")
        self.refresh()
        self.root.focus_set()
        return True

    def change_q_method(self, event=None):
        self.q_agent = None
        self.q_model_path = None
        self.q_model_label.configure(text="ยังไม่โหลดโมเดล", fg=MUTED)
        self.reset()

    def start_q_demo(self):
        if self.q_agent is None:
            try:
                method = Q_METHODS[self.q_method.get()]
                try:
                    path = latest_comparison_model(self.results_dir, method)
                except FileNotFoundError:
                    if method != "plain":
                        raise
                    path = latest_demo_model(self.results_dir)
            except (OSError, ValueError) as exc:
                self.feedback.configure(text=f"ยังใช้ Q ไม่ได้: {exc}", fg=RED)
                return
            if not self.load_q_model(path):
                return
        self.cancel_demo()
        self.env.reset()
        self.demo_agent = self.q_agent
        self.demo_kind = self.q_method.get()
        self.demo_running = True
        self.last_event = "reset"
        self.message = f"{self.demo_kind} · เริ่มจาก S จริง · ε=0 · ไม่อัปเดตตารางระหว่างเดโม"
        self.message_color = TEAL
        self.transition.configure(text="")
        self.refresh()
        self.root.focus_set()
        self.schedule_demo()

    def schedule_demo(self):
        if self.demo_running and self.demo_timer is None and not self.env.done:
            delay = {"ช้า": 400, "ปกติ": 120, "เร็ว": 25}.get(self.demo_speed.get(), 120)
            self.demo_timer = self.root.after(delay, self.demo_tick)

    def demo_tick(self):
        self.demo_timer = None
        if not self.demo_running or self.demo_agent is None or self.env.done:
            return
        action = self.demo_agent.act(self.env.position)
        self.move(action, source="agent")
        if self.env.done:
            self.demo_running = False
            self.refresh()
        else:
            self.schedule_demo()

    def toggle_demo_pause(self):
        if self.demo_agent is None or self.env.done:
            return
        self.demo_running = not self.demo_running
        if self.demo_running:
            self.schedule_demo()
        elif self.demo_timer is not None:
            self.root.after_cancel(self.demo_timer)
            self.demo_timer = None
        self.refresh()

    def cancel_demo(self):
        self.demo_running = False
        if self.demo_timer is not None:
            self.root.after_cancel(self.demo_timer)
            self.demo_timer = None
        self.demo_agent = None

    def close(self):
        self.cancel_demo()
        self.root.destroy()

    def reset(self):
        self.cancel_demo()
        self.env.reset()
        self.last_event = "reset"
        self.message = "เริ่มรอบใหม่แล้ว · ลองหาทางไปยัง G"
        self.message_color = MUTED
        self.transition.configure(text="")
        self.refresh()
        self.root.focus_set()

    def refresh(self):
        self.values["steps"].configure(text=f"{self.env.steps} / {self.env.max_steps}")
        self.values["reward"].configure(text=f"{self.env.total_reward:+d}")
        self.values["walls"].configure(text=str(self.env.wall_hits))
        self.values["position"].configure(text=str(self.env.position))
        status, color = (("ถึงเป้าหมายแล้ว", TEAL) if self.env.terminated else
                         ("ครบจำนวนก้าว", GOLD) if self.env.truncated else ("กำลังเล่น", TEAL))
        if self.demo_agent is not None and not self.env.done:
            status = f"{self.demo_kind} · กำลังเดิน" if self.demo_running else f"{self.demo_kind} · พักอยู่"
        self.status.configure(text=status, fg=color)
        self.feedback.configure(text=self.message, fg=self.message_color)
        for button in self.move_buttons:
            button.configure(state="disabled" if self.env.done or self.demo_agent is not None else "normal")
        self.demo_pause_button.configure(
            text="พัก" if self.demo_running else "เล่นต่อ",
            state="normal" if self.demo_agent is not None and not self.env.done else "disabled",
        )
        self.render_maze()

    def render_maze(self):
        canvas = self.canvas
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        size = min((width - 4) / self.env.cols, (height - 4) / self.env.rows)
        if size < 1:
            return
        ox, oy = (width - size * self.env.cols) / 2, (height - size * self.env.rows) / 2
        for r, row in enumerate(self.env.maze):
            for c, cell in enumerate(row):
                if cell == "S":
                    cell = "."
                if (r, c) == self.env.episode_start:
                    cell = "S"
                x, y = ox + c * size, oy + r * size
                fill = "#29394f" if cell == "#" else "#142133"
                if (r, c) in self.env.visited:
                    fill = "#183c40"
                if cell == "S":
                    fill = "#224655"
                elif cell == "G":
                    fill = "#514125"
                canvas.create_rectangle(x + 2, y + 2, x + size - 2, y + size - 2, fill=fill, outline="")
                if cell in "SG":
                    canvas.create_text(x + size / 2, y + size / 2, text=cell,
                                       fill=GOLD if cell == "G" else "#94dce5",
                                       font=(FONT, max(11, int(size * .30)), "bold"))
                elif (r, c) in self.env.visited:
                    dot = max(2, size * .04)
                    canvas.create_oval(x + size / 2 - dot, y + size / 2 - dot,
                                       x + size / 2 + dot, y + size / 2 + dot, fill="#78b4a7", outline="")
        r, c = self.env.position
        x, y = ox + (c + .5) * size, oy + (r + .5) * size
        radius = size * .28
        color = RED if self.last_event == "wall" and not self.env.done else TEAL
        canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline="")
        canvas.create_text(x, y, text="✓" if self.env.terminated else "A", fill=BG,
                           font=(FONT, max(10, int(size * .23)), "bold"))


def main():
    root = tk.Tk()
    MazeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
