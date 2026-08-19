"""Interactive terminal browser for retrieved QA examples."""

import json
top_k = 15
class Task:
    """Load questions and expose navigation state for the terminal UI."""

    class Question:
        """Store one question, its answer, and displayed retrieval contexts."""

        def __init__(self, question_id: str, question: str, answer: str):
            self.question_id = question_id
            self.question = question
            self.answer = answer
            self.contexts = []
        def add_context(self, context: str):
            """Append a context to this question's display list."""

            self.contexts.append(context)
    def __init__(self, path: str):
        self.questions: list[Task.Question] = []
        with open(path, "r") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                question_id = example['id']
                question_id = question_id
                question = Task.Question(question_id, example['question'], example['answers'])             
                for ctx in example['ctxs'][:top_k]:
                    text = ctx['text']
                    question.add_context(text)
                self.questions.append(question)
        self.current_question = 0
    def show_current(self):
        """Build the prompt-toolkit container for the current question."""

        try:
            from prompt_toolkit.application import get_app
            from prompt_toolkit.layout.containers import HSplit, VSplit, Window
            from prompt_toolkit.layout.controls import FormattedTextControl
        except ImportError as error:
            raise RuntimeError(
                "The interactive task browser requires prompt-toolkit."
            ) from error

        new_container = VSplit([
            HSplit([
            Window(content=FormattedTextControl(text=lambda: f"[Question] ({self.current_question + 1}/{len(self.questions)}):\n{self.questions[self.current_question].question}"), height=3, style="bg:#555555", wrap_lines=True),
            Window(content=FormattedTextControl(text=lambda: f"[Answer]:\n{self.questions[self.current_question].answer}"), height=3, style="bg:#555555", wrap_lines=True),
            ], width=lambda: int(get_app().output.get_size().columns * 1 / 3)),
            Window(content=FormattedTextControl(text=lambda: "\n".join([f"[{i + 1}]. {ctx}" for i, ctx in enumerate(self.questions[self.current_question].contexts)])), style="bg:#333333", wrap_lines=True),
        ])
        return new_container
    def shift_right(self):
        """Advance one question without passing the end of the dataset."""

        if self.current_question < len(self.questions) - 1:
            self.current_question += 1
    def shift_left(self):
        """Move back one question without passing the start of the dataset."""

        if self.current_question > 0:
            self.current_question -= 1
    def jump(self, id):
        """Jump to a zero-based question index when it is in range."""

        if id < len(self.questions):
            self.current_question = id
def main() -> int:
    """Launch the optional prompt-toolkit task browser."""

    try:
        from prompt_toolkit import Application, prompt
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout
    except ImportError as error:
        raise RuntimeError(
            "The interactive task browser requires prompt-toolkit."
        ) from error

    path = "data/retr_result/popqa_longtail_w_gs.jsonl"
    task = Task(path)
    bindings = KeyBindings()
    buffer1 = Buffer()
    root_container = task.show_current()
    @bindings.add("right")
    def _(event):
        task.shift_right()
        event.app.layout.container = task.show_current()
    @bindings.add("left")
    def _(event):
        task.shift_left()
        event.app.layout.container = task.show_current()
    @bindings.add(":")
    def _(event):
        id = prompt("Jump to question ID: ")
        try:
            id = int(id)
            task.jump(id)
            event.app.layout.container = task.show_current()
        except ValueError:
            print("Invalid ID. Please enter a number.")
    @bindings.add("q")
    def _(event):
        event.app.exit()
    layout = Layout(root_container)
    app = Application(layout=layout, key_bindings=bindings, full_screen=True)
    app.run()
    task.show_current()
    return 0


if __name__ == "__main__":  # pragma: no cover - optional interactive UI.
    raise SystemExit(main())
