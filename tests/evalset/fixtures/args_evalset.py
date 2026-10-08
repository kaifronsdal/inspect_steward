"""A script parameterized by its own options, which Steward renders from `-A KEY=VALUE`."""

import argparse

from inspect_ai import Task, eval_set, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate

parser = argparse.ArgumentParser()
parser.add_argument("--difficulty", default="easy")
args = parser.parse_args()


@task
def sweep(difficulty: str = "easy") -> Task:
    return Task(
        dataset=[Sample(input=f"question ({difficulty})", target="answer")],
        solver=[generate()],
        scorer=exact(),
    )


eval_set(
    tasks=[sweep(difficulty=args.difficulty)],
    model="mockllm/model",
    log_dir="logs",
)
