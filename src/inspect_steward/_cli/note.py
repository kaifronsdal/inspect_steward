"""`steward note` — writing something down where the next reader will find it.

The journal is the only record one session leaves that another session reads, and two acts leave no entry of their own: stopping to ask, where `notify` posts to a channel and writes nothing, and reaching into a worker through `inspect ctl`, where the change is recorded in the eval log and no `collect` looks. This is the verb for both — one append, shown under *what happened*, so a 6am reader and the next session see the state and the hypothesis at the moment they were formed.

**It takes no claim and changes nothing.** A note is not a decision: it opens nothing, closes nothing, and pauses nothing. The verbs that do those things carry their own `--reason`.
"""

import click

from .._workspace import NOTED, append_event
from .turn import find_workspace


@click.command("note")
@click.argument("message")
@click.option(
    "--by",
    type=click.Choice(["operator", "agent"]),
    default="agent",
    show_default=True,
    help="Whose note. Defaults to the agent, whose observations are what this verb exists to keep.",
)
@click.option(
    "--retried",
    nargs=3,
    type=str,
    default=None,
    metavar="TASK_ID SAMPLE_ID EPOCH",
    help="The sample this note records a stuck-ladder cancel-and-requeue of. What the retry-once guard reads: a sample noted here gets no second standing retry.",
)
def note_command(message: str, by: str, retried: tuple[str, str, str] | None) -> None:
    """Write a note into the journal, for whoever reads this run next.

    MESSAGE is free text: the state of something and what you think it means. It appears under *what happened* in `status` and `collect`, in order with everything else that was done to the run.

    `--retried` marks the note as the record of a standing stuck retry (`stuck_action: retry`): the sample it names, by task id, sample id and epoch, is counted as having had its one retry, and its next wedge is an operator's.
    """
    workspace = find_workspace()
    text = message.strip()
    if not text:
        raise click.ClickException("a note needs some text")
    if retried is None:
        append_event(workspace.journal, NOTED, by=by, text=text)
    else:
        task_id, sample_id, epoch = retried
        if not epoch.isdecimal():
            raise click.ClickException(
                f"--retried takes TASK_ID SAMPLE_ID EPOCH, and the epoch is a "
                f"number — not {epoch!r}"
            )
        append_event(
            workspace.journal,
            NOTED,
            by=by,
            text=text,
            retried={"task": task_id, "sample": sample_id, "epoch": int(epoch)},
        )
    click.echo("noted")
