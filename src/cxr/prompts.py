"""Single source of truth for every prompt shown to a vision-language model.

The same instruction is used for LoRA training targets, zero-shot scoring and
fine-tuned scoring, so train/test prompt drift cannot occur.
"""

from .tasks import Task

SYSTEM_ROLE = "You are an expert radiologist."


def instruction(task: Task) -> str:
    a, b = task.class_names
    return (
        f"{SYSTEM_ROLE} Analyze this chest X-ray and classify it as '{a}' or '{b}'. "
        "Output only the diagnosis."
    )


def icl_query_instruction(task: Task) -> str:
    a, b = task.class_names
    return (
        "\n--- TASK ---\nAnalyze the FINAL X-ray below based on the visual patterns shown in "
        f"the examples above. Classify it as '{a}' or '{b}'. Output only the diagnosis.\n"
        "Final Image:"
    )


def icl_header(task: Task) -> str:
    a, b = task.class_names
    return (
        f"{SYSTEM_ROLE} Below are labelled example chest X-rays, each either '{a}' or '{b}'."
    )


def zero_shot_messages(task: Task):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction(task)},
                {"type": "image"},
            ],
        }
    ]


def icl_messages(task: Task, exemplar_labels):
    """Build a multi-image user turn: header, labelled exemplars, then the query image.

    Exemplar images and the query image must be passed to the processor in the
    same order as the ``{"type": "image"}`` placeholders produced here.
    """
    content = [{"type": "text", "text": icl_header(task)}]
    for i, label in enumerate(exemplar_labels, start=1):
        content.append({"type": "text", "text": f"Example {i}:"})
        content.append({"type": "image"})
        content.append({"type": "text", "text": f"Diagnosis: {label}"})
    content.append({"type": "text", "text": icl_query_instruction(task)})
    content.append({"type": "image"})
    return [{"role": "user", "content": content}]


def training_conversation(task: Task, image, label_name: str):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction(task)},
                    {"type": "image", "image": image},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": label_name}]},
        ]
    }
