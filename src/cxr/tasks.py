from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    name: str
    dataset_id: str
    class_names: tuple
    description: str
    revision: str

    @property
    def positive_class(self) -> str:
        return self.class_names[1]

    @property
    def negative_class(self) -> str:
        return self.class_names[0]


TASKS = {
    "task1": Task(
        name="task1",
        dataset_id="Henryayay/POLCOVID_task1_70_10_20",
        class_names=("Normal", "Pneumonia"),
        description="Normal vs. Pneumonia",
        revision="435f00fc15744fc2cedfea9efcdb041f1d93113b",
    ),
    "task2": Task(
        name="task2",
        dataset_id="Henryayay/POLCOVID_task2_70_10_20",
        class_names=("Non-COVID Pneumonia", "COVID-19"),
        description="Non-COVID pneumonia vs. COVID-19",
        revision="6178041d37e25b33aaaf3d1356072dcbcc45af45",
    ),
}


def get_task(name: str) -> Task:
    try:
        return TASKS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown task '{name}'. Choose from {sorted(TASKS)}") from exc
