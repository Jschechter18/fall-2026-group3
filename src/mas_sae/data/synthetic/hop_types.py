"""Supported MuSiQue dependency shapes and their derived hop counts."""

# Each tuple lists the one-based answers referenced by that step.
HOP_SHAPES: dict[str, list[tuple[int, ...]]] = {
    "2hop": [(), (1,)],
    "3hop1": [(), (1,), (2,)],
    "3hop2": [(), (), (1, 2)],
    "4hop1": [(), (1,), (2,), (3,)],
    "4hop2": [(), (), (1, 2), (3,)],
    "4hop3": [(), (1,), (), (2, 3)],
}

HOP_TYPES = {
    hop_type: len(shape) for hop_type, shape in HOP_SHAPES.items()
}
