"""Compact examples for synthetic generation prompts.

Each step tuple contains:
(question, answer, supporting paragraph title, supporting fact).
Examples illustrate structure; generated paragraphs should be longer.
"""

EXAMPLE_SPECS = {
    "2hop": (
        "When was the town containing Luma Observatory founded?",
        [
            (
                "In which town is Luma Observatory located?",
                "Selin",
                "Luma Observatory",
                "Luma Observatory is located in Selin.",
            ),
            (
                "When was #1 founded?",
                "1842",
                "Selin",
                "Selin was founded in 1842.",
            ),
        ],
    ),
    "3hop1": (
        "Who designed the hall in the town containing Luma Observatory?",
        [
            (
                "In which town is Luma Observatory located?",
                "Selin",
                "Luma Observatory",
                "Luma Observatory is located in Selin.",
            ),
            (
                "What is the main public hall in #1?",
                "Vela Hall",
                "Selin",
                "Selin's main public hall is Vela Hall.",
            ),
            (
                "Who designed #2?",
                "Mira Dalen",
                "Vela Hall",
                "Vela Hall was designed by Mira Dalen.",
            ),
        ],
    ),
    "3hop2": (
        "Who designed the joint exhibition of Luma Observatory "
        "and the museum founded by Ena Voss?",
        [
            (
                "What is Luma Observatory's exhibition called?",
                "Night Atlas",
                "Luma Observatory",
                "Luma Observatory's exhibition is called Night Atlas.",
            ),
            (
                "Which museum did Ena Voss found?",
                "Orin Museum",
                "Ena Voss",
                "Ena Voss founded Orin Museum.",
            ),
            (
                "Who designed the joint exhibition of #1 and #2?",
                "Taren Vale",
                "Exhibition designs",
                "Taren Vale designed the joint exhibition of "
                "Night Atlas and Orin Museum.",
            ),
        ],
    ),
    "4hop1": (
        "Where was the designer of the hall in the town "
        "containing Luma Observatory born?",
        [
            (
                "In which town is Luma Observatory located?",
                "Selin",
                "Luma Observatory",
                "Luma Observatory is located in Selin.",
            ),
            (
                "What is the main public hall in #1?",
                "Vela Hall",
                "Selin",
                "Selin's main public hall is Vela Hall.",
            ),
            (
                "Who designed #2?",
                "Mira Dalen",
                "Vela Hall",
                "Vela Hall was designed by Mira Dalen.",
            ),
            (
                "Where was #3 born?",
                "Ivara",
                "Mira Dalen",
                "Mira Dalen was born in Ivara.",
            ),
        ],
    ),
    "4hop2": (
        "Where was the designer of the joint exhibition of Luma "
        "Observatory and the museum founded by Ena Voss born?",
        [
            (
                "What is Luma Observatory's exhibition called?",
                "Night Atlas",
                "Luma Observatory",
                "Luma Observatory's exhibition is called Night Atlas.",
            ),
            (
                "Which museum did Ena Voss found?",
                "Orin Museum",
                "Ena Voss",
                "Ena Voss founded Orin Museum.",
            ),
            (
                "Who designed the joint exhibition of #1 and #2?",
                "Taren Vale",
                "Exhibition designs",
                "Taren Vale designed the joint exhibition of "
                "Night Atlas and Orin Museum.",
            ),
            (
                "Where was #3 born?",
                "Ivara",
                "Taren Vale",
                "Taren Vale was born in Ivara.",
            ),
        ],
    ),
    "4hop3": (
        "Who designed the joint exhibition of the museum in "
        "Luma Observatory's town and the museum founded by Ena Voss?",
        [
            (
                "In which town is Luma Observatory located?",
                "Selin",
                "Luma Observatory",
                "Luma Observatory is located in Selin.",
            ),
            (
                "What museum is in #1?",
                "Vela Museum",
                "Selin",
                "The museum in Selin is Vela Museum.",
            ),
            (
                "Which museum did Ena Voss found?",
                "Orin Museum",
                "Ena Voss",
                "Ena Voss founded Orin Museum.",
            ),
            (
                "Who designed the joint exhibition of #2 and #3?",
                "Taren Vale",
                "Exhibition designs",
                "Taren Vale designed the joint exhibition of "
                "Vela Museum and Orin Museum.",
            ),
        ],
    ),
}


def build_example(hop_type: str) -> dict:
    """Build the raw model-response format for one subtype."""
    question, steps = EXAMPLE_SPECS[hop_type]
    return {
        "question": question,
        "answer": steps[-1][1],
        "answer_aliases": [],
        "supporting_paragraphs": [
            {"title": title, "paragraph_text": fact}
            for _, _, title, fact in steps
        ],
        "question_decomposition": [
            {
                "question": step_question,
                "answer": answer,
                "support": position,
            }
            for position, (step_question, answer, _, _) in enumerate(steps)
        ],
        "distractors": [
            {
                "title": "Talven Gallery",
                "paragraph_text": (
                    "Talven Gallery presents an unrelated sculpture collection."
                ),
            },
        ],
    }
