"""LLM-edited SAE experiment (exploratory): an LLM edits the decoder rows of the selected features by hand.

See LLM_EDITED_SAE_PROMPT.md. `edit_rules` holds the edit rules and the edits.py record; `sandbox` runs
the LLM's snippets in a network-less Docker container (its in-container code is in `container/`).
"""
