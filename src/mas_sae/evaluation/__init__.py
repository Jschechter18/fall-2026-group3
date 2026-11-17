"""Answer scoring, behavior labels and Claude labeling of Solver-Critic episodes.

``scoring`` and ``behavior`` compare answers as normalized text during collection (stage 1);
``claude_validator`` and ``claude_adaptive_client`` produce the Claude labels of stage 2.
``behavior_labels`` holds the earlier rule-based (v0.x) labels and the labels.csv writer that the
stage-2 export still uses; ``behavior_qc`` and ``feedback_coherence`` check those rule-based
labels. The other modules compare label sets or analyse mini experiments.
"""
