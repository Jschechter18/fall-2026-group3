"""Behavior-only presentation of adoption experiments using direct answer matching."""
from collections import Counter

from mas_sae.evaluation.scoring import answer_matches


def summarize_behavior(method, records):
    """Count mutually exclusive source matches without task-correctness aliases."""
    counts = Counter()
    for record in records:
        adopted = answer_matches(record['solver_attempt_2'], record['critic_advocated_answer'])
        retained = answer_matches(record['solver_attempt_2'], record['solver_attempt_1'])
        label = 'overlap' if adopted and retained else 'adopted' if adopted else 'retained_a1' if retained else 'other'
        counts[label] += 1
    return dict(method=method, trials=len(records), questions=len({r['question_id'] for r in records}),
                **{key: counts[key] for key in ('adopted', 'retained_a1', 'other', 'overlap')})


def render_behavior_chart(output, cohorts, matched=False):
    """Render one chart with cohort separators and explicit sample sizes."""
    import os
    import tempfile
    os.environ.setdefault('MPLCONFIGDIR', tempfile.mkdtemp(prefix='adoption-behavior-mpl-'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    palette = {'adopted': '#4575b4', 'retained_a1': '#1b9e77', 'other': '#bdbdbd', 'overlap': '#e6ab02'}
    titles = {'adopted': 'Adopted Critic answer', 'retained_a1': 'Retained A1',
              'other': 'Other', 'overlap': 'Matches both'}
    fig, axis = plt.subplots(figsize=(15, 9))
    positions, labels = [], []
    cursor = 0
    for cohort, rows in cohorts:
        if cursor:
            axis.axhline(cursor - .65, color='#dddddd', linewidth=1)
        n, q = rows[0]['trials'], rows[0]['questions']
        axis.text(0, cursor, f'{cohort}  |  {n} trials / {q} questions per row',
                  transform=axis.get_yaxis_transform(), fontsize=11, fontweight='bold', va='center')
        cursor += .9
        for row in rows:
            positions.append(cursor)
            labels.append(row['method'].replace(' candidates:', ':'))
            start = 0
            for key in palette:
                value = row[key]
                width = 100 * value / row['trials']
                axis.barh(cursor, width, left=start, color=palette[key], height=.65)
                if width >= 6:
                    axis.text(start + width / 2, cursor, str(value), ha='center', va='center', fontsize=11,
                              color='white' if key in ('adopted', 'retained_a1') else '#222222')
                start += width
            cursor += 1
        cursor += .65
    axis.set_yticks(positions, labels, fontsize=10)
    axis.set_ylim(cursor - 1, -.6)
    axis.set_xlim(0, 100)
    axis.set_xticks([0, 25, 50, 75, 100], ['0%', '25%', '50%', '75%', '100%'])
    axis.set_axisbelow(True)
    axis.grid(axis='x', alpha=.15)
    axis.spines[['top', 'right', 'left']].set_visible(False)
    axis.set_xlabel('Share of completed trials', fontsize=11)
    fig.suptitle('Paired protocol comparison on the same questions' if matched else
                 'Answer adoption depends on the interaction protocol', fontsize=18, x=.52, y=.965)
    used = [key for key in palette if any(row[key] for _, rows in cohorts for row in rows)]
    fig.legend(handles=[Patch(facecolor=palette[k], label=titles[k]) for k in used],
               loc='lower center', bbox_to_anchor=(.6, .095), ncol=len(used), frameon=False, fontsize=10)
    fig.subplots_adjust(left=.32, right=.97, top=.89, bottom=.22)
    comparison_note = ('Same question IDs across all rows; repeated controlled recommendations can produce more trials than questions.'
                       if matched else 'Comparisons are paired within each group. Natural and controlled samples differ; counts are not pooled. Screening is automatic.')
    fig.text(.04, .035,
             'Bar widths show percentages; labels show trial counts. Direct normalized matching to A1/recommendation; no gold aliases.\n'
             + comparison_note + '\n'
             'Original Natural revision sees full feedback; candidate comparison sees recommendation only.',
             fontsize=9, color='#444444', linespacing=1.6)
    fig.savefig(output / 'adoption_comparison.svg')
    fig.savefig(output / 'adoption_comparison.png', dpi=240)
    plt.close(fig)
