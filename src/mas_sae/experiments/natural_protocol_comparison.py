"""Eight protocol conditions on the exact fixed-feedback plot's questions."""
from collections import Counter

from mas_sae.agents.critic import Critic, CriticCondition
from mas_sae.evaluation.scoring import answer_matches, score_revision
from mas_sae.experiments.artifacts import sha256_text
from mas_sae.experiments.records import append_jsonl, read_jsonl

METHODS = (
    "Original revision baseline", "Evidence verification",
    "Labeled candidates: A1 first", "Labeled candidates: critic first",
    "Anonymous candidates: A1 first", "Anonymous candidates: critic first",
    "Labeled candidates: randomized order", "Anonymous candidates: randomized order",
)


def select_cohort(packet, controlled, assignments):
    """Keep every question represented in the fixed plot, including missing feedback."""
    ids = {r['question_id'] for r in controlled if r['status'] == 'completed'}
    rows = [r for r in packet['rows'] if r['question_id'] in ids]
    if len(rows) != len(ids) or not ids or not ids <= assignments.keys():
        raise ValueError('Frozen sample/order assignments do not cover the plotted cohort')
    for row in rows:
        if assignments[row['question_id']] not in ('a1_first', 'critic_first'):
            raise ValueError('Invalid original candidate order')
        feedback = Critic.parse_natural(row['critic_raw_output'])
        if feedback.to_solver_text() != row['critic_feedback'] or feedback.advocated_answer != row['critic_advocated_answer']:
            raise ValueError('Saved natural feedback differs from parsed source')
    return rows


def needs_feedback(row):
    feedback = Critic.parse_natural(row['critic_raw_output'])
    return feedback.noncommittal or not feedback.advocated_answer


def prepare_feedback(rows, examples, critic, path):
    """Checkpoint one usable recommendation per question; never replace usable originals."""
    saved = read_jsonl(path) if path.exists() else []
    index = {r['question_id']: r for r in saved}
    if len(index) != len(saved) or not index.keys() <= {r['question_id'] for r in rows}:
        raise ValueError('Invalid feedback checkpoint')
    for row in rows:
        qid = row['question_id']
        if qid in index:
            continue
        feedback = Critic.parse_natural(row['critic_raw_output'])
        source = 'saved_natural'
        if needs_feedback(row):
            ex = examples[qid]
            feedback = critic.critique(question=row['question'], paragraphs=ex['paragraphs'],
                                       solver_answer=row['solver_attempt_1'], condition=CriticCondition.NATURAL)
            source = 'regenerated_natural'
            retry_instructions = (
                '\nKeep the JSON response compact. Limit explanation to one short sentence. '
                'Return one complete JSON object and stop immediately after its closing brace.',
                '\nReturn only a complete JSON object with verdict, advocated_answer, and explanation. '
                'Use an empty string for explanation. Do not include reasoning or any additional text.',
            )
            for attempt in range(len(retry_instructions) + 1):
                if not feedback.noncommittal and feedback.advocated_answer:
                    break
                append_jsonl(path.with_name('failed_feedback.jsonl'), dict(
                    question_id=qid, source=source, raw_output=feedback.raw_output,
                    attempt=attempt, generation=dict(getattr(critic, 'last_generation', None) or {})))
                if attempt == len(retry_instructions):
                    raise ValueError(f'{qid}: natural critic returned no usable recommendation after '
                                     '3 attempts; see failed_feedback.jsonl')
                prompt = critic.build_critique_prompt(
                    question=row['question'], paragraphs=ex['paragraphs'],
                    solver_answer=row['solver_attempt_1'], condition=CriticCondition.NATURAL)
                prompt += retry_instructions[attempt]
                feedback = Critic.parse_natural(critic._generate(prompt))
            generation = dict(getattr(critic, 'last_generation', None) or {})
        record = dict(question_id=qid, source=source, raw_output=feedback.raw_output,
                      critic_feedback=feedback.to_solver_text(), critic_advocated_answer=feedback.advocated_answer)
        if source == 'regenerated_natural':
            record.update(generation=generation, retry_attempt=attempt,
                          retry_instruction=retry_instructions[attempt - 1] if attempt else None)
        if feedback.noncommittal or not feedback.advocated_answer:
            # Preserve the failed output for inspection, without fabricating a target.
            append_jsonl(path.with_name('failed_feedback.jsonl'), record)
            raise ValueError(f'{qid}: regenerated critic still has no usable recommendation')
        append_jsonl(path, record)
        index[qid] = record
    return index


def run_conditions(rows, examples, feedback, assignments, agents, path):
    """Run the six unique prompts; randomized rows select the original assigned order."""
    records = read_jsonl(path) if path.exists() else []
    index = {(r['question_id'], r['method']): r for r in records}
    expected = {(r['question_id'], m) for r in rows for m in METHODS[:6]}
    if len(index) != len(records) or not index.keys() <= expected:
        raise ValueError('Invalid response checkpoint')
    for row in rows:
        qid = row['question_id']
        ex, recommendation = examples[qid], feedback[qid]
        a1, target = row['solver_attempt_1'], recommendation['critic_advocated_answer']
        correct = answer_matches(target, ex['answer'], ex.get('answer_aliases', []))
        for method in METHODS[:6]:
            if (qid, method) in index:
                continue
            if method in METHODS[:2]:
                agent = agents['original' if method == METHODS[0] else 'verification']
                prompt = agent.build_revise_prompt(row['question'], ex['paragraphs'], a1, recommendation['critic_feedback'])
            else:
                agent = agents['candidate']
                condition = 'anonymous' if method.startswith('Anonymous') else 'labeled'
                order = 'a1_first' if method.endswith('A1 first') else 'critic_first'
                prompt = agent.build_candidate_prompt(row['question'], ex['paragraphs'], a1, target, condition, order)
            a2 = agent._generate(prompt)
            record = dict(question_id=qid, method=method, question=row['question'],
                          solver_attempt_1=a1, solver_attempt_2=a2,
                          critic_feedback=recommendation['critic_feedback'], critic_advocated_answer=target,
                          recommendation_source=recommendation['source'], recommendation_correct=correct,
                          assigned_order=assignments[qid], prompt_sha256=sha256_text(prompt),
                          evaluation_prompt=prompt, generation=dict(agent.last_generation or {}),
                          **score_revision(a2, a1, target, ex['answer'], ex.get('answer_aliases', []),
                                           accept_gold_aliases=correct))
            append_jsonl(path, record)
            records.append(record)
            index[qid, method] = record
    methods = [(m, [index[r['question_id'], m] for r in rows]) for m in METHODS[:6]]
    for label, prefix in ((METHODS[6], 'Labeled'), (METHODS[7], 'Anonymous')):
        selected = []
        for row in rows:
            qid = row['question_id']
            suffix = 'A1 first' if assignments[qid] == 'a1_first' else 'critic first'
            selected.append({**index[qid, f'{prefix} candidates: {suffix}'], 'method': label})
        methods.append((label, selected))
    return methods


def summarize_method(name, rows):
    """Disjoint stacks; adoption rates include matches-both as in saved scoring."""
    counts = Counter('overlap' if r['adopted_target'] and r['retained_a1'] else
                     'adopted' if r['adopted_target'] else 'retained_a1' if r['retained_a1'] else 'other'
                     for r in rows)
    result = dict(method=name, completed_trials=len(rows), distinct_questions=len({r['question_id'] for r in rows}),
                  saved_feedback=sum(r['recommendation_source'] == 'saved_natural' for r in rows),
                  regenerated_feedback=sum(r['recommendation_source'] == 'regenerated_natural' for r in rows),
                  **{k: counts[k] for k in ('adopted', 'retained_a1', 'other', 'overlap')})
    for correct, prefix in ((False, 'incorrect'), (True, 'correct')):
        subset = [r for r in rows if r['recommendation_correct'] == correct]
        result['correct_recommendation_trials' if correct else 'wrong_recommendation_trials'] = len(subset)
        result[f'{prefix}_accept_pct'] = 100 * sum(r['adopted_target'] for r in subset) / len(subset) if subset else float('nan')
    return result
