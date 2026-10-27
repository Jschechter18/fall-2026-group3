"""Snapshot of the preceding scratch proposal, for reproducibility only.

These heuristics have known errors. Use policy.classify_review for the guarded
review draft. No notebook execution, model, held-out text review, or pinned
question IDs is used at runtime. Source hashes are in snapshot_sources.json.
"""
import re
from mas_sae.evaluation import behavior
from mas_sae.evaluation.behavior_labels import (
    LETTER_SOURCES, PREMISE, POLICIES, V020_CANDIDATE as V020,
    answer_kind, classify_candidate as classify, override_feedback,
    V012_PREMISE_PATTERNS, V012_A2_NONANSWER_PATTERNS,
)
from mas_sae.evaluation.scoring import normalize_answer as n
normalize_answer = n
POLICY = V020
MAX_WORDS = V020.candidate_rules.max_answer_words


FUNCTION_WORDS = frozenset('a an the of in on at to for and or by from with as is was were are be been its it this that which who s'.split())
DIAGNOSTIC_REFUSAL = re.compile('\\b(?:do|does|did)\\s+not\\s+(?:mention|specify|provide|state|contain|include|identify|say|indicate|give|list|name|support)\\b|\\bno\\s+(?:such|mention|information|record|evidence|valid)\\b|\\bnone\\s+of\\s+the\\b|\\bnot\\s+(?:enough|sufficient)\\s+information\\b|\\b(?:is|are)\\s+not\\s+(?:mentioned|specified|provided|stated|given|identified)\\b|\\binsufficient\\b', re.I)
NEGATION = re.compile("\\b(?:no\\s+one|nobody|never|failed\\s+to|not)\\b|n't\\b", re.I)
LETTER_PREFIX = re.compile('^\\W*candidate\\s+([ab])\\b', re.I)

def n_words(text):
    return len(text.split()) if isinstance(text, str) else 0

def tokens(text):
    if not isinstance(text, str):
        return set()
    text = re.sub('(?<=\\d)[,\u202f](?=\\d{3}\\b)', '', text.casefold())
    text = re.sub('\\b(\\d+)(?:st|nd|rd|th)\\b', '\\1', text)
    return set(re.findall('[^\\W_]+', text))

def content(text):
    return tokens(text) - FUNCTION_WORDS

def same(left, right):
    return behavior.textual_relation(left, right) == 'exact'

def refusal_source(text, *, objection=False):
    if not isinstance(text, str) or answer_kind(text) in ('nonanswer', 'missing'):
        return 'label rules: non-answer wording'
    if override_feedback('answer', text, POLICY)[0] != 'answer':
        return 'label rules: policy refusal / premise patterns'
    if PREMISE.search(text):
        return 'label rules: premise pattern'
    if DIAGNOSTIC_REFUSAL.search(text):
        return 'diagnostic: refusal phrase'
    if objection and n_words(text) > MAX_WORDS and NEGATION.search(re.split('(?<=[.;])\\s', text, maxsplit=1)[0]):
        return 'diagnostic: negated first sentence (objection)'
    return None

def a2_side(row):
    a2, a1, critic = (row['solver_attempt_2'], row['solver_attempt_1'], row['critic_advocated_answer'])
    letter = LETTER_PREFIX.match(a2) if isinstance(a2, str) else None
    if letter:
        side = LETTER_SOURCES.get(row['candidate_order'], {}).get(letter.group(1).upper(), 'neither')
        return (side, 'letter' if letter.group(0).strip() == a2.strip() else 'letter with extra text')
    to_a1, to_critic = (same(a2, a1), same(a2, critic))
    if to_a1 or to_critic:
        return ('both' if to_a1 and to_critic else 'a1' if to_a1 else 'critic', 'exact')
    words = content(a2)
    in_a1, in_critic = (bool(words) and words <= content(a1), bool(words) and words <= content(critic))
    if in_a1 != in_critic:
        return ('a1' if in_a1 else 'critic', 'shortened copy')
    return ('neither', 'none')

USABLE, UNUSABLE, AGREE, REFUSAL, AMBIGUOUS, RECOVER = ('Usable now', 'Unusable', 'Agreement', 'Refusal involved', 'Ambiguous', 'Recoverable by re-evaluation')
BUCKETS = [USABLE, RECOVER, REFUSAL, AMBIGUOUS, AGREE, UNUSABLE]

def triage(row, label):
    if label['eligible_strict']:
        return (USABLE, 'Two different short answers; A2 picks one')
    a1, critic, a2 = (row[key] for key in ('solver_attempt_1', 'critic_advocated_answer', 'solver_attempt_2'))
    a1_refused, a2_refused = (refusal_source(a1) is not None, refusal_source(a2) is not None)
    critic_source = refusal_source(critic, objection=True)
    critic_refused = critic_source is not None
    critic_what = 'objected to the question' if critic_source and 'objection' in critic_source else 'refused'
    side, via = a2_side(row)
    if a1_refused and critic_refused:
        return (UNUSABLE, 'A1 and Critic both refused')
    if label['a1_vs_critic'] == 'exact' or same(a1, critic):
        return (AGREE, 'Critic agrees with A1')
    if critic_refused:
        if side == 'a1':
            return (REFUSAL, f"Critic {critic_what}; A2 kept A1's answer")
        if side == 'critic' or a2_refused:
            return (REFUSAL, f"Critic {critic_what}; A2 switched to the Critic's reply or a refusal")
        return (UNUSABLE, f'Critic {critic_what}; A2 gave a new answer')
    if a1_refused:
        if side == 'critic':
            return (REFUSAL, "A1 refused; A2 took the Critic's answer")
        if side == 'a1' or a2_refused:
            return (REFUSAL, 'A1 refused; A2 stayed refused')
        return (UNUSABLE, 'A1 refused; A2 gave a new answer')
    if side == 'both':
        return (AMBIGUOUS, 'A2 matches both candidates')
    if side == 'neither':
        if a2_refused:
            return (UNUSABLE, 'A2 refused after two different answers')
        return (UNUSABLE, 'A2 gave a third answer')
    long_a1, long_critic = (n_words(a1) > MAX_WORDS, n_words(critic) > MAX_WORDS)
    if long_a1 and long_critic:
        return (AMBIGUOUS, 'A1 and Critic are both sentences (needs semantic comparison)')
    if long_a1 or long_critic:
        short, sentence = (critic, a1) if long_a1 else (a1, critic)
        if content(short) and content(short) <= tokens(sentence):
            return (AMBIGUOUS, 'Short answer appears inside the other sentence')
        return (RECOVER, ('A1' if long_a1 else 'Critic') + ' is a sentence; the short answer is not inside it')
    relation = label['a1_vs_critic']
    if relation == 'ambiguous_alias' or (relation == 'containment' and content(a1) & content(critic)):
        return (AMBIGUOUS, 'Short answers overlap (e.g. Henry vs Henry County)')
    if relation == 'containment':
        return (RECOVER, 'Character-level containment with no shared word (e.g. 11 vs 1)')
    if via in ('letter with extra text', 'shortened copy'):
        return (RECOVER, f'A2 picked one side by {via}')
    if label['a1_kind'] == 'uncertain':
        return (RECOVER, 'A1 flagged as hedged by a wording rule')
    return (RECOVER, 'Other rule: ' + ', '.join(label['behavior_exclusion_reasons']))

DIAG = DIAGNOSTIC_REFUSAL
NEG = NEGATION


alnum = lambda s: re.sub('[\\W_]+', '', s.casefold())
first_sentence = lambda s: re.split('(?<=[.;])\\s', s, maxsplit=1)[0]

def situation(lab, a1, c):
    k1, ft = (lab['a1_kind'], lab['feedback_type'])
    same = bool(a1) and a1 == c
    if k1 == 'nonanswer' and (same or ft in ('refusal', 'premise_rejection')):
        return 'both_refused'
    if same:
        return 'agreement'
    if ft == 'unresolved':
        return 'critic_unresolved'
    if k1 == 'answer':
        return {'answer': 'competing_answer', 'refusal': 'critic_refusal', 'premise_rejection': 'critic_objection'}[ft]
    if k1 == 'nonanswer':
        return 'a1_refusal'
    return 'a1_long_or_hedged'

def outcome(r, lab):
    """(outcome, via). Letter replies, bare or with text after them, are mapped through candidate_order."""
    a1r, cr, a2r = (r.get(k) or '' for k in ('solver_attempt_1', 'critic_advocated_answer', 'solver_attempt_2'))
    a1, c, a2 = (n(a1r), n(cr), n(a2r))
    m = LETTER_PREFIX.match(a2r)
    if m:
        side = LETTER_SOURCES.get(r['candidate_order'], {}).get(m.group(1).upper())
        via = 'letter' if m.group(0).strip() == a2r.strip() else 'letter_with_text'
        return ('retained_a1' if side == 'a1' else 'followed_c1', via)
    if a2 and a2 == a1:
        return ('retained_a1', 'exact')
    if a2 and a2 == c:
        return ('followed_c1', 'exact')
    if lab['a2_kind'] == 'nonanswer':
        return ('refused_own_words', '-')
    if alnum(a2r) and alnum(a2r) in alnum(a1r):
        return ('part_of_a1', '-')
    if alnum(a2r) and alnum(a2r) in alnum(cr):
        return ('part_of_c1', '-')
    return ('third_answer', '-')

STOP = {'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'and', 'or', 'by', 'from', 'with', 'as', 'is', 'was', 'were', 'are', 'be', 'its', 'it', 'this', 'that', 'which', 'who', 's'}

def refuses(t):
    fs = first_sentence(t)
    return bool(DIAG.search(t) or answer_kind(t) == 'nonanswer' or NEG.search(fs) or re.match('^\\s*(?:no|none|there (?:is|are) no)\\b', fs, re.I))
PREM = list(V012_PREMISE_PATTERNS.values())
A2NA = list(V012_A2_NONANSWER_PATTERNS.values())

def contains(big, small):
    return bool(small) and f' {small} ' in f' {big} '

def objects_first(t):
    f1 = first_sentence(t)
    return bool(any((p.search(f1) for p in PREM)) or NEG.search(f1) or DIAG.search(f1) or (answer_kind(f1) == 'nonanswer'))

def short_refusal(t):
    return len(t.split()) <= 14 and bool(DIAG.search(t) or any((p.search(t) for p in A2NA)))

def a1_refuses_c1_not(a1, c1):
    return bool((DIAG.search(a1) or answer_kind(a1) == 'nonanswer') and (not (DIAG.search(c1) or answer_kind(c1) == 'nonanswer')))

def prefix_side(a2, a1, c1):
    A2, A1, C1 = (n(a2), n(a1), n(c1))
    if len(A2.split()) < 3:
        return None
    pa, pc = (A1.startswith(A2) and A2 != A1, C1.startswith(A2) and A2 != C1)
    return 'retained_a1' if pa and (not pc) else 'followed_c1' if pc and (not pa) else None
DECISION_SITS = ('critic_refusal', 'critic_objection', 'a1_refusal', 'competing_answer', 'critic_unresolved', 'a1_long_or_hedged')

def source(r, lab):
    """(source key, outcome used for adopt/retain). First match wins."""
    a1, c1, q, a2 = (r['solver_attempt_1'] or '', r['critic_advocated_answer'] or '', r['question'] or '', r['solver_attempt_2'] or '')
    sit = situation(lab, n(a1), n(c1))
    oc, via = outcome(r, lab)
    choice = oc in ('followed_c1', 'retained_a1')
    b, m = triage(r, lab)
    if lab['eligible_strict']:
        return ('main', oc)
    if sit == 'agreement':
        return ('x_agreement', oc)
    if sit == 'both_refused' or m == 'A1 and Critic both refused':
        return ('x_both_refused', oc)
    if choice and sit in DECISION_SITS and (contains(n(c1), n(a1)) or contains(n(a1), n(c1))):
        return ('x_same_in_disguise', oc)
    if choice and via != 'exact' and (sit in DECISION_SITS):
        return ('letters', oc)
    if choice:
        if sit == 'critic_refusal':
            return ('refusal', oc)
        if sit == 'critic_objection':
            return ('objection', oc)
        if sit == 'a1_refusal':
            return ('a1_refusal', oc)
        if b == 'Recoverable by re-evaluation' and (m.endswith('not inside it') or m.startswith('Character-level')):
            if refuses(c1) or refuses(a1):
                return ('x_recover_refusal', oc)
            if (content(a1) & content(c1)) - content(q) - STOP:
                return ('x_recover_same', oc)
            return ('recover_clean', oc)
        if sit == 'critic_unresolved' and lab['a1_kind'] == 'answer' and DIAG.search(c1) and (DIAG.search(first_sentence(c1)) or answer_kind(first_sentence(c1)) == 'nonanswer'):
            return ('long_critic_refusal', oc)
        if sit == 'a1_long_or_hedged' and lab['feedback_type'] == 'refusal' and (not DIAG.search(a1)):
            return ('long_a1_vs_refusal', oc)
        if sit == 'critic_unresolved' and lab['a1_kind'] == 'answer' and objects_first(c1):
            return ('long_critic_objection', oc)
        if sit == 'competing_answer' and short_refusal(c1):
            return ('missed_short_refusal', oc)
        if sit in ('critic_unresolved', 'a1_long_or_hedged', 'competing_answer') and a1_refuses_c1_not(a1, c1):
            return ('a1_refusal_long', oc)
    if oc in ('refused_own_words', 'part_of_a1', 'part_of_c1', 'third_answer') and sit in DECISION_SITS:
        side = prefix_side(a2, a1, c1)
        if side and (not contains(n(c1), n(a1))) and (not contains(n(a1), n(c1))):
            return ('shortened_copy', side)
    if b == 'Ambiguous' and (m.startswith('Short answers overlap') or m.startswith('Short answer appears inside')):
        return ('x_agreement_like', oc)
    if b == 'Ambiguous' and m.startswith('A1 and Critic are both sentences'):
        return ('x_meaning', oc)
    if oc == 'refused_own_words':
        return ('x_own_refusal', oc)
    if oc in ('part_of_a1', 'part_of_c1', 'third_answer'):
        return ('x_partial_third', oc)
    return ('x_other', oc)

GROUPS = ('x_same_in_disguise', 'x_agreement_like', 'x_recover_same')
TYPE = {'county', 'city', 'state', 'province', 'parish', 'music', 'town', 'village', 'borough', 'township', 'municipality'}
FILLER = {'in', 'on', 'at', 'around', 'about', 'approximately', 'circa'}

def strip(t, words):
    return ' '.join((w for w in t.split() if w not in words))

def kind(a1, c1):
    A, C = (n(a1), n(c1))
    if max(len(a1.split()), len(c1.split())) > 6:
        return 'long'
    if A and C and (strip(A, TYPE) == strip(C, TYPE)) and strip(A, TYPE):
        return 'type word'
    parts = lambda t: [p.strip() for p in t.split(',') if p.strip()]
    pa, pc = (parts(a1.lower()), parts(c1.lower()))
    if len(pa) != len(pc) and (set(map(n, pa)) <= set(map(n, pc)) or set(map(n, pc)) <= set(map(n, pa))):
        return 'comma part'
    if A != C and strip(A, FILLER) == strip(C, FILLER) and strip(A, FILLER):
        return 'P1 filler word or punctuation'
    (s, ls), (l, ll) = sorted(((A.split(), a1), (C.split(), c1)), key=lambda x: len(x[0]))
    if s and len(l) == len(s) + 1 and (l[:-1] == s) and any((ch.isdigit() for ch in A)) and (not any((ch.isdigit() for ch in l[-1]))):
        return 'P2 number plus one word'
    raw_extra = ll.split()[:len(l) - len(s)]
    if len(s) >= 2 and len(l) > len(s) and (l[-len(s):] == s) and all((w[:1].isupper() for w in raw_extra)) and (not set((w.lower().strip(',') for w in raw_extra)) & {'and', '&', 'of'}) and (not any((ch.isdigit() for ch in ll))):
        return 'P3 fuller name'
    MONTHS = {'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'}
    ordinal = lambda t: re.sub('\\b(\\d+)(?:st|nd|rd|th)\\b', '\\1', t)
    if ordinal(A) == ordinal(C) and A != C:
        return 'P4 date detail or ordinal'
    if re.fullmatch('\\d{3,4}', ' '.join(s)) and s[0] in l and all((w in MONTHS or re.fullmatch('\\d{1,2}(?:st|nd|rd|th)?', w) for w in l if w != s[0])):
        return 'P4 date detail or ordinal'
    return 'still out'

def excluded_type(kd, a1, c1):
    if kd.startswith('dropped'):
        return 'E0 shorter side copied from the question'
    if kd == 'long':
        if refuses(c1) or refuses(a1):
            return 'E4 long refusal or objection that mentions the other answer'
        return 'E5 one side is a long sentence or passage'
    t = (a1 + ' | ' + c1).lower()
    if re.search('\\b\\d{4}-\\d{2}-\\d{2}\\b|\\b\\d{4} - \\d{2} - \\d{2}\\b', t):
        return 'E2x ISO date'
    if re.search('\\d\\s*(?:–|-|to|through|until)\\s*\\d', t) or 'onward' in t:
        return 'E2 range or span vs one end'
    if re.search('\\band\\b|&|/', t):
        return 'E1 list or two things vs one'
    return 'E3 other short pairs (different words around a shared word)'
APPROX = {'roughly', 'around', 'about', 'approximately', 'some', 'circa'}
DIRECTION = {'western', 'eastern', 'northern', 'southern', 'central', 'northwestern', 'northeastern', 'southwestern', 'southeastern', 'upper', 'lower'}
UNITS = {'per', 'cent', 'percent', 'metres', 'meters', 'feet', 'ft', 'km', 'miles', 'mi', 'years', 'people', 'inches', 'kilometres', 'kilometers', 'm'}
VALUE_WORDS = '\\b(?:more than|less than|fewer than|at least|at most|up to|over|under|nearly|almost)\\b'

def changes_value(a1, c1):
    """Same text apart from a word that changes the value ("over 30%" vs "30%")."""
    a, c = (n(a1), n(c1))
    if a == c:
        return False
    ra, rc = (' '.join(re.sub(VALUE_WORDS, ' ', t).split()) for t in (a, c))
    return bool(ra) and ra == rc and (bool(re.search(VALUE_WORDS, a)) != bool(re.search(VALUE_WORDS, c)))

def p5(a1, c1):
    """Widened same-answer check: drop bracketed text, approximation words, direction words; numbers may lose units."""
    clean = lambda t: n(re.sub('\\([^)]*\\)', ' ', t))
    A, C = (clean(a1), clean(c1))
    if A == C and A:
        return True
    lower_dir = {w.lower() for w in (a1 + ' ' + c1).split() if w.islower() and w in DIRECTION}
    A2, C2 = (strip(A, APPROX | lower_dir), strip(C, APPROX | lower_dir))
    if A2 == C2 and A2:
        return True
    if any((ch.isdigit() for ch in A2 + C2)) and strip(A2, UNITS) == strip(C2, UNITS) and strip(A2, UNITS):
        return True
    return False

def sentence_mentions(long_text, short_text):
    s = n(short_text)
    return [x for x in re.split('(?<=[.;])\\s+', long_text) if s and f' {s} ' in f' {n(x)} ']

def e4_split(a1, c1):
    long_t, short_t = (c1, a1) if len(c1.split()) >= len(a1.split()) else (a1, c1)
    sents = sentence_mentions(long_t, short_t)
    positive = [x for x in sents if not (NEG.search(x) or DIAG.search(x) or re.search('\\b(?:not|no|never|neither|nor)\\b', x, re.I))]
    return 'E4b Critic states the answer positively' if positive else 'E4a negates or refuses around the answer'
HEDGE = {'approximately', 'about', 'around', 'roughly', 'circa', 'some'}
AR_SOURCES = {'main', 'refusal', 'objection', 'a1_refusal', 'long_critic_refusal', 'long_a1_vs_refusal', 'long_critic_objection', 'missed_short_refusal', 'a1_refusal_long'}

def assign(r, lab):
    k, oc = source(r, lab)
    b, m = triage(r, lab)
    copied = oc in ('followed_c1', 'retained_a1')
    a1, c1 = (r[f] or '' for f in ('solver_attempt_1', 'critic_advocated_answer'))
    if k in AR_SOURCES:
        return ('ADOPT/RETAIN', k)
    if k == 'x_both_refused' or (k == 'x_agreement' and m == 'A1 and Critic both refused'):
        return ('BOTH REFUSED', k)
    if k == 'x_agreement':
        return ('AGREEMENT', k) if copied else ('OUT', 'agreement but A2 did not repeat')
    if k not in GROUPS:
        return ('OUT', k)
    if max(len(a1.split()), len(c1.split())) <= 6 and changes_value(a1, c1):
        return ('ADOPT/RETAIN', 'over/under vs plain value') if copied else ('OUT', 'no copy')
    kd = kind(a1, c1)
    short = min((n(a1), n(c1)), key=len)
    if kd not in ('long', 'still out') and short and (f' {short} ' in f" {n(r['question'])} "):
        kd = 'dropped'
    if kd not in ('long', 'still out', 'dropped'):
        if set(n(a1).split()) & HEDGE != set(n(c1).split()) & HEDGE:
            return ('OUT', 'hedged value (approximately 30% vs 30%)')
        return ('AGREEMENT', 'same answer: ' + kd) if copied else ('OUT', 'no copy')
    e = excluded_type(kd, a1, c1)
    if e.startswith('E3') and p5(a1, c1):
        if set(n(a1).split()) & HEDGE != set(n(c1).split()) & HEDGE:
            return ('OUT', 'hedged value (approximately 30% vs 30%)')
        return ('AGREEMENT', 'same answer: units/brackets/western') if copied else ('OUT', 'no copy')
    if e.startswith('E2x'):
        return ('AGREEMENT', 'same answer: ISO date') if copied else ('OUT', 'no copy')
    if e.startswith(('E1', 'E2')):
        return ('ADOPT/RETAIN', e[:2] + ' list/range') if copied else ('OUT', 'no copy')
    if e.startswith('E4'):
        sp = e4_split(a1, c1)
        if sp.startswith('E4a'):
            return ('ADOPT/RETAIN', 'E4a refusal mentions the answer') if copied else ('OUT', 'no copy')
        return ('OUT', 'E4b positive Critic reply (mixed)')
    return ('OUT', e[:60])

n2 = lambda t: n(re.sub('[-–—]', ' ', t or ''))
LETTER_TEXT = re.compile('^\\s*\\**\\(?Candidate\\s+([AB])\\)?\\**\\b[^:\\n]{0,60}:\\s*(.+)$', re.S | re.I)

def side_of_letter(r, letter):
    a_is_a1 = r['candidate_order'] == 'a1_first'
    return 'a1' if (letter.upper() == 'A') == a_is_a1 else 'c1'

def assign2(r, depth=0):
    lab = classify(r, V020)
    k, oc = source(r, lab)
    b, m = triage(r, lab)
    a1, c1, a2 = (r[f] or '' for f in ('solver_attempt_1', 'critic_advocated_answer', 'solver_attempt_2'))
    if k == 'x_both_refused' or (k == 'x_agreement' and m == 'A1 and Critic both refused'):
        if refuses(a2):
            return ('AGREEMENT', 'both refused, A2 keeps refusing', True)
        return ('OUT', 'both refused, A2 ' + ('letter only' if re.match('^\\W*Candidate\\s+[AB]\\W*(?:\\([^)]*\\))?\\W*$', a2, re.I) else 'gives an answer'), True)
    if depth == 0:
        mt = LETTER_TEXT.match(a2)
        if mt:
            side = side_of_letter(r, mt.group(1))
            txt = mt.group(2)
            hit = [s for s, t in (('a1', a1), ('c1', c1)) if n(t) == n(txt)]
            if hit == [side]:
                cls, why, br = assign2(dict(r, solver_attempt_2=txt), 1)
                return (cls, why + ' [letter + full text]', br)
            if hit:
                return ('OUT', 'letter and text point to different sides', False)
        tel = (r.get('generation_telemetry') or {}).get('solver_a2') or {}
        ps = prefix_side(a2, a1, c1)
        if tel.get('reached_token_budget') and ps:
            cls, why, br = assign2(dict(r, solver_attempt_2=a1 if ps == 'retained_a1' else c1), 1)
            return (cls, why + ' [cut off at 64 tokens]', br)
    if n(a1) != n(c1) and n2(a1) == n2(c1) and n2(a1):
        return ('AGREEMENT', 'same text apart from hyphen spacing', False) if n2(a2) == n2(a1) else ('OUT', 'no copy', False)
    cls, why = assign(r, lab)
    if why == 'E4b positive Critic reply (mixed)':
        long_t = c1 if len(c1.split()) >= len(a1.split()) else a1
        last = [x for x in re.split('(?<=[.;])\\s+', long_t.strip()) if x][-1]
        short_t = a1 if long_t is c1 else c1
        ends_on_answer = f' {n(short_t)} ' in f' {n(last)} ' and (not (NEG.search(last) or DIAG.search(last)))
        if DIAG.search(long_t) and (not ends_on_answer):
            return ('ADOPT/RETAIN', 'E4b long side says the information is missing', False) if oc in ('followed_c1', 'retained_a1') else ('OUT', 'no copy', False)
    return (cls, why, False)

has = lambda big, s: bool(n(s)) and f' {n(s)} ' in f' {n(big)} '
CHAIN = re.compile('\\b(?:and then|then|before being|subsequently|afterwards|from there|transported to|moved to|relocated to|transferred to|shipped to|taken to)\\b', re.I)
NUMLIKE = re.compile('^\\s*(?:\\d[\\d,.]*\\s*(?:%|percent|st|nd|rd|th)?(?:\\s+century)?|\\d{4}s)\\s*\\.?\\s*$', re.I)

def extra_rules(r, why):
    a1, c1, a2 = (r[f] or '' for f in ('solver_attempt_1', 'critic_advocated_answer', 'solver_attempt_2'))
    copied = n(a2) in (n(a1), n(c1))
    side = 'followed_c1' if n(a2) == n(c1) else 'retained_a1'
    if why == 'x_own_refusal':
        r1, rc = (refuses(a1), refuses(c1))
        if rc and (not r1):
            return ('ADOPT/RETAIN', 'R1 A2 refuses in its own words, only the Critic refused', 'followed_c1')
        if r1 and (not rc):
            return ('ADOPT/RETAIN', 'R1 A2 refuses in its own words, only A1 refused', 'retained_a1')
    if why == 'x_partial_third' and len(a2.split()) <= 8 and (not refuses(a2)):
        i1, ic = (has(a1, a2), has(c1, a2))
        if ic and (not i1) and (len(c1.split()) > len(a2.split())):
            return ('ADOPT/RETAIN', 'R2 A2 lifts its answer out of C1 only', 'followed_c1')
        if i1 and (not ic) and (len(a1.split()) > len(a2.split())):
            return ('ADOPT/RETAIN', 'R2 A2 lifts its answer out of A1 only', 'retained_a1')
    if why.startswith('E5') and copied:
        if r['critic_verdict'] == 'disagree' and len(c1.split()) > len(a1.split()) + 2 and has(c1, a1):
            rest = n(c1).split(n(a1), 1)[1]
            if CHAIN.search(rest):
                return ('ADOPT/RETAIN', "R3b Critic names A1's answer, then moves on to another", side)
            if n(c1).startswith(n(a1) + ' ') and len(rest.split()) >= 4:
                return ('ADOPT/RETAIN', "R3c Critic's reply starts with A1's answer and adds a new answer", side)
        if len(a1.split()) > len(c1.split()) and NUMLIKE.match(c1) and has(a1, c1):
            nums = re.findall('\\d[\\d,.]*', a1)
            before = n(a1).split(n(c1))[0].split()[-2:]
            if len(set(nums)) == 1 and (not re.search(VALUE_WORDS, ' '.join(before))):
                return ('AGREEMENT', "R3d A1's sentence states the Critic's number or date", None)
    if why.startswith('E3') and copied and (r['critic_verdict'] == 'disagree') and ('(' not in a1 + c1) and (not any((ch.isdigit() for ch in a1 + c1))):
        s, l = sorted((a1, c1), key=lambda t: len(t.split()))
        if set(n(l).split()) - set(n(s).split()) & set(n(r['question']).split()) - {'the', 'a', 'of', 'in'}:
            return ('ADOPT/RETAIN', 'R4b short pair, the longer side adds the thing the question asks for', side)
    return None

def assign_final(r):
    cls, why, br = assign2(r)
    if cls == 'OUT':
        x = extra_rules(r, why)
        if x:
            return (x[0], x[1], br)
    return (cls, why, br)
