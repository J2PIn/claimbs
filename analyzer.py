import json, re
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]

def find_spans_regex(pattern: re.Pattern, text: str) -> List[Dict[str, Any]]:
    return [{"start": m.start(), "end": m.end(), "text": text[m.start():m.end()]} for m in pattern.finditer(text)]

def find_phrase_spans(text_l: str, phrases: List[str], original: str) -> List[Dict[str, Any]]:
    spans = []
    for p in phrases:
        p_l = p.lower()
        start = 0
        while True:
            idx = text_l.find(p_l, start)
            if idx == -1:
                break
            spans.append({"start": idx, "end": idx + len(p_l), "text": original[idx:idx + len(p_l)]})
            start = idx + len(p_l)
    return spans

def make_flag(flag_type: str, label: str, spans: List[Dict[str, Any]]):
    return {"type": flag_type, "label": label, "spans": spans}

@dataclass
class SentenceAnalysis:
    sentence: str
    semantic_class: str
    features: Dict[str, bool]
    flags: List[Dict[str, Any]]
    score_breakdown: Dict[str, int]
    total_score: int

class Analyzer:
    def __init__(self, rules_dir: str):
        lex = load_json(f"{rules_dir}/lexicons.json")
        rx  = load_json(f"{rules_dir}/regex.json")
        wt  = load_json(f"{rules_dir}/weights.json")
        sc  = load_json(f"{rules_dir}/semantic_classes.json")

        self.lex = lex
        self.wt = wt
        self.sc = sc

        self.re_number_unit = re.compile(rx["number_unit"], re.IGNORECASE)
        self.re_timeframe   = re.compile(rx["timeframe"], re.IGNORECASE)
        self.re_baseline    = re.compile(rx["baseline"], re.IGNORECASE)
        self.re_scope       = re.compile(rx["scope"], re.IGNORECASE)
        self.re_passive     = re.compile(rx["passive"], re.IGNORECASE)

    def analyze_sentence(self, s: str) -> SentenceAnalysis:
        s_l = s.lower()

        # phrase spans
        outcome_spans = find_phrase_spans(s_l, self.lex["outcome_verbs"], s)
        speed_spans   = find_phrase_spans(s_l, self.lex["speed_words"], s)
        hedge_spans   = find_phrase_spans(s_l, self.lex["hedges"], s)
        super_spans   = find_phrase_spans(s_l, self.lex["superlatives"], s)
        proof_spans   = find_phrase_spans(s_l, self.lex["proof_theater"], s)
        buzz_spans    = find_phrase_spans(s_l, self.lex["buzzwords"], s)
        mech_spans    = find_phrase_spans(s_l, self.lex["mechanism_tokens"], s)
        values_spans  = find_phrase_spans(s_l, self.lex["values_language"], s)

        # regex spans
        metric_spans    = find_spans_regex(self.re_number_unit, s)
        timeframe_spans = find_spans_regex(self.re_timeframe, s)
        baseline_spans  = find_spans_regex(self.re_baseline, s)
        scope_spans     = find_spans_regex(self.re_scope, s)
        passive_spans   = find_spans_regex(self.re_passive, s)

        # features
        features = {
            "outcome_present": bool(outcome_spans),
            "speed_present": bool(speed_spans),
            "hedge_present": bool(hedge_spans),
            "superlative_present": bool(super_spans),
            "proof_implied": bool(proof_spans),
            "buzzword_present": bool(buzz_spans),
            "mechanism_present": bool(mech_spans),
            "values_present": bool(values_spans),

            "metric_present": bool(metric_spans),
            "timeframe_present": bool(timeframe_spans),
            "baseline_present": bool(baseline_spans),
            "scope_present": bool(scope_spans),
            "passive_present": bool(passive_spans)
        }

        # semantic class (Option B: label values language)
        semantic_class = "operational_or_mixed"
        rule = self.sc.get("values_non_operational")
        if rule:
            if all(features.get(k, False) for k in rule["requires"]) and all(not features.get(k, False) for k in rule["forbids"]):
                semantic_class = "values_non_operational"

        flags: List[Dict[str, Any]] = []
        penalties = 0

        P = self.wt["penalties"]
        B = self.wt["bonuses"]

        # penalties (with spans)
        if features["outcome_present"] and not features["metric_present"]:
            penalties += P["outcome_without_metric"]
            flags.append(make_flag("outcome_without_metric", "Outcome claim without metric", outcome_spans))

        if features["speed_present"] and not features["timeframe_present"]:
            penalties += P["speed_without_timeframe"]
            flags.append(make_flag("speed_without_timeframe", "Speed claim without timeframe", speed_spans))

        if features["outcome_present"] and not features["baseline_present"]:
            penalties += P["outcome_without_baseline"]
            flags.append(make_flag("outcome_without_baseline", "Outcome claim without baseline", outcome_spans))

        if features["outcome_present"] and not features["scope_present"]:
            penalties += P["outcome_without_scope"]
            flags.append(make_flag("outcome_without_scope", "Outcome claim without scope", outcome_spans))

        if features["proof_implied"]:
            penalties += P["proof_implied_no_evidence"]
            flags.append(make_flag("proof_implied_no_evidence", "Proof implied without evidence structure", proof_spans))

        if features["buzzword_present"] and not features["mechanism_present"]:
            penalties += P["buzzword_no_mechanism"]
            flags.append(make_flag("buzzword_no_mechanism", "Buzzword used as mechanism substitute", buzz_spans))

        if features["outcome_present"] and not features["mechanism_present"]:
            penalties += P["outcome_without_mechanism"]
            flags.append(make_flag("outcome_without_mechanism", "Outcome claim without concrete mechanism", outcome_spans))

        if re.search(r"\b(discovery|strategy|execution|optimization|planning process|customer journey)\b", s_l) and not features["mechanism_present"]:
            penalties += P["process_no_levers"]
            flags.append(make_flag("process_no_levers", "Process language without levers", [{"start": 0, "end": len(s), "text": s}]))

        if features["passive_present"]:
            penalties += P["passive_voice"]
            flags.append(make_flag("passive_voice", "Passive voice / agency evasion", passive_spans))

        if features["superlative_present"] and not features["metric_present"]:
            penalties += P["superlative_unbounded"]
            flags.append(make_flag("superlative_unbounded", "Superlative without constraint", super_spans))

        # bonuses
        bonus = 0
        if features["metric_present"]: bonus += B["metric_present"]
        if features["timeframe_present"]: bonus += B["timeframe_present"]
        if features["baseline_present"]: bonus += B["baseline_present"]
        if features["mechanism_present"]: bonus += B["mechanism_present"]
        if features["scope_present"]: bonus += B["scope_present"]

        total = max(0, min(100, penalties - bonus))

        return SentenceAnalysis(
            sentence=s,
            semantic_class=semantic_class,
            features=features,
            flags=flags,
            score_breakdown={"penalties": penalties, "bonus": bonus},
            total_score=total
        )

    def analyze_text(self, text: str) -> Dict[str, Any]:
        sents = split_sentences(text)
        analyses = [self.analyze_sentence(s) for s in sents]
        if not analyses:
            return {"overall": {"score_mean": 0, "label": "Empty", "n_sentences": 0}, "sentences": []}

        scores = [a.total_score for a in analyses]
        mean = sum(scores) / len(scores)
        worst = max(scores)
        pct_high = 100 * sum(1 for x in scores if x >= 51) / len(scores)

        def label(x: float) -> str:
            if x <= 25: return "Solid / operational"
            if x <= 50: return "Soft marketing"
            if x <= 75: return "High BS risk"
            return "Persuasion fog"

        return {
            "overall": {
                "score_mean": round(mean, 1),
                "score_worst_sentence": worst,
                "pct_sentences_high_bs": round(pct_high, 1),
                "label": label(mean),
                "n_sentences": len(analyses)
            },
            "sentences": [asdict(a) for a in analyses]
        }

if __name__ == "__main__":
    analyzer = Analyzer("rules/agency_v0")
    text = """Our advertising contracts are always open-ended.
Instead of relying on termination periods, we prefer to foster long-term customer relationships by delivering on our promises—producing effective advertising."""
    import json
    print(json.dumps(analyzer.analyze_text(text), indent=2, ensure_ascii=False))
