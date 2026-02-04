// Cloudflare Worker (Module syntax) — deterministic claim analysis
// Endpoints:
//   GET /analyze?text=...
//   POST /analyze  { text: "...", mode?: "agency_v0" }
//   GET /analyze-url?url=...

import lexicons from "./rules/agency_v0/lexicons.json";
import regexes from "./rules/agency_v0/regex.json";
import weights from "./rules/agency_v0/weights.json";
import semanticClasses from "./rules/agency_v0/semantic_classes.json";

type Span = { start: number; end: number; text: string };
type Flag = { type: string; label: string; spans: Span[] };

type SentenceAnalysis = {
  sentence: string;
  semantic_class: string;
  features: Record<string, boolean>;
  flags: Flag[];
  score_breakdown: { penalties: number; bonus: number };
  total_score: number;
};

type AnalyzeResult = {
  overall: {
    score_mean: number;
    score_worst_sentence: number;
    pct_sentences_high_bs: number;
    label: string;
    n_sentences: number;
  };
  sentences: SentenceAnalysis[];
};

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "access-control-allow-origin": "*",
      "access-control-allow-methods": "GET,POST,OPTIONS",
      "access-control-allow-headers": "content-type",
      "cache-control": "no-store",
    },
  });
}

function splitSentences(text: string): string[] {
  const cleaned = text.replace(/\s+/g, " ").trim();
  if (!cleaned) return [];
  // deterministic sentence split (simple)
  return cleaned.split(/(?<=[.!?])\s+/).map(s => s.trim()).filter(Boolean);
}

function findSpansRegex(re: RegExp, text: string): Span[] {
  const spans: Span[] = [];
  // ensure global
  const flags = re.flags.includes("g") ? re.flags : re.flags + "g";
  const rg = new RegExp(re.source, flags);
  let m: RegExpExecArray | null;
  while ((m = rg.exec(text)) !== null) {
    spans.push({ start: m.index, end: m.index + m[0].length, text: text.slice(m.index, m.index + m[0].length) });
    // avoid infinite loops on zero-length matches
    if (m[0].length === 0) rg.lastIndex++;
  }
  return spans;
}

function findPhraseSpans(textLower: string, phrases: string[], original: string): Span[] {
  const spans: Span[] = [];
  for (const p of phrases) {
    const pL = p.toLowerCase();
    let start = 0;
    while (true) {
      const idx = textLower.indexOf(pL, start);
      if (idx === -1) break;
      spans.push({ start: idx, end: idx + pL.length, text: original.slice(idx, idx + pL.length) });
      start = idx + pL.length;
    }
  }
  return spans;
}

function makeFlag(type: string, label: string, spans: Span[]): Flag {
  return { type, label, spans };
}

function labelScore(x: number): string {
  if (x <= 25) return "Solid / operational";
  if (x <= 50) return "Soft marketing";
  if (x <= 75) return "High BS risk";
  return "Persuasion fog";
}

// Basic HTML → text extractor (deterministic, no DOM libs)
function htmlToText(html: string): string {
  // remove scripts/styles
  let t = html.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, " ");
  t = t.replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, " ");
  // remove noscript
  t = t.replace(/<noscript\b[^>]*>[\s\S]*?<\/noscript>/gi, " ");
  // replace breaks with newline-ish
  t = t.replace(/<(br|p|div|li|h\d|tr|td)\b[^>]*>/gi, "\n");
  // strip tags
  t = t.replace(/<\/?[^>]+>/g, " ");
  // decode a few common entities deterministically
  t = t
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, "\"")
    .replace(/&#39;/g, "'")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">");
  // collapse whitespace but keep some newlines
  t = t.replace(/[ \t\r\f\v]+/g, " ");
  t = t.replace(/\n{3,}/g, "\n\n");
  return t.trim();
}

function analyzeSentence(s: string): SentenceAnalysis {
  const sL = s.toLowerCase();

  // compile regexes (from config)
  const RE_NUMBER_UNIT = new RegExp(regexes.number_unit, "i");
  const RE_TIMEFRAME   = new RegExp(regexes.timeframe, "i");
  const RE_BASELINE    = new RegExp(regexes.baseline, "i");
  const RE_SCOPE       = new RegExp(regexes.scope, "i");
  const RE_PASSIVE     = new RegExp(regexes.passive, "i");

  // spans
  const outcomeSpans = findPhraseSpans(sL, lexicons.outcome_verbs, s);
  const speedSpans   = findPhraseSpans(sL, lexicons.speed_words, s);
  const hedgeSpans   = findPhraseSpans(sL, lexicons.hedges, s);
  const superSpans   = findPhraseSpans(sL, lexicons.superlatives, s);
  const proofSpans   = findPhraseSpans(sL, lexicons.proof_theater, s);
  const buzzSpans    = findPhraseSpans(sL, lexicons.buzzwords, s);
  const mechSpans    = findPhraseSpans(sL, lexicons.mechanism_tokens, s);
  const valuesSpans  = findPhraseSpans(sL, lexicons.values_language, s);

  const metricSpans    = findSpansRegex(new RegExp(regexes.number_unit, "ig"), s);
  const timeframeSpans = findSpansRegex(new RegExp(regexes.timeframe, "ig"), s);
  const baselineSpans  = findSpansRegex(new RegExp(regexes.baseline, "ig"), s);
  const scopeSpans     = findSpansRegex(new RegExp(regexes.scope, "ig"), s);
  const passiveSpans   = findSpansRegex(new RegExp(regexes.passive, "ig"), s);

  const features: Record<string, boolean> = {
    outcome_present: outcomeSpans.length > 0,
    speed_present: speedSpans.length > 0,
    hedge_present: hedgeSpans.length > 0,
    superlative_present: superSpans.length > 0,
    proof_implied: proofSpans.length > 0,
    buzzword_present: buzzSpans.length > 0,
    mechanism_present: mechSpans.length > 0,
    values_present: valuesSpans.length > 0,

    metric_present: RE_NUMBER_UNIT.test(s),
    timeframe_present: RE_TIMEFRAME.test(s),
    baseline_present: RE_BASELINE.test(s),
    scope_present: RE_SCOPE.test(s),
    passive_present: RE_PASSIVE.test(s),
  };

  // semantic class (Option B)
  let semantic_class = "operational_or_mixed";
  const vRule = (semanticClasses as any).values_non_operational;
  if (vRule) {
    const requires: string[] = vRule.requires ?? [];
    const forbids: string[] = vRule.forbids ?? [];
    const okReq = requires.every(k => Boolean(features[k]));
    const okForb = forbids.every(k => !features[k]);
    if (okReq && okForb) semantic_class = "values_non_operational";
  }

  const P = (weights as any).penalties as Record<string, number>;
  const B = (weights as any).bonuses as Record<string, number>;

  let penalties = 0;
  const flags: Flag[] = [];

  if (features.outcome_present && !features.metric_present) {
    penalties += P.outcome_without_metric;
    flags.push(makeFlag("outcome_without_metric", "Outcome claim without metric", outcomeSpans));
  }

  if (features.speed_present && !features.timeframe_present) {
    penalties += P.speed_without_timeframe;
    flags.push(makeFlag("speed_without_timeframe", "Speed claim without timeframe", speedSpans));
  }

  if (features.outcome_present && !features.baseline_present) {
    penalties += P.outcome_without_baseline;
    flags.push(makeFlag("outcome_without_baseline", "Outcome claim without baseline", outcomeSpans));
  }

  if (features.outcome_present && !features.scope_present) {
    penalties += P.outcome_without_scope;
    flags.push(makeFlag("outcome_without_scope", "Outcome claim without scope", outcomeSpans));
  }

  if (features.proof_implied) {
    penalties += P.proof_implied_no_evidence;
    flags.push(makeFlag("proof_implied_no_evidence", "Proof implied without evidence structure", proofSpans));
  }

  if (features.buzzword_present && !features.mechanism_present) {
    penalties += P.buzzword_no_mechanism;
    flags.push(makeFlag("buzzword_no_mechanism", "Buzzword used as mechanism substitute", buzzSpans));
  }

  if (features.outcome_present && !features.mechanism_present) {
    penalties += P.outcome_without_mechanism;
    flags.push(makeFlag("outcome_without_mechanism", "Outcome claim without concrete mechanism", outcomeSpans));
  }

  // process language heuristic (deterministic)
  if (/\b(discovery|strategy|execution|optimization|planning process|customer journey)\b/i.test(s) && !features.mechanism_present) {
    penalties += P.process_no_levers;
    flags.push(makeFlag("process_no_levers", "Process language without levers", [{ start: 0, end: s.length, text: s }]));
  }

  if (features.passive_present) {
    penalties += P.passive_voice;
    flags.push(makeFlag("passive_voice", "Passive voice / agency evasion", passiveSpans));
  }

  if (features.superlative_present && !features.metric_present) {
    penalties += P.superlative_unbounded;
    flags.push(makeFlag("superlative_unbounded", "Superlative without constraint", superSpans));
  }

  // bonuses
  let bonus = 0;
  if (features.metric_present) bonus += B.metric_present;
  if (features.timeframe_present) bonus += B.timeframe_present;
  if (features.baseline_present) bonus += B.baseline_present;
  if (features.mechanism_present) bonus += B.mechanism_present;
  if (features.scope_present) bonus += B.scope_present;

  const total_score = Math.max(0, Math.min(100, penalties - bonus));

  return {
    sentence: s,
    semantic_class,
    features,
    flags,
    score_breakdown: { penalties, bonus },
    total_score,
  };
}

function analyzeText(text: string): AnalyzeResult {
  const sents = splitSentences(text);
  const analyses = sents.map(analyzeSentence);

  if (analyses.length === 0) {
    return {
      overall: { score_mean: 0, score_worst_sentence: 0, pct_sentences_high_bs: 0, label: "Empty", n_sentences: 0 },
      sentences: [],
    };
  }

  const scores = analyses.map(a => a.total_score);
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
  const worst = Math.max(...scores);
  const pctHigh = (scores.filter(x => x >= 51).length / scores.length) * 100;

  return {
    overall: {
      score_mean: Math.round(mean * 10) / 10,
      score_worst_sentence: worst,
      pct_sentences_high_bs: Math.round(pctHigh * 10) / 10,
      label: labelScore(mean),
      n_sentences: analyses.length,
    },
    sentences: analyses,
  };
}

async function readJson(req: Request): Promise<any> {
  const ct = req.headers.get("content-type") || "";
  if (!ct.includes("application/json")) return null;
  try {
    return await req.json();
  } catch {
    return null;
  }
}

function badRequest(msg: string) {
  return json({ error: msg }, 400);
}

export default {
  async fetch(req: Request): Promise<Response> {
    if (req.method === "OPTIONS") return json({ ok: true }, 204);

    const url = new URL(req.url);
    const path = url.pathname;

    if (req.method === "GET" && path === "/health") {
      return json({ ok: true, mode: "agency_v0" });
    }

    if (req.method === "GET" && path === "/analyze") {
      const text = url.searchParams.get("text") || "";
      if (!text.trim()) return badRequest("Missing ?text=");
      return json(analyzeText(text));
    }

    if (req.method === "POST" && path === "/analyze") {
      const body = await readJson(req);
      if (!body?.text || typeof body.text !== "string") return badRequest("Body must be { text: string }");
      return json(analyzeText(body.text));
    }

    if (req.method === "GET" && path === "/analyze-url") {
      const target = url.searchParams.get("url");
      if (!target) return badRequest("Missing ?url=");
      let parsed: URL;
      try {
        parsed = new URL(target);
      } catch {
        return badRequest("Invalid url");
      }
      if (!["http:", "https:"].includes(parsed.protocol)) return badRequest("Only http/https allowed");

      const resp = await fetch(parsed.toString(), {
        headers: {
          "user-agent": "ClaimBullshitBot/0.1 (+deterministic-analysis)",
          "accept": "text/html,application/xhtml+xml",
        },
      });

      const ctype = resp.headers.get("content-type") || "";
      const raw = await resp.text();

      // If it's not HTML, just analyze raw text
      const extracted = ctype.includes("text/html") ? htmlToText(raw) : raw;

      // Optional: limit input size deterministically to avoid huge pages
      const MAX_CHARS = 40_000;
      const clipped = extracted.length > MAX_CHARS ? extracted.slice(0, MAX_CHARS) : extracted;

      const analysis = analyzeText(clipped);

      return json({
        fetched_url: parsed.toString(),
        content_type: ctype,
        extracted_chars: clipped.length,
        ...analysis,
      });
    }

    return json({ error: "Not found. Try /health, /analyze, /analyze-url" }, 404);
  },
};
