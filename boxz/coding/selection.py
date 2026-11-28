"""Decide which template (if any) a coding request wants.

This is a **deterministic matcher**, not a model.  It is built from the real
examples in ``coding_requests.txt``: every template block contributes the
nouns and verbs that appear in its requests, weighted by how specific they
are across the whole corpus.  A word that appears in every game block carries
almost no signal; a word that appears in one block carries a lot.  That is
plain TF-IDF, no training and no retraining.

Three outcomes, and each one is a *success*:

``match``      one template scored clearly above the rest -> build it
``clarify``    a coding request with no clear winner -> ask which one
``unsupported`` another language/technology/framework was asked for -> decline

The matcher refuses to guess.  If the top two templates are within
``min_margin`` of each other the answer is a clarification, because building
the wrong game is worse than asking one short question.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .requests import RequestBlock, RequestCorpus, parse_coding_requests
from .templates import Template

__all__ = [
    "Selection",
    "TemplateSelector",
    "STOPWORDS",
]

#: Words that carry no topical signal.  Deliberately generous: "make", "game"
#: and "page" appear in nearly every request and would otherwise attract
#: everything to everything.
STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "for", "to", "of", "in", "on", "at",
    "by", "with", "from", "into", "onto", "over", "under", "up", "down", "out",
    "i", "me", "my", "we", "you", "your", "it", "its", "this", "that", "these",
    "those", "is", "are", "was", "were", "be", "been", "being", "am", "do",
    "does", "did", "can", "could", "would", "should", "will", "shall", "may",
    "might", "must", "have", "has", "had", "want", "need", "like", "please",
    "make", "made", "create", "build", "write", "give", "let", "get", "put",
    "add", "show", "use", "using", "want", "some", "any", "one", "two", "three",
    "game", "games", "app", "apps", "page", "pages", "website", "websites",
    "simple", "basic", "small", "little", "nice", "cool", "good", "great",
    "new", "clean", "modern", "browser", "web", "html", "css", "javascript",
    "js", "code", "thing", "things", "something", "stuff", "way", "kind",
    "sort", "type", "also", "just", "really", "very", "more", "most", "less",
    "than", "then", "there", "here", "when", "while", "as", "if", "so", "not",
    "no", "yes", "all", "each", "every", "both", "few", "many", "much", "own",
    "same", "other", "another", "such", "only", "about", "after", "before",
    "again", "still", "already", "now", "how", "what", "which", "who", "why",
    "where", "able", "able", "help", "boxz", "start", "run", "work", "works",
}

#: Keywords that mean "this is not a coding request at all".
_CONVERSATIONAL = re.compile(
    r"^\s*(hi|hello|hey|yo|good morning|good evening|thanks|thank you|"
    r"how are you|what can you|who are you|okay|ok|lets go)\b", re.I)

#: Technology words that the template library cannot serve.  Matched against
#: the request text to produce an honest "unsupported" instead of a bad guess.
_UNSUPPORTED_TECH = {
    "react": "framework_not_supported", "vue": "framework_not_supported",
    "angular": "framework_not_supported", "svelte": "framework_not_supported",
    "next.js": "framework_not_supported", "nextjs": "framework_not_supported",
    "laravel": "framework_not_supported", "django": "framework_not_supported",
    "flask": "framework_not_supported", "express": "framework_not_supported",
    "flutter": "framework_not_supported", "react native": "framework_not_supported",
    "three.js": "technology_not_supported", "threejs": "technology_not_supported",
    "webgl": "technology_not_supported", "unity": "technology_not_supported",
    "unreal": "technology_not_supported", "godot": "technology_not_supported",
    "android": "technology_not_supported", "ios": "technology_not_supported",
    "swift": "language_not_supported", "kotlin": "language_not_supported",
    "python": "language_not_supported", "java": "language_not_supported",
    "c++": "language_not_supported", "cpp": "language_not_supported",
    "rust": "language_not_supported", "golang": "language_not_supported",
    "php": "language_not_supported", "ruby": "language_not_supported",
    "cuda": "language_not_supported", "objc": "language_not_supported",
    "virtual reality": "technology_not_supported",
    "augmented reality": "technology_not_supported",
    "vr": "technology_not_supported", "ar": "technology_not_supported",
    "3d object": "technology_not_supported", "3d scene": "technology_not_supported",
    "orbit controls": "technology_not_supported",
}

#: Words that are also ordinary English or are too short to be safe on their
#: own.  "react" must not fire on "make the page react to the mouse"; "c" must
#: not fire inside "canvas".  These need a programming context nearby.
_CONTEXT_TECH = {
    "react": "framework_not_supported",
    "express": "framework_not_supported",
    "c": "language_not_supported",
    "go": "language_not_supported",
    "r": "language_not_supported",
    "windows": "technology_not_supported",
    "macos": "technology_not_supported",
    "linux": "technology_not_supported",
    "desktop": "technology_not_supported",
    "mobile app": "technology_not_supported",
}

#: "make the page react to the mouse" is a verb, not a framework.
_VERB_REACT = re.compile(r"\breact(s|ed|ing)?\s+(to|when|on)\b", re.I)

#: Nouns that put an ambiguous word in a programming context ("a C *program*",
#: "a React *dashboard*").  Deliberately excludes "on"/"in"/"with": "make
#: windows pop up on click" is a DOM request, not an operating system.
_TECH_CONTEXT = re.compile(
    r"\b(app|apps|application|applications|program|programs|script|scripts|"
    r"project|projects|website|websites|page|pages|site|server|backend|"
    r"frontend|code|coding|console|cli|component|components|dashboard|api|"
    r"native|development|develop|framework|library|stack|package|module)\b",
    re.I)

#: "written in C", "built using React" -- a preposition directly before the name.
_TECH_PREP = re.compile(r"\b(in|using|via|with)\s+(the\s+)?$", re.I)


def _mentions_tech(low: str, word: str) -> bool:
    """Is ``word`` used as a technology, rather than as plain English?"""
    pattern = r"(?<![a-z0-9+#])" + re.escape(word) + r"(?![a-z0-9+#])"
    if not re.search(pattern, low):
        return False
    if word not in _CONTEXT_TECH:
        return True
    if word == "react" and _VERB_REACT.search(low):
        return False
    # "React dashboard" / "a C program" / "Windows desktop application": the
    # context noun must be *next to* the name, not merely somewhere in the
    # sentence.  "I need a personal site but ... what should go on it" has
    # "site" in it, and used to turn "go" into the Go language.
    nearby = re.search(pattern + r"(\s+\w+){0,2}\s+" + _TECH_CONTEXT.pattern, low)
    if nearby:
        return True
    match = re.search(pattern, low)
    return bool(match and _TECH_PREP.search(low[:match.start()]))


#: A bare "python" must not fire on "python snake game" if a Python template
#: ever exists; today there is none, so any Python request is unsupported.


def _tokens(text: str) -> List[str]:
    """Lower-case word tokens, keeping simple tech names together."""
    text = text.lower()
    text = text.replace("three.js", "threejs").replace("c++", "cpp")
    text = re.sub(r"[^a-z0-9_+#. ]+", " ", text)
    words = [w.strip("._") for w in text.split()]
    return [w for w in words if w and w not in STOPWORDS and len(w) > 1]


@dataclass
class Selection:
    """What the matcher concluded."""

    outcome: str                        # "match" | "clarify" | "unsupported" | "none"
    template_id: Optional[str] = None
    score: float = 0.0
    runner_up: Optional[str] = None
    runner_up_score: float = 0.0
    reason: str = ""
    candidates: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def matched(self) -> bool:
        return self.outcome == "match"

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "template_id": self.template_id,
            "score": round(self.score, 4),
            "runner_up": self.runner_up,
            "runner_up_score": round(self.runner_up_score, 4),
            "reason": self.reason,
            "candidates": [(tid, round(s, 4)) for tid, s in self.candidates[:5]],
        }


class TemplateSelector:
    """TF-IDF matcher over the request corpus, restricted to real templates."""

    def __init__(
        self,
        corpus: RequestCorpus,
        templates: Dict[str, Template],
        min_score: float = 0.16,
        min_margin: float = 0.35,
        enforce_categories: bool = True,
    ) -> None:
        self.corpus = corpus
        self.templates = templates
        self.min_score = min_score
        self.min_margin = min_margin
        self.enforce_categories = enforce_categories

        #: template id -> its own vocabulary (from the requests in its block)
        self._tokens: Dict[str, Dict[str, float]] = {}
        #: template id -> extra keyword tokens from template.json
        self._keywords: Dict[str, Set[str]] = {}
        self._build()

    # -- index -----------------------------------------------------------
    def _build(self) -> None:
        blocks = [b for b in self.corpus.templates if b.id in self.templates]
        if not blocks:
            raise ValueError(
                "no template blocks in the corpus match a loaded template; "
                f"corpus has {len(self.corpus.templates)} template blocks, "
                f"library has {len(self.templates)} templates"
            )
        self._blocks: Dict[str, RequestBlock] = {b.id: b for b in blocks}

        # document frequency across blocks, for the idf term
        df: Dict[str, int] = {}
        per_block: Dict[str, Set[str]] = {}
        for b in blocks:
            vocab: Set[str] = set()
            for req in b.requests:
                vocab.update(_tokens(req))
            per_block[b.id] = vocab
            for w in vocab:
                df[w] = df.get(w, 0) + 1

        n = len(blocks)
        for b in blocks:
            extra = set()
            tpl = self.templates[b.id]
            for kw in tpl.keywords + [tpl.task]:
                extra.update(_tokens(kw))
            for w in extra:
                # keywords count as if they appeared in one extra block
                df.setdefault(w, 1)
            per_block[b.id] |= extra
            vocab = per_block[b.id]
            weights = {}
            for w in vocab:
                idf = math.log((1 + n) / (1 + df.get(w, 1))) + 1.0
                # a keyword is a stronger claim than an incidental word
                boost = 1.6 if w in extra else 1.0
                weights[w] = idf * boost
            total = math.sqrt(sum(v * v for v in weights.values())) or 1.0
            self._tokens[b.id] = {w: v / total for w, v in weights.items()}
            self._keywords[b.id] = extra

        # the vocabulary tells us whether a word is even routeable
        self._known: Set[str] = set()
        for d in self._tokens.values():
            self._known.update(d)

    # -- scoring ---------------------------------------------------------
    def score_all(self, text: str) -> List[Tuple[str, float]]:
        toks = _tokens(text)
        if not toks:
            return []
        weights = {}
        for w in toks:
            weights[w] = weights.get(w, 0.0) + 1.0
        norm = math.sqrt(sum(v * v for v in weights.values())) or 1.0
        weights = {w: v / norm for w, v in weights.items()}
        scored: List[Tuple[str, float]] = []
        for tid, vec in self._tokens.items():
            s = sum(weights.get(w, 0.0) * vec.get(w, 0.0) for w in weights)
            scored.append((tid, s))
        scored.sort(key=lambda t: (-t[1], t[0]))
        return scored

    def _keyword_scores(self, low: str) -> List[Tuple[str, int]]:
        """How many of each template's own keywords appear in the request."""
        hits: List[Tuple[str, int]] = []
        for tid, tpl in self.templates.items():
            if not tpl.keywords:
                hits.append((tid, 0))
                continue
            n = 0
            for word in tpl.keywords:
                if re.search(r"(?<![a-z])" + re.escape(word) + r"(?![a-z])", low):
                    n += 1
            hits.append((tid, n))
        hits.sort(key=lambda t: (-t[1], t[0]))
        return hits

    # -- decisions -------------------------------------------------------
    def classify(self, text: str) -> Selection:
        """Full decision: unsupported / match / clarify / none."""
        raw = (text or "").strip()
        if not raw:
            return Selection("none", reason="empty request")

        toks = _tokens(raw)

        # 1. an explicit technology we cannot serve wins over everything:
        #    the user named something concrete that is out of scope.
        low = raw.lower()
        table = {**_UNSUPPORTED_TECH, **_CONTEXT_TECH}
        for word, reason in table.items():
            if _mentions_tech(low, word):
                return Selection("unsupported", reason=reason,
                                 template_id=None,
                                 score=0.0)

        # 2. pure conversation is not a coding request
        if _CONVERSATIONAL.match(raw):
            return Selection("none", reason="conversational")

        # "make something cool", "build something fun" -- there is a wish to
        # build, but nothing to build.  Asking is the honest answer.
        if (_VAGUE_WISH.search(raw)
                and not re.search(r"\b(game|website|page|site|app|animation|"
                                  r"button|counter|timer|effect|pong|snake|"
                                  r"puzzle|platformer)\b", low)):
            return Selection("clarify", reason="task_not_specified")

        scored = self.score_all(raw)
        if not scored:
            # "Can you build a small game for my website?" is made entirely of
            # stopwords, yet it is plainly a coding wish.  Ask instead of
            # pretending it was conversation.
            if _names_a_buildable_kind(raw):
                return Selection("clarify", reason="no template matched confidently")
            return Selection("none", reason="no recognisable coding words")

        top_id, top_score = scored[0]
        second_id, second_score = scored[1] if len(scored) > 1 else ("", 0.0)

        # A single weak word is not evidence.  "first" and "second" both match
        # some template vocabulary, but one generic token cannot justify
        # claiming (or even clarifying) a coding turn -- that belongs to the
        # model.  Require either two distinct informative tokens or a strong
        # score before acting at all.
        # One distinctive word ("snake", "pong", "stopwatch") is strong enough
        # to act on; one *generic* word ("first", "second") is not.  The test is
        # whether the single token is truly characteristic of its template, so
        # A distinctive template keyword is direct evidence, and TF-IDF alone
        # punishes it for being in a longer sentence: "Make a 30x30 snake game"
        # averages to 0.13 while "Make a snake game" scores 0.18.  A keyword hit
        # is the user naming the thing, so it short-circuits the threshold --
        # but only when it is not a dead heat with another template.
        kw = self._keyword_scores(low)
        if kw and kw[0][1] >= 1:
            best = kw[0][0]
            # Another template matching just as many keywords means the request
            # is genuinely ambiguous ("a game with a ball and a paddle": pong or
            # snake?), so let the normal margin rules handle it.
            runner = kw[1][1] if len(kw) > 1 else 0
            if kw[0][1] > runner:
                return Selection("match", template_id=best, score=top_score,
                                 candidates=scored)

        # require it to clear the normal threshold on its own.
        informative = [w for w in toks if w in self._known]
        if len(informative) < 2 and top_score < self.min_score:
            # A vague request that still names something we build ("a browser
            # game with some fun mechanics") should ask, not shrug.  Only a
            # request that names nothing buildable falls through to the model.
            if _names_a_buildable_kind(raw):
                return Selection("clarify",
                                 reason="no template matched confidently",
                                 score=top_score, candidates=scored)
            return Selection("none", reason="too little signal",
                             score=top_score, candidates=scored)

        if top_score < self.min_score:
            # it looked like a coding wish but nothing matched well
            codingish = (bool(set(toks) & self._known)
                         or _looks_like_coding(raw)
                         or _names_a_buildable_kind(raw))
            if codingish:
                return Selection(
                    "clarify", reason="no template matched confidently",
                    score=top_score, runner_up=second_id or None,
                    runner_up_score=second_score, candidates=scored)
            return Selection("none", reason="not a coding request",
                             candidates=scored)

        # a margin check turns "two games are close" into a question
        if second_score > 0 and top_score > 0:
            margin = (top_score - second_score) / top_score
            if margin < self.min_margin:
                return Selection(
                    "clarify",
                    reason=(f"ambiguous between {top_id} and {second_id}"),
                    score=top_score, runner_up=second_id,
                    runner_up_score=second_score, candidates=scored)

        return Selection("match", template_id=top_id, score=top_score,
                         runner_up=second_id or None,
                         runner_up_score=second_score, candidates=scored)

    # -- helpers ---------------------------------------------------------
    def block_for(self, template_id: str) -> Optional[RequestBlock]:
        return self._blocks.get(template_id)

    def template_ids(self) -> List[str]:
        return sorted(self._tokens)


#: "make me something", "build something nice" -- a wish with no object.
_VAGUE_WISH = re.compile(
    r"\b(make|create|build|do|code|write|start)\b[^.?!]{0,18}"
    r"\b(something|anything|stuff|things)\b", re.I)

#: Nouns for things this system genuinely builds.  A request containing one is
#: a coding wish even when no template scores high enough to claim it.
_BUILDABLE = re.compile(
    r"\b(game|games|website|websites|page|pages|site|homepage|app|application|"
    r"animation|animations|button|buttons|effect|effects|canvas|layout|timer|"
    r"stopwatch|countdown|counter|menu|modal|popup|gallery|form|dashboard|"
    r"portfolio|quiz|puzzle|platformer|shooter|particles|starfield)\b", re.I)


def _names_a_buildable_kind(text: str) -> bool:
    """True when the request names something we could plausibly build."""
    return bool(_BUILDABLE.search(text or ""))


def _looks_like_coding(text: str) -> bool:
    return bool(re.search(
        r"\b(game|site|website|page|button|animation|app|background|component|"
        r"effect|canvas|layout|timer|counter|menu|modal|gallery|form)\b",
        text, re.I))
