//! Minimal, dependency-free text utilities shared across components.
//!
//! Deliberately simple and stable: the DevPULSE models do their own
//! tokenization; this exists so the default (weightless) components — the
//! deterministic embedder, the lexical reranker, the heuristic classifier —
//! agree on what a "token" is.

/// Lowercase the input and split it into alphanumeric tokens.
///
/// Runs of non-alphanumeric characters are separators. Empty tokens are
/// dropped. This is intentionally naive; it is a floor, not the model.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

/// A small English stopword set for the lexical fallbacks: articles,
/// conjunctions, prepositions, pronouns, and the question/command fillers that
/// pad natural-language queries ("how **do** I", "**tell** me **about**").
pub const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with", "what", "how", "why", "when", "where", "who",
    // query/command fillers and pronouns
    "do", "does", "did", "i", "me", "my", "we", "our", "you", "your", "can", "could", "would",
    "should", "please", "tell", "about", "give", "show", "need", "want", "get", "got", "from",
    "than", "also", "just", "am",
];

/// Whether `token` is a stopword.
pub fn is_stopword(token: &str) -> bool {
    STOPWORDS.contains(&token)
}

/// Tokenize and drop stopwords — the "content" tokens of a string.
pub fn content_tokens(text: &str) -> Vec<String> {
    tokenize(text)
        .into_iter()
        .filter(|t| !is_stopword(t))
        .collect()
}
