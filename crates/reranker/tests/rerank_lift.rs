//! The reranker gate (docs/MODELS.md §4): a reranker must **lift** top-k
//! relevance versus not reranking. If it doesn't, that is a real finding and
//! this test says so rather than being quietly dropped.
//!
//! `#[ignore]` — needs live servers:
//!
//! ```sh
//! RRO_TEST_RERANK_LLAMACPP=http://127.0.0.1:8093/v1/rerank \
//! RRO_TEST_RERANK_VLLM=http://127.0.0.1:8092/rerank \
//!   cargo test -p reranker --test rerank_lift -- --ignored --nocapture
//! ```

use rro_core::{Candidate, Reranker};
use reranker::{HttpRerankConfig, HttpRerankKind, HttpReranker, LexicalReranker};

/// A small golden set: each query has exactly one correct document among
/// distractors that share vocabulary with it. Lexical overlap alone should
/// struggle; a cross-encoder should not.
struct Case {
    query: &'static str,
    gold: &'static str,
    docs: &'static [&'static str],
}

const CASES: &[Case] = &[
    Case {
        query: "What is the capital of China?",
        gold: "beijing",
        docs: &[
            // Distractors deliberately loaded with query words.
            "China is a large country. This document is about the capital markets of Asia.",
            "The capital of France is Paris, not China.",
            "What is a capital? A capital is the seat of government of a country.",
            "The capital of China is Beijing.", // gold, last on purpose
        ],
    },
    Case {
        query: "How do plants make food from sunlight?",
        gold: "photosynthesis",
        docs: &[
            "Plants need food and sunlight to grow well in a garden.",
            "Sunlight is made of photons. Food is made in kitchens.",
            "Photosynthesis is the process by which plants convert light energy into chemical \
             energy, producing glucose from carbon dioxide and water.",
            "How do you make plant food at home from kitchen scraps?",
        ],
    },
];

fn gold_index(c: &Case) -> usize {
    match c.gold {
        "beijing" => 3,
        "photosynthesis" => 2,
        _ => unreachable!(),
    }
}

fn candidates(c: &Case) -> Vec<Candidate> {
    c.docs
        .iter()
        .enumerate()
        .map(|(i, t)| Candidate::new(format!("d{i}"), *t, 0.0))
        .collect()
}

/// golden@1: did the correct document end up first?
async fn golden_at_1(r: &dyn Reranker) -> f32 {
    let mut hits = 0.0;
    for c in CASES {
        let out = r.rerank(c.query, candidates(c), 4).await.unwrap();
        let want = format!("d{}", gold_index(c));
        let got = out[0].id.as_str();
        println!("    q={:?}\n      top1={:?}", c.query, out[0].text);
        if got == want {
            hits += 1.0;
        }
    }
    hits / CASES.len() as f32
}

async fn http(var: &str, kind: HttpRerankKind) -> Option<HttpReranker> {
    let ep = std::env::var(var).ok().filter(|s| !s.trim().is_empty())?;
    match HttpReranker::connect(HttpRerankConfig::new(&ep, kind)).await {
        Ok(r) => Some(r),
        // Set-but-broken must fail, never skip.
        Err(e) => panic!("{var}={ep} is set but connecting failed: {e}"),
    }
}

/// The BM25 floor. Recorded, not asserted: it is the baseline the cross-encoders
/// must beat, and its value is a fact about the corpus, not a pass/fail.
#[tokio::test]
async fn lexical_baseline_golden_at_1() {
    let r = LexicalReranker::new();
    println!("  lexical (BM25):");
    let score = golden_at_1(&r).await;
    println!("  => BM25 golden@1 = {score:.2}");
}

#[tokio::test]
#[ignore]
async fn llamacpp_reranker_lifts_over_bm25() {
    let Some(r) = http("RRO_TEST_RERANK_LLAMACPP", HttpRerankKind::LlamaCpp).await else {
        eprintln!("SKIP: set RRO_TEST_RERANK_LLAMACPP");
        return;
    };
    let bm25 = golden_at_1(&LexicalReranker::new()).await;
    println!("  llamacpp ({}):", r.model_name());
    let ce = golden_at_1(&r).await;
    println!("  => BM25 {bm25:.2} -> llamacpp {ce:.2}");
    assert!(
        ce >= bm25,
        "the cross-encoder ({ce}) did WORSE than BM25 ({bm25}) — that is a real finding, \
         not a flaky test: report it rather than deleting this assertion"
    );
    assert!(ce >= 1.0, "cross-encoder should rank every gold first, got {ce}");
}

#[tokio::test]
#[ignore]
async fn vllm_reranker_lifts_over_bm25() {
    let Some(r) = http("RRO_TEST_RERANK_VLLM", HttpRerankKind::Vllm).await else {
        eprintln!("SKIP: set RRO_TEST_RERANK_VLLM");
        return;
    };
    let bm25 = golden_at_1(&LexicalReranker::new()).await;
    println!("  vllm ({}):", r.model_name());
    let ce = golden_at_1(&r).await;
    println!("  => BM25 {bm25:.2} -> vllm {ce:.2}");
    assert!(ce >= bm25, "vLLM cross-encoder ({ce}) did worse than BM25 ({bm25})");
    assert!(ce >= 1.0, "cross-encoder should rank every gold first, got {ce}");
}

/// Both engines serve the same model (llama-nemotron-rerank-1b-v2) on this box,
/// so they must agree on the ORDER even though their score scales differ wildly
/// (vLLM normalizes to [0,1]; llama.cpp returns raw logits like 18.68 / -11.89).
#[tokio::test]
#[ignore]
async fn llamacpp_and_vllm_agree_on_order() {
    let (Some(l), Some(v)) = (
        http("RRO_TEST_RERANK_LLAMACPP", HttpRerankKind::LlamaCpp).await,
        http("RRO_TEST_RERANK_VLLM", HttpRerankKind::Vllm).await,
    ) else {
        eprintln!("SKIP: set both RRO_TEST_RERANK_LLAMACPP and RRO_TEST_RERANK_VLLM");
        return;
    };
    for c in CASES {
        let lo = l.rerank(c.query, candidates(c), 4).await.unwrap();
        let vo = v.rerank(c.query, candidates(c), 4).await.unwrap();
        let lord: Vec<&str> = lo.iter().map(|c| c.id.as_str()).collect();
        let vord: Vec<&str> = vo.iter().map(|c| c.id.as_str()).collect();
        println!("  q={:?}\n    llamacpp={lord:?}\n    vllm    ={vord:?}", c.query);
        assert_eq!(
            lord[0], vord[0],
            "same model, same query, but the engines disagree on the top document"
        );
    }
}
