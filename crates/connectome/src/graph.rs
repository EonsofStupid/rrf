//! The serializable graph model the UI renders.

use serde::{Deserialize, Serialize};

/// What a node represents in the map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    /// The user's query — the root of the map.
    Query,
    /// A pipeline stage (embedder, recall, reranker, classifier).
    Stage,
    /// A retrieved/ranked candidate.
    Candidate,
    /// The reason-ready verdict.
    Readiness,
}

/// What a relationship represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    /// Flow of control from one stage to the next.
    Flow,
    /// A candidate ranked by a stage; `weight` carries the score.
    Ranked,
    /// A stage producing the verdict.
    Verdict,
}

/// A node in the connectome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Stable node id (unique within a graph).
    pub id: String,
    /// What this node is.
    pub kind: NodeKind,
    /// Display label.
    pub label: String,
    /// Optional score/strength for rendering (size, colour, opacity).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
}

/// A directed, weighted edge in the connectome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Source node id.
    pub from: String,
    /// Target node id.
    pub to: String,
    /// What the edge means.
    pub kind: EdgeKind,
    /// Strength for rendering (edge thickness / distance).
    pub weight: f32,
}

/// A full renderable map of one flow pass.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConnectomeGraph {
    /// All nodes.
    pub nodes: Vec<Node>,
    /// All edges.
    pub edges: Vec<Edge>,
}

impl ConnectomeGraph {
    /// An empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node.
    pub fn node(
        &mut self,
        id: impl Into<String>,
        kind: NodeKind,
        label: impl Into<String>,
        score: Option<f32>,
    ) {
        self.nodes.push(Node {
            id: id.into(),
            kind,
            label: label.into(),
            score,
        });
    }

    /// Add an edge.
    pub fn edge(
        &mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        kind: EdgeKind,
        weight: f32,
    ) {
        self.edges.push(Edge {
            from: from.into(),
            to: to.into(),
            kind,
            weight,
        });
    }

    /// Serialize to pretty JSON for the UI.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Render to Graphviz DOT for a quick visual sanity check.
    pub fn to_dot(&self) -> String {
        let mut s = String::from("digraph connectome {\n  rankdir=LR;\n  node [style=filled];\n");
        for n in &self.nodes {
            let color = match n.kind {
                NodeKind::Query => "gold",
                NodeKind::Stage => "lightblue",
                NodeKind::Candidate => "palegreen",
                NodeKind::Readiness => "salmon",
            };
            let label = n.label.replace('"', "'");
            s.push_str(&format!(
                "  \"{}\" [label=\"{}\", fillcolor={}];\n",
                n.id, label, color
            ));
        }
        for e in &self.edges {
            s.push_str(&format!(
                "  \"{}\" -> \"{}\" [label=\"{:.2}\"];\n",
                e.from, e.to, e.weight
            ));
        }
        s.push_str("}\n");
        s
    }
}
