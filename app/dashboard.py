"""
dashboard.py — Plotly interactive evaluation dashboard.

Generates an interactive HTML dashboard from evaluation results.
Visualizes:
  - Per-category ROUGE-L comparison (bar chart)
  - Hallucination rate comparison (grouped bar)
  - Win rate pie chart
  - ROUGE-L scatter plot (base vs lora per question)
  - Term coverage radar chart

Usage:
    python app/dashboard.py --metrics outputs/eval_results/metrics_summary.json
    python app/dashboard.py  # Auto-detects latest results
"""

import argparse
import json
import sys
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def load_metrics(metrics_path: str = None) -> dict:
    """Load evaluation metrics, auto-detect if not specified."""
    if metrics_path:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Auto-detect
    candidates = sorted(Path("outputs/eval_results").glob("metrics_summary.json"))
    if not candidates:
        print("ERROR: No metrics found. Run 06_evaluate.py first.")
        sys.exit(1)

    latest = candidates[-1]
    print(f"Using metrics: {latest}")
    with open(latest, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dashboard(metrics: dict) -> go.Figure:
    """Build the full Plotly dashboard figure."""
    per_cat = metrics.get("per_category", {})
    overall = metrics.get("overall", {})

    categories = sorted(per_cat.keys())

    base_rl = [per_cat[c]["base_rouge_l"] for c in categories]
    lora_rl = [per_cat[c]["lora_rouge_l"] for c in categories]
    base_hall = [per_cat[c]["base_hallucination_rate"] for c in categories]
    lora_hall = [per_cat[c]["lora_hallucination_rate"] for c in categories]
    improvements = [per_cat[c]["rouge_l_improvement_pct"] for c in categories]

    # Friendly category names（来自领域配置）
    from app.domain_config import get_domain
    cat_names_map = {k: v.label for k, v in get_domain().intents.items()}
    cat_names_map["general"] = "综合"
    cat_names = [cat_names_map.get(c, c) for c in categories]

    # Create subplot grid: 2x3
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "ROUGE-L by Category (Base vs LoRA)",
            "Hallucination Rate by Category",
            "ROUGE-L Improvement % by Category",
            "Win/Loss/Tie Distribution",
            "Term Coverage Comparison",
            "Quick Summary",
        ],
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "pie"}],
            [{"type": "bar"}, {"type": "table"}],
        ],
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )

    # ── Chart 1: ROUGE-L bar chart ──
    fig.add_trace(
        go.Bar(name="Base Model", x=cat_names, y=base_rl,
               marker_color="#ff6b6b", text=[f"{v:.4f}" for v in base_rl],
               textposition="outside"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(name="LoRA Model", x=cat_names, y=lora_rl,
               marker_color="#51cf66", text=[f"{v:.4f}" for v in lora_rl],
               textposition="outside"),
        row=1, col=1,
    )

    # ── Chart 2: Hallucination rate ──
    fig.add_trace(
        go.Bar(name="Base Hallucination", x=cat_names, y=base_hall,
               marker_color="#ff6b6b", opacity=0.7),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(name="LoRA Hallucination", x=cat_names, y=lora_hall,
               marker_color="#51cf66", opacity=0.7),
        row=1, col=2,
    )

    # ── Chart 3: Improvement % ──
    colors = ["#51cf66" if v > 0 else "#ff6b6b" for v in improvements]
    fig.add_trace(
        go.Bar(x=cat_names, y=improvements, marker_color=colors,
               text=[f"{v:+.1f}%" for v in improvements],
               textposition="outside"),
        row=2, col=1,
    )

    # ── Chart 4: Win rate pie ──
    wins = overall.get("lora_wins", 0)
    losses = overall.get("base_wins", 0)
    ties = overall.get("ties", 0)
    fig.add_trace(
        go.Pie(
            labels=["LoRA Wins", "Base Wins", "Ties"],
            values=[wins, losses, ties],
            marker_colors=["#51cf66", "#ff6b6b", "#adb5bd"],
            textinfo="label+percent",
        ),
        row=2, col=2,
    )

    # ── Chart 5: Term coverage ──
    base_tc = [per_cat[c]["base_term_coverage"] for c in categories]
    lora_tc = [per_cat[c]["lora_term_coverage"] for c in categories]
    fig.add_trace(
        go.Bar(name="Base Coverage", x=cat_names, y=base_tc,
               marker_color="#ff6b6b", opacity=0.7),
        row=3, col=1,
    )
    fig.add_trace(
        go.Bar(name="LoRA Coverage", x=cat_names, y=lora_tc,
               marker_color="#51cf66", opacity=0.7),
        row=3, col=1,
    )

    # ── Chart 6: Summary table ──
    table_data = [
        ["ROUGE-L (Base)", f"{overall.get('base_rouge_l', 0):.4f}"],
        ["ROUGE-L (LoRA)", f"{overall.get('lora_rouge_l', 0):.4f}"],
        ["ROUGE-L Improvement", f"{overall.get('rouge_l_improvement_pct', 0):+.1f}%"],
        ["Hallucination (Base)", f"{overall.get('base_hallucination_rate', 0):.3f}"],
        ["Hallucination (LoRA)", f"{overall.get('lora_hallucination_rate', 0):.3f}"],
        ["Term Cov (Base)", f"{overall.get('base_term_coverage', 0):.3f}"],
        ["Term Cov (LoRA)", f"{overall.get('lora_term_coverage', 0):.3f}"],
        ["Win Rate", f"{overall.get('win_rate', 0):.1%}"],
        ["Total Questions", str(metrics.get("total_questions", "N/A"))],
    ]

    fig.add_trace(
        go.Table(
            header=dict(
                values=["Metric", "Value"],
                fill_color="#1a73e8",
                font=dict(color="white", size=12),
                align="left",
            ),
            cells=dict(
                values=list(zip(*table_data)),
                fill_color=[["#f8f9fa", "white"] * 10],
                font=dict(size=11),
                align="left",
            ),
        ),
        row=3, col=2,
    )

    # ── Layout ──
    fig.update_layout(
        title={
            "text": "⚖️ LoRA 法律问答微调 — 效果评估仪表盘",
            "x": 0.5,
            "font": {"size": 22, "color": "#1a73e8"},
        },
        height=1200,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        barmode="group",
    )

    # Axis labels
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_yaxes(title_text="ROUGE-L Score", row=1, col=1)
    fig.update_yaxes(title_text="Hallucination Rate", row=1, col=2)
    fig.update_yaxes(title_text="Improvement %", row=2, col=1)
    fig.update_yaxes(title_text="Term Coverage", row=3, col=1)

    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate Plotly evaluation dashboard")
    parser.add_argument("--metrics", type=str, default=None, help="Path to metrics_summary.json")
    parser.add_argument("--output", type=str, default="outputs/eval_results/evaluation_dashboard.html")
    args = parser.parse_args()

    metrics = load_metrics(args.metrics)

    print("Building dashboard...")
    fig = build_dashboard(metrics)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"✅ Dashboard saved to: {output_path}")

    # Also try to show it
    try:
        fig.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
