"""
visualisation.py
================
Generates three figures used in the project:
  1. RAG pipeline diagram        — our implementation (Always Retrieval)
  2. Results bar chart           — comparing all 3 strategies
  3. Original paper architecture — RetrievalQA (Zhang et al., ACL 2024)

Authors : Nishi Patel (501356244), Avi Patel (501376903)
Course  : Natural Language Processing — Final Project
Paper   : RetrievalQA (Zhang et al., ACL Findings 2024)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch


# ── 1. OUR RAG PIPELINE DIAGRAM ─────────────────────────────────────────────

def plot_pipeline(save_path='rag_pipeline.png'):
    """
    Generate and save the Always Retrieval pipeline diagram showing
    the five sequential steps of our implementation:
    User Question → Encode → Search → Augment → Generate → Evaluate.

    Args:
        save_path (str): File path to save the PNG image
    """
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    TEAL   = '#0D9488'
    NAVY   = '#0D1B2A'
    PURPLE = '#7C3AED'
    AMBER  = '#B45309'
    GREEN  = '#059669'
    WHITE  = '#FFFFFF'

    boxes = [
        ("User Question",    "",
         9.2, NAVY),
        ("Step 1: Encode",   "SentenceTransformers\nall-MiniLM-L6-v2 -> 384-dim vector",
         7.5, TEAL),
        ("Step 2: Search",   "FAISS IndexFlatIP\nCosine similarity -> top-3 passages",
         5.9, PURPLE),
        ("Step 3: Augment",  "Prompt Builder\nContext + Question -> enriched prompt",
         4.3, AMBER),
        ("Step 4: Generate", "google/flan-t5-base\nPrompt -> short answer",
         2.7, GREEN),
        ("Step 5: Evaluate", "Exact Match + Token F1\nPrediction vs ground truth",
         1.1, NAVY),
    ]

    for title, subtitle, yc, color in boxes:
        b = FancyBboxPatch(
            (2.5, yc - 0.55), 5, 1.0,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor=WHITE,
            linewidth=2, zorder=3
        )
        ax.add_patch(b)

        if subtitle:
            ax.text(5, yc - 0.05, title,
                    ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color=WHITE, zorder=4)
            ax.text(5, yc - 0.38, subtitle,
                    ha='center', va='center',
                    fontsize=8, color=WHITE,
                    alpha=0.9, zorder=4)
        else:
            ax.text(5, yc, title,
                    ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    color=WHITE, zorder=4)

    arrow_ys = [
        (8.75, 8.05),
        (7.00, 6.45),
        (5.45, 4.85),
        (3.85, 3.25),
        (2.25, 1.65),
    ]
    for y_start, y_end in arrow_ys:
        ax.annotate(
            '', xy=(5, y_end), xytext=(5, y_start),
            arrowprops=dict(
                arrowstyle='->', color=TEAL,
                lw=2.5, connectionstyle='arc3,rad=0'
            ),
            zorder=2
        )

    ax.text(5, 9.8, "Always Retrieval - RAG Pipeline",
            ha='center', va='center',
            fontsize=13, fontweight='bold', color=NAVY)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print("Pipeline diagram saved as " + save_path)


# ── 2. RESULTS BAR CHART ────────────────────────────────────────────────────

def plot_results(results_df, save_path='results_chart.png'):
    """
    Generate and save a side-by-side grouped bar chart comparing
    all three strategies on Overall results and Retrieval-Needed
    questions only.

    Args:
        results_df (pd.DataFrame): Scored results from evaluation.py
        save_path  (str)         : File path to save the PNG image
    """
    strategies = ['No Retrieval', 'Always Retrieval', 'Adaptive RAG']

    em_scores = [results_df['em_no_ret'].mean(),
                 results_df['em_always'].mean(),
                 results_df['em_adaptive'].mean()]
    f1_scores = [results_df['f1_no_ret'].mean(),
                 results_df['f1_always'].mean(),
                 results_df['f1_adaptive'].mean()]

    ret_df = results_df[results_df['needs_retrieval'] == 0]
    em_ret = [ret_df['em_no_ret'].mean(),
              ret_df['em_always'].mean(),
              ret_df['em_adaptive'].mean()]
    f1_ret = [ret_df['f1_no_ret'].mean(),
              ret_df['f1_always'].mean(),
              ret_df['f1_adaptive'].mean()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'RAG Strategy Comparison - RetrievalQA Benchmark\n'
        'Nishi Patel (501356244)  Avi Patel (501376903)',
        fontsize=13, fontweight='bold', color='#0D1B2A', y=1.02
    )

    x     = np.arange(len(strategies))
    width = 0.35
    c_em  = ['#94A3B8', '#0D9488', '#7C3AED']
    c_f1  = ['#CBD5E1', '#5DCAA5', '#AFA9EC']

    for ax, em, f1, title in [
        (axes[0], em_scores, f1_scores, 'Overall (100 questions)'),
        (axes[1], em_ret,    f1_ret,    'Retrieval-Needed Only (n=60)')
    ]:
        b1 = ax.bar(x - width/2, em, width, label='Exact Match',
                    color=c_em, edgecolor='white', linewidth=1.5)
        b2 = ax.bar(x + width/2, f1, width, label='Token F1',
                    color=c_f1, edgecolor='white', linewidth=1.5)

        for bar in b1:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.008,
                    '{:.3f}'.format(bar.get_height()),
                    ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='#0D1B2A')
        for bar in b2:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.008,
                    '{:.3f}'.format(bar.get_height()),
                    ha='center', va='bottom',
                    fontsize=9, color='#374151')

        ax.set_title(title, fontsize=12,
                     fontweight='bold', color='#0D1B2A', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, fontsize=10)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_ylim(0, 0.55)
        ax.legend(fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#FAFAFA')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print("Results chart saved as " + save_path)


# ── 3. ORIGINAL PAPER ARCHITECTURE DIAGRAM ──────────────────────────────────

def plot_paper_architecture(save_path='original_paper_architecture.png'):
    """
    Generate and save the original RetrievalQA paper architecture diagram.
    Split into three clearly labelled panels for easy reading:
      Part 1 - Dataset construction  (9,336 raw -> 2,785 final)
      Part 2 - ARAG evaluation       (3 strategies x 6 LLMs)
      Part 3 - Key findings          (what the paper discovered)

    Source: arxiv.org/abs/2402.16457

    Args:
        save_path (str): File path to save the PNG image
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 22))
    fig.patch.set_facecolor('white')
    fig.suptitle(
        "RetrievalQA - Original Paper Architecture\n"
        "Zhang, Fang & Chen  ACL Findings 2024  arxiv.org/abs/2402.16457",
        fontsize=14, fontweight='bold', color='#1E3A5F', y=0.99
    )

    TEAL   = '#0D9488'
    NAVY   = '#1E3A5F'
    GRAY   = '#6B7280'
    PURPLE = '#7C3AED'
    GREEN  = '#059669'
    AMBER  = '#B45309'
    BLUE   = '#0891B2'
    WHITE  = '#FFFFFF'
    DKGRAY = '#374151'

    def box(ax, x, y, w, h, color, title, sub=None, fontsize=9):
        """Draw a rounded rectangle with title and optional subtitle."""
        b = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08",
            facecolor=color, edgecolor=WHITE,
            linewidth=2, zorder=3
        )
        ax.add_patch(b)
        ty = y + h / 2 + (0.12 if sub else 0)
        ax.text(x + w / 2, ty, title,
                ha='center', va='center',
                fontsize=fontsize, fontweight='bold',
                color=WHITE, zorder=4)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.18, sub,
                    ha='center', va='center',
                    fontsize=fontsize - 1.5, color=WHITE,
                    alpha=0.88, zorder=4)

    def arr(ax, x1, y1, x2, y2, color=GRAY):
        """Draw a straight arrow from (x1,y1) to (x2,y2)."""
        ax.annotate(
            '', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle='->', color=color,
                lw=1.5, connectionstyle='arc3,rad=0'
            ),
            zorder=2
        )

    # ════════════════════════════════════════════════════════════════════
    # PART 1 — DATASET CONSTRUCTION
    # ════════════════════════════════════════════════════════════════════
    ax = axes[0]
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_facecolor('#F8FFFE')

    ax.text(0.15, 6.65, "Part 1 - Dataset construction pipeline",
            fontsize=12, fontweight='bold', color=TEAL, va='top')
    ax.text(0.15, 6.25,
            "9,336 raw questions from 5 sources  ->  GPT-4 filter  ->  2,785 final questions",
            fontsize=9, color=GRAY, va='top')

    # Five source boxes
    sources = [
        ("RealTimeQA", "397 Qs",  0.4),
        ("FreshQA",    "127 Qs",  2.0),
        ("ToolQA",     "100 Qs",  3.6),
        ("PopQA",      "1,399 Qs",5.2),
        ("TriviaQA",   "7,313 Qs",6.8),
    ]
    for name, count, x in sources:
        b = FancyBboxPatch((x, 4.6), 1.4, 0.9,
                            boxstyle="round,pad=0.06",
                            facecolor=DKGRAY, edgecolor=WHITE,
                            linewidth=1.5, zorder=3)
        ax.add_patch(b)
        ax.text(x + 0.7, 5.12, name,
                ha='center', va='center',
                fontsize=9, fontweight='bold', color=WHITE, zorder=4)
        ax.text(x + 0.7, 4.85, count,
                ha='center', va='center',
                fontsize=8, color=WHITE, alpha=0.85, zorder=4)
        arr(ax, x + 0.7, 4.6, x + 0.7, 4.2, GRAY)

    ax.text(9.2, 5.1,  "Total:",       fontsize=8.5, color=GRAY, va='center')
    ax.text(9.2, 4.78, "9,336 Qs", fontsize=10,  fontweight='bold',
            color=DKGRAY, va='center')

    # GPT-4 filter box
    box(ax, 3.0, 3.1, 6.5, 0.85, AMBER,
        "GPT-4 filtering  (closed-book setting)",
        "Keep only questions where Token F1 = 0  (GPT-4 fails without retrieval)")

    for x in [1.1, 2.7, 4.3, 5.9, 7.5]:
        arr(ax, x, 4.2, 6.25, 3.95, AMBER)

    # Two output paths from filter
    ax.annotate('', xy=(2.0, 2.4), xytext=(4.5, 3.1),
                arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.5), zorder=2)
    ax.annotate('', xy=(9.5, 2.4), xytext=(8.0, 3.1),
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5), zorder=2)

    box(ax, 0.4, 1.5, 3.8, 0.8, TEAL,
        "Retrieval-needed  (1,271 kept)",
        "New world + long-tail  -  must retrieve")
    box(ax, 8.2, 1.5, 3.8, 0.8, BLUE,
        "Parametric  (1,514 added)",
        "GPT-2 answers from memory  -  no retrieval")

    # Merge into final dataset
    ax.annotate('', xy=(6.4, 0.75), xytext=(2.3, 1.5),
                arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.5,
                                connectionstyle='arc3,rad=0'), zorder=2)
    ax.annotate('', xy=(8.0, 0.75), xytext=(10.1, 1.5),
                arrowprops=dict(arrowstyle='->', color='#185FA5', lw=1.5,
                                connectionstyle='arc3,rad=0'), zorder=2)

    box(ax, 5.2, 0.05, 4.0, 0.85, TEAL,
        "RetrievalQA benchmark  (2,785 total)",
        "1,271 retrieval-needed  +  1,514 parametric")

    # ════════════════════════════════════════════════════════════════════
    # PART 2 — ARAG EVALUATION FRAMEWORK
    # ════════════════════════════════════════════════════════════════════
    ax = axes[1]
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_facecolor('#FEFEFF')

    ax.text(0.15, 6.65, "Part 2 - ARAG evaluation framework",
            fontsize=12, fontweight='bold', color=PURPLE, va='top')
    ax.text(0.15, 6.25,
            "Every question is tested through 3 strategies  x  6 LLMs",
            fontsize=9, color=GRAY, va='top')

    # Input question
    box(ax, 5.5, 5.2, 4.5, 0.8, DKGRAY,
        "Short-form question  x",
        "new world or long-tail knowledge")

    # Three routing arrows to strategies
    arr(ax, 6.5,  5.2, 2.0,  4.5, DKGRAY)
    arr(ax, 7.75, 5.2, 7.75, 4.5, PURPLE)
    arr(ax, 9.0,  5.2, 13.5, 4.5, TEAL)

    # Three strategy boxes
    box(ax, 0.4,  3.5, 3.2, 0.8, DKGRAY,
        "No retrieval",
        "Answer from memory only")
    box(ax, 6.1,  3.5, 3.3, 0.8, PURPLE,
        "Adaptive retrieval",
        "LLM decides whether to retrieve")
    box(ax, 11.8, 3.5, 3.8, 0.8, TEAL,
        "Always retrieval",
        "Always fetch top-5 docs first")

    # Retriever feeds Adaptive and Always
    box(ax, 9.0, 5.2, 3.5, 0.8, BLUE,
        "Retriever",
        "Contriever / Google  -  top-5 docs")
    arr(ax, 10.75, 5.2, 13.7, 4.3, '#185FA5')
    arr(ax, 10.0,  5.2,  8.0, 4.3, '#185FA5')

    # Adaptive -> TA-ARE detail
    arr(ax, 7.75, 3.5, 7.75, 2.85, PURPLE)
    box(ax, 5.2, 1.95, 5.1, 0.8, '#5B21B6',
        "TA-ARE  (paper's proposed method)",
        "Today's date  +  2 Yes  +  2 No  in-context examples")
    ax.text(7.75, 1.76,
            "vs vanilla: 'Do you need retrieval? [Yes] / [No]'",
            ha='center', fontsize=8.5, color=PURPLE, style='italic')

    # Six LLM boxes
    ax.text(7.75, 1.36, "LLMs tested in the paper:",
            ha='center', fontsize=8.5, color=GRAY)

    models = [
        ("TinyLlama", "1.1B",        0.4),
        ("Phi-2",     "2.7B",        2.55),
        ("Llama-2",   "7B",          4.7),
        ("Self-RAG",  "7B fine-tune",6.85),
        ("GPT-3.5",   "OpenAI API",  9.0),
        ("GPT-4",     "250 samples", 11.15),
    ]
    for name, size, x in models:
        b = FancyBboxPatch((x, 0.2), 1.9, 0.75,
                            boxstyle="round,pad=0.05",
                            facecolor=DKGRAY, edgecolor=WHITE,
                            linewidth=1.5, zorder=3)
        ax.add_patch(b)
        ax.text(x + 0.95, 0.64, name,
                ha='center', va='center',
                fontsize=8.5, fontweight='bold', color=WHITE, zorder=4)
        ax.text(x + 0.95, 0.38, size,
                ha='center', va='center',
                fontsize=7.5, color=WHITE, alpha=0.85, zorder=4)

    # ════════════════════════════════════════════════════════════════════
    # PART 3 — KEY FINDINGS
    # ════════════════════════════════════════════════════════════════════
    ax = axes[2]
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_facecolor('#F8FFF8')

    ax.text(0.15, 4.65, "Part 3 - Key findings",
            fontsize=12, fontweight='bold', color=GREEN, va='top')
    ax.text(0.15, 4.25,
            "What the paper discovered after running all experiments",
            fontsize=9, color=GRAY, va='top')

    # Three finding boxes
    box(ax, 0.4,  2.4, 4.5, 1.4, TEAL,
        "Finding 1 - Always Retrieval wins",
        "Always Ret. >= Adaptive >= No Ret.  "
        "Retrieval consistently helps on hard questions")
    box(ax, 5.8,  2.4, 4.5, 1.4, PURPLE,
        "Finding 2 - TA-ARE beats vanilla",
        "avg +14.9% retrieval accuracy  "
        "across all LLMs vs vanilla prompting")
    box(ax, 11.2, 2.4, 4.4, 1.4, AMBER,
        "Finding 3 - Main claim",
        "GPT-3.5 fails to retrieve >50% of the time  "
        "LLMs cannot judge their own knowledge gaps")

    # Arrows down to conclusion
    arr(ax, 2.65, 2.4,  6.5, 1.5, GRAY)
    arr(ax, 8.05, 2.4,  8.05, 1.5, GRAY)
    arr(ax, 13.4, 2.4,  9.6, 1.5, GRAY)

    # Overall conclusion box
    box(ax, 4.5, 0.3, 7.0, 1.1, '#3B6D11',
        "Overall conclusion",
        "Vanilla prompting fails adaptive RAG  -  "
        "TA-ARE with date awareness and ICL examples fixes it")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print("Original paper architecture saved as " + save_path)