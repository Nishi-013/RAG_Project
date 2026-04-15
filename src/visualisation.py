import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def plot_pipeline(save_path='rag_pipeline.png'):
   
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Color palette
    TEAL   = '#0D9488'
    NAVY   = '#0D1B2A'
    PURPLE = '#7C3AED'
    AMBER  = '#B45309'
    GREEN  = '#059669'
    WHITE  = '#FFFFFF'

    # Box definitions: (title, subtitle, y_centre, color)
    boxes = [
        ("User Question",    "",
         9.2, NAVY),
        ("Step 1: Encode",   "SentenceTransformers\nall-MiniLM-L6-v2 → 384-dim vector",
         7.5, TEAL),
        ("Step 2: Search",   "FAISS IndexFlatIP\nCosine similarity → top-3 passages",
         5.9, PURPLE),
        ("Step 3: Augment",  "Prompt Builder\nContext + Question → enriched prompt",
         4.3, AMBER),
        ("Step 4: Generate", "google/flan-t5-base\nPrompt → short answer",
         2.7, GREEN),
        ("Step 5: Evaluate", "Exact Match + Token F1\nPrediction vs ground truth",
         1.1, NAVY),
    ]

    for title, subtitle, yc, color in boxes:
        box = FancyBboxPatch(
            (2.5, yc - 0.55), 5, 1.0,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor=WHITE,
            linewidth=2, zorder=3
        )
        ax.add_patch(box)

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

    # Arrows between boxes
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

    ax.text(5, 9.8, "Always Retrieval — RAG Pipeline",
            ha='center', va='center',
            fontsize=13, fontweight='bold', color=NAVY)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Pipeline diagram saved as {save_path}")


def plot_results(results_df, save_path='results_chart.png'):
    
    strategies = ['No Retrieval', 'Always Retrieval', 'Adaptive RAG']

    # Overall scores
    em_scores = [results_df['em_no_ret'].mean(),
                 results_df['em_always'].mean(),
                 results_df['em_adaptive'].mean()]
    f1_scores = [results_df['f1_no_ret'].mean(),
                 results_df['f1_always'].mean(),
                 results_df['f1_adaptive'].mean()]

    # Retrieval-needed only scores
    ret_df = results_df[results_df['needs_retrieval'] == 0]
    em_ret = [ret_df['em_no_ret'].mean(),
              ret_df['em_always'].mean(),
              ret_df['em_adaptive'].mean()]
    f1_ret = [ret_df['f1_no_ret'].mean(),
              ret_df['f1_always'].mean(),
              ret_df['f1_adaptive'].mean()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'RAG Strategy Comparison — RetrievalQA Benchmark\n',
        fontsize=13, fontweight='bold', color='#0D1B2A', y=1.02
    )

    x      = np.arange(len(strategies))
    width  = 0.35
    c_em   = ['#94A3B8', '#0D9488', '#7C3AED']
    c_f1   = ['#CBD5E1', '#5DCAA5', '#AFA9EC']

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
                    f'{bar.get_height():.3f}',
                    ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='#0D1B2A')
        for bar in b2:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.008,
                    f'{bar.get_height():.3f}',
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
    print(f"Results chart saved as {save_path}")
