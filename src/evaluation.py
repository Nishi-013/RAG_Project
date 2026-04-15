import re
import json
import string
import pandas as pd


# ── Evaluation Metric Functions ──────────────────────────────────────

def normalize(text):

    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text.strip()


def exact_match(prediction, gold_list):
   
    pred_norm = normalize(prediction)
    return int(any(normalize(g) == pred_norm for g in gold_list))


def token_f1(prediction, gold_list):
    
    pred_tokens = normalize(prediction).split()
    best_f1     = 0.0

    for gold in gold_list:
        gold_tokens = normalize(gold).split()

        # Words appearing in both prediction and gold answer
        common = set(pred_tokens) & set(gold_tokens)

        if not common:
            continue   # no overlap -> F1 = 0 for this gold answer

        precision = len(common) / len(pred_tokens)
        recall    = len(common) / len(gold_tokens)
        f1        = 2 * precision * recall / (precision + recall)
        best_f1   = max(best_f1, f1)

    return best_f1


# ── Experiment Runner ─────────────────────────────────────────────────

def run_experiments(sample_df, generate_answer_fn,
                    build_no_ret_fn, build_always_fn, build_adaptive_fn):
    
    results = []

    print(f"Running experiments on {len(sample_df)} questions...")

    for _, row in sample_df.iterrows():

        question        = row['question']
        ground_truth    = list(row['ground_truth'])
        needs_retrieval = row['param_knowledge_answerable']
        data_source     = row['data_source']

        # Extract top-3 pre-retrieved context passages
        # These are stored as JSON strings in the dataset's context field
        context_list = []
        for ctx_str in list(row['context'])[:3]:
            try:
                ctx  = json.loads(ctx_str)
                text = ctx.get('text', '').strip()
                if text:
                    context_list.append(text)
            except Exception:
                continue

        # Strategy 1 — No Retrieval
        pred_no_ret = generate_answer_fn(
            build_no_ret_fn(question)
        )

        # Strategy 2 — Always Retrieval
        pred_always = generate_answer_fn(
            build_always_fn(question, context_list)
        )

        # Strategy 3 — Adaptive Retrieval
        pred_adaptive = generate_answer_fn(
            build_adaptive_fn(question, context_list, needs_retrieval)
        )

        results.append({
            'question'        : question,
            'ground_truth'    : ground_truth,
            'needs_retrieval' : needs_retrieval,
            'data_source'     : data_source,
            'pred_no_ret'     : pred_no_ret,
            'pred_always'     : pred_always,
            'pred_adaptive'   : pred_adaptive,
        })

        if len(results) % 10 == 0:
            print(f"  Progress: {len(results)}/{len(sample_df)} done...")

    print(f"\nAll {len(results)} questions completed!")
    return results


# ── Scoring Functions ─────────────────────────────────────────────────

def compute_scores(results):
    
    # Add EM and F1 columns for each strategy
    for r in results:
        r['em_no_ret']   = exact_match(r['pred_no_ret'],   r['ground_truth'])
        r['em_always']   = exact_match(r['pred_always'],   r['ground_truth'])
        r['em_adaptive'] = exact_match(r['pred_adaptive'], r['ground_truth'])

        r['f1_no_ret']   = token_f1(r['pred_no_ret'],   r['ground_truth'])
        r['f1_always']   = token_f1(r['pred_always'],   r['ground_truth'])
        r['f1_adaptive'] = token_f1(r['pred_adaptive'], r['ground_truth'])

    results_df = pd.DataFrame(results)

    # Overall results
    print("=" * 55)
    print("OVERALL RESULTS (100 questions)")
    print("=" * 55)
    print(f"{'Strategy':<20} {'Exact Match':>12} {'Token F1':>10}")
    print("-" * 45)
    for col, label in [('no_ret','No Retrieval'),
                        ('always','Always Retrieval'),
                        ('adaptive','Adaptive RAG')]:
        print(f"{label:<20} "
              f"{results_df[f'em_{col}'].mean():>12.3f} "
              f"{results_df[f'f1_{col}'].mean():>10.3f}")

    # Split by retrieval category
    ret_df  = results_df[results_df['needs_retrieval'] == 0]
    par_df  = results_df[results_df['needs_retrieval'] == 1]

    for subset, label in [(ret_df, f"RETRIEVAL-NEEDED (n={len(ret_df)})"),
                           (par_df, f"PARAMETRIC ONLY  (n={len(par_df)})")]:
        print(f"\n{'='*55}")
        print(label)
        print("=" * 55)
        print(f"{'Strategy':<20} {'Exact Match':>12} {'Token F1':>10}")
        print("-" * 45)
        for col, name in [('no_ret','No Retrieval'),
                           ('always','Always Retrieval'),
                           ('adaptive','Adaptive RAG')]:
            print(f"{name:<20} "
                  f"{subset[f'em_{col}'].mean():>12.3f} "
                  f"{subset[f'f1_{col}'].mean():>10.3f}")

    return results_df


# ── Error Analysis ────────────────────────────────────────────────────

def error_analysis(results_df):
    
    print("ERROR ANALYSIS — 5 Selected Examples")
    print("=" * 60)

    # Case 1 — Where RAG improved the answer
    rag_helps = results_df[
        (results_df['em_no_ret'] == 0) &
        (results_df['em_always'] == 1)
    ].head(3)

    print("\n[CASE 1] RAG HELPED — No Retrieval wrong, Always RAG correct")
    print("-" * 60)
    for _, r in rag_helps.iterrows():
        print(f"Question     : {r['question']}")
        print(f"Gold answer  : {r['ground_truth']}")
        print(f"No Retrieval : '{r['pred_no_ret']}'  WRONG")
        print(f"Always RAG   : '{r['pred_always']}'  CORRECT")
        print(f"Data source  : {r['data_source']}")
        print()

    # Case 2 — Where RAG did not help (retriever failure)
    rag_fails = results_df[
        (results_df['em_no_ret']   == 0) &
        (results_df['em_always']   == 0) &
        (results_df['needs_retrieval'] == 0)
    ].head(2)

    print("\n[CASE 2] RAG FAILED — Both strategies wrong")
    print("(Retriever failure or context does not contain the answer)")
    print("-" * 60)
    for _, r in rag_fails.iterrows():
        print(f"Question     : {r['question']}")
        print(f"Gold answer  : {r['ground_truth']}")
        print(f"No Retrieval : '{r['pred_no_ret']}'  WRONG")
        print(f"Always RAG   : '{r['pred_always']}'  WRONG")
        print(f"Data source  : {r['data_source']}")
        print()

    # Summary counts
    print("=" * 60)
    print("Summary:")
    print(f"  RAG helped (wrong->correct) : "
          f"{len(results_df[(results_df['em_no_ret']==0)&(results_df['em_always']==1)])}")
    print(f"  RAG hurt   (correct->wrong) : "
          f"{len(results_df[(results_df['em_no_ret']==1)&(results_df['em_always']==0)])}")
    print(f"  Both correct               : "
          f"{len(results_df[(results_df['em_no_ret']==1)&(results_df['em_always']==1)])}")
    print(f"  Both wrong                 : "
          f"{len(results_df[(results_df['em_no_ret']==0)&(results_df['em_always']==0)])}")
