import json
import pandas as pd
from datasets import load_dataset


def load_retrievalqa_dataset():
    """
    Load the RetrievalQA benchmark from HuggingFace.
    The dataset contains 2,785 short-form QA questions with
    pre-retrieved context passages and ground-truth answers.

    """
    dataset = load_dataset("hyintell/RetrievalQA")

    df = dataset['train'].to_pandas()

    print(f"Dataset loaded successfully!")
    print(f"  Total questions : {len(df)}")
    print(f"  Columns         : {df.columns.tolist()}")
    return df


def explore_dataset(df):
    # Split by param_knowledge_answerable label
    # 0 = needs retrieval, 1 = answerable from LLM memory
    needs_retrieval = df[df['param_knowledge_answerable'] == 0]
    parametric_only = df[df['param_knowledge_answerable'] == 1]

    print("=" * 45)
    print(f"Questions needing retrieval   : {len(needs_retrieval)}")
    print(f"Questions answerable in memory: {len(parametric_only)}")
    print(f"Total                         : {len(df)}")
    print("=" * 45)
    print("\nBreakdown by data source:")
    print(df['data_source'].value_counts())

    return needs_retrieval, parametric_only


def show_examples(df, indices=[10, 100, 500]):
    for idx in indices:
        row = df.iloc[idx]
        print("=" * 55)
        print(f"Question       : {row['question']}")
        print(f"Correct answer : {row['ground_truth']}")
        label = 'YES <- must retrieve' if row['param_knowledge_answerable'] == 0 \
                else 'NO  <- AI knows this'
        print(f"Needs retrieval: {label}")
        print(f"Data source    : {row['data_source']}")

        # Context is stored as JSON strings — parse the first one
        context_list = list(row['context'])
        if len(context_list) > 0:
            ctx = json.loads(context_list[0])
            print(f"Context title  : {ctx.get('title', '(no title)')}")
            print(f"Context text   : {ctx.get('text', '')[:180]}...")
            print(f"Relevance score: {ctx.get('score', 'N/A')}")
        else:
            print("Context        : (no context available)")
        print()


def create_sample(needs_retrieval, parametric_only,
                  n_retrieval=60, n_parametric=40, random_state=42):
    sample_ret   = needs_retrieval.sample(n_retrieval,  random_state=random_state)
    sample_param = parametric_only.sample(n_parametric, random_state=random_state)

    # Combine and shuffle
    sample_df = pd.concat([sample_ret, sample_param])
    sample_df = sample_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"Sample created!")
    print(f"  Total            : {len(sample_df)}")
    print(f"  Needs retrieval  : {len(sample_df[sample_df['param_knowledge_answerable']==0])}")
    print(f"  Parametric only  : {len(sample_df[sample_df['param_knowledge_answerable']==1])}")
    print(f"\nBreakdown by data source:")
    print(sample_df['data_source'].value_counts())

    return sample_df
