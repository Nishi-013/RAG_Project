import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


# Global model variables — set when load_model() is called
_tokenizer = None
_model     = None


def load_model(model_name='google/flan-t5-base'):
    
    global _tokenizer, _model

    print(f"Loading language model: {model_name}...")

    # Tokenizer converts text strings into token IDs the model reads
    _tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Load the model weights
    _model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Set to evaluation mode — disables dropout, no gradients needed
    _model.eval()

    print(f"Model loaded successfully!")
    return _tokenizer, _model


def generate_answer(prompt, max_new_tokens=50):
    
    if _tokenizer is None or _model is None:
        raise RuntimeError("Call load_model() before generate_answer().")

    # Step 1 — Tokenize: convert text to numerical token IDs
    inputs = _tokenizer(
        prompt,
        return_tensors="pt",   # return PyTorch tensors
        truncation=True,        # cut off if prompt exceeds max length
        max_length=512          # flan-t5-base maximum input length
    )

    # Step 2 — Generate: produce output token IDs
    # torch.no_grad() disables gradient tracking 
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

    # Step 3 — Decode: convert token IDs back to a text string
    answer = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()
