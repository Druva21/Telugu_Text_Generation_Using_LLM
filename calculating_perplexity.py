

    # if isinstance(model, PreTrainedModel):
    #     encodings = tokenizer(text, return_tensors='pt')
    #     input_ids = encodings.input_ids.to(device)
    #     with torch.no_grad():
    #         outputs = model(input_ids, labels=input_ids)
    #         loss = outputs.loss.item()
    #     return math.exp(loss)



import torch
import math
from transformers import PreTrainedModel

def compute_perplexity(model, tokenizer, text, device='cpu'):
    """
    Compute perplexity for HuggingFace or SmallGPT model.
    """

    model.eval()
    model.to(device)

    # HuggingFace model
    if isinstance(model, PreTrainedModel):
        encodings = tokenizer(text, return_tensors='pt')
        input_ids = encodings.input_ids.to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
        return math.exp(loss)

    # SmallGPT model
    else:
        # Encode text
        if hasattr(tokenizer, "encode"):
            encoded = tokenizer.encode(text)
            input_ids = encoded.ids
        else:
            input_ids = tokenizer.encode(text)

        # Use actual model attributes
        vocab_size = model.token_emb.num_embeddings
        seq_len = model.pos_emb.num_embeddings

        # Remove out-of-vocab tokens
        input_ids = [idx for idx in input_ids if 0 <= idx < vocab_size]

        # Truncate sequence to model's max sequence length
        input_ids = input_ids[-seq_len:] if len(input_ids) > seq_len else input_ids

        if len(input_ids) < 2:
            raise ValueError("Not enough valid tokens to compute perplexity.")

        input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

        # Compute negative log-likelihood
        nll = 0.0
        with torch.no_grad():
            for i in range(1, input_ids.size(1)):
                seq = input_ids[:, :i]       
                target = input_ids[:, i]     

                logits = model(seq)          
                logits_last = logits[:, -1, :]  

                log_probs = torch.log_softmax(logits_last, dim=-1)
                nll -= log_probs[0, target].item()

        avg_nll = nll / (input_ids.size(1) - 1)
        perplexity = math.exp(avg_nll)
        return perplexity
