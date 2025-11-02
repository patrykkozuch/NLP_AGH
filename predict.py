import torch

from transformer.dataset import prepare_mask


def complete_sentence(model, input_ids, attention_mask, tokenizer, max_new_tokens=128, device='cuda'):
    """
    Complete a sentence by generating tokens autoregressively.
    """
    model.eval()

    # Clone input to avoid modifying original
    current_ids = input_ids.clone()
    current_mask = attention_mask.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get prediction for the entire sequence
            mask = prepare_mask(current_mask).to(device)
            output = model(current_ids, mask)  # (batch, seq_len, vocab_size)

            # Get logits for the last real token position
            real_positions = (current_mask == 1).long()
            last_real_idx = real_positions.sum(dim=1) - 1  # (batch_size,)
            batch_indices = torch.arange(current_ids.size(0), device=device)
            next_token_logits = output[batch_indices, last_real_idx]  # (batch_size, vocab_size)

            # Sample next token
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)

            # Check for end-of-sequence
            if (next_token == tokenizer.eos_token_id).all():
                break

            # Simply append the new token to the sequence
            current_ids = torch.cat([current_ids, next_token], dim=1)
            current_mask = torch.cat([current_mask, torch.ones_like(next_token)], dim=1)

    # Decode to text
    completed_text = tokenizer.batch_decode(current_ids, skip_special_tokens=True)
    return current_ids, completed_text