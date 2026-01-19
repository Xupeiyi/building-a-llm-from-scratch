from typing import Optional

import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from gpt.tokenize import text_to_token_ids, token_ids_to_text
from gpt.model import GPTModel, generate_text_simple



def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    logits = model(input_batch)
    logits_flat = logits.flatten(0, 1)

    target_batch = target_batch.to(device)
    target_flat = target_batch.flatten()

    loss = torch.nn.functional.cross_entropy(logits_flat, target_flat)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches: Optional[int] = None):
    """Calculates the average loss of the first n batches in the data loader."""
    if len(data_loader) == 0:
        return float("nan")
    
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    total_loss = 0.
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(
    model, 
    train_loader, validation_loader, 
    device, 
    num_batches: int
):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches)
        validation_loss = calc_loss_loader(validation_loader, model, device, num_batches)
    model.train()
    return train_loss, validation_loss


def generate_and_print_sample(model: GPTModel, tokenizer, device, start_tokens: str):
    model.eval()
    context_size = model.position_embedding_layer.weight.shape[0]
    token_ids = text_to_token_ids(start_tokens, tokenizer).to(device)
    with torch.no_grad():
        generated_token_ids = generate_text_simple(
            model, token_ids, num_new_tokens=50, context_size=context_size
        )
    generated_tokens = token_ids_to_text(generated_token_ids, tokenizer)
    print(generated_tokens.replace("\n", " "))
    model.train()


def train_model_simple(
    model: GPTModel,
    train_loader, validation_loader, 
    optimizer, device, num_epochs, 
    eval_freq: int, eval_num_batches: int, 
    start_tokens: str, tokenizer
):
    (
        train_losses, 
        validation_losses, 
        num_tokens_seen_list
    ) = [], [], []
    num_tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            # Reset loss gradients from the previous batch iteration
            optimizer.zero_grad()  
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            
            num_tokens_seen += input_batch.numel()
            global_step += 1

            # Evaluate model at specified frequency
            if global_step % eval_freq == 0:
                train_loss, validation_loss = evaluate_model(
                    model, 
                    train_loader, 
                    validation_loader, 
                    device, 
                    eval_num_batches
                )
                train_losses.append(train_loss)
                validation_losses.append(validation_loss)
                num_tokens_seen_list.append(num_tokens_seen)
                print(
                    f"Ep {epoch + 1} (Step {global_step: 06d}): "
                    f"Train loss {train_loss:.3f}, Validation loss {validation_loss:.3f}"
                )

        generate_and_print_sample(model, tokenizer, device, start_tokens)

    return train_losses, validation_losses, num_tokens_seen_list


def generate(
    model, 
    token_ids, 
    num_new_tokens, 
    context_size, 
    temperature=0.0, 
    top_k=None, 
    eos_id=None
):
    for _ in range(num_new_tokens):
        token_ids_in_context = token_ids[:, -context_size:]
        
        with torch.no_grad():
            logits = model(token_ids_in_context)
        logits = logits[:, -1, :]
        
        if top_k is not None:
            top_k_logits, _ = torch.topk(logits, top_k)
            kth_logit = top_k_logits[:, -1]
            logits = torch.where(
                logits < kth_logit, 
                torch.tensor(float("-inf")).to(logits.device), 
                logits
            )

        if temperature > 0.0:
            logits = logits / temperature
            probabilities = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probabilities, num_samples=1)
        else:
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
        
        if next_token_id == eos_id:
            break
        
        token_ids = torch.cat((token_ids, next_token_id), dim=1)
        
    return token_ids


def plot_losses(epochs_seen, tokens_seen, train_losses, validation_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
        epochs_seen, 
        validation_losses, 
        linestyle="-.", 
        label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()