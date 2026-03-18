import evaluate
import torch
from tqdm import tqdm


def model_eval_metrics(model, loader, tokenizer, device):
    rouge = evaluate.load("rouge")
    total_rouge1 = 0
    total_rouge2 = 0
    model.eval()
    with torch.no_grad():
        for beginings, endings in tqdm(loader):
            batch_generated_sequences = []
            for begining in beginings:
                begining_tokenized = tokenizer(
                    begining, return_tensors="pt").to(device)
                max_new_tokens = max(int(0.333*len(begining.split())),1)
                output = model.generate(max_new_tokens = max_new_tokens, 
                                        **begining_tokenized,
                                         pad_token_id=tokenizer.eos_token_id ).cpu()
                batch_generated_sequences.append(tokenizer.decode(
                    output[0], skip_special_tokens=True))
            rouge_metrics = rouge.compute(predictions=batch_generated_sequences,
                                          references=endings)
            total_rouge1 += rouge_metrics["rouge1"]
            total_rouge2 += rouge_metrics["rouge2"]

    return total_rouge1/len(loader), total_rouge2/len(loader)
