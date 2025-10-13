import re
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader


def replace_person_ids(text: str, p2n: dict) -> str:
    def repl(match):
        key = match.group(0)  # 'person_123' 같은 전체 문자열
        return p2n.get(key, key)  # 없으면 그대로 유지

    return re.sub(r'person_\d+', repl, text)

class PromptDataset(Dataset):
    def __init__(self, data_list, tokenizer: AutoTokenizer, max_length=128):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        prompt_text = sample['input']
        full_input = sample['init_text']

        encoding = self.tokenizer(full_input,
                                  max_length=self.max_length,
                                  padding='max_length',
                                  truncation=True,
                                  return_tensors='pt')

        input_ids = encoding.input_ids.squeeze(0)
        attention_mask = encoding.attention_mask.squeeze(0)
        labels = input_ids.clone()

        seq_len = attention_mask.sum().item()
        prompt_len = len(self.tokenizer.encode(prompt_text, add_special_tokens=False))

        content_start = len(input_ids) - seq_len
        prompt_end = content_start + prompt_len

        labels[: prompt_end] = -100
        labels[attention_mask == 0] = -100
        labels[labels == self.tokenizer.eos_token_id] = -100

        return {
            'sample' : sample,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }