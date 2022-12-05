from typing import List
import torch
import torch.nn.functional as F
import argparse
import os, sys
import json
from tqdm import tqdm
from tokenizers import Tokenizer
from model.baseline.config import BATCH_SIZE, DEVICE

from src.model.baseline.models import Model
from src.model.baseline.search import BeamSearch, GreedySearch
from src.datasets.config import PAD_TOKEN

class TestDataset(torch.utils.data.Dataset):
    
    def __init__(self, sentences : List[str], tokenizer : Tokenizer) -> None:
        super().__init__()
        self.sentences = sentences
        self.tokenizer = tokenizer
                
    def __len__(self) -> int:
        return len(self.sentences)
    
    def __getitem__(self, i) -> List:
        
        x = self.tokenizer.encode(self.sentences[i].get("input", None))        
        return torch.tensor(x.ids, dtype=torch.int32).to(DEVICE), self.sentences[i].get("is_cuda", False), self.sentences[i].get("input", None)
    
    def decode_batch(self, batch, *args, **kwargs):
        return self.tokenizer.decode_batch(batch, *args, **kwargs)
    
    def get_token_id(self, token : str) -> int:
        return self.tokenizer.token_to_id(token)
 

class CollateFunctor:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, samples: list):
        ids, cuda_mask, sentences = zip(*samples)
        ids, mask = self.collate_sentences(ids)
        return (ids, mask, torch.tensor(cuda_mask, dtype=torch.bool), sentences)

    def collate_sentences(self, samples: list):
        lengths = [sentence.size(0) for sentence in samples]
        max_length = max(lengths)

        subword_ids = torch.stack([
            F.pad(sentence, (0, max_length - length), value=self.pad_id)
            for length, sentence in zip(lengths, samples)
        ])
        attention_mask = subword_ids == self.pad_id

        return subword_ids.to(DEVICE), attention_mask.to(DEVICE)



def main():
    argparser = argparse.ArgumentParser("Generate code")
    argparser.add_argument("--tokenizer", "-t", type=str, default="../../../../data/tokenizer/vocab_20000.json")
    argparser.add_argument("--model", "-m", type=str, default="../../../../models/baseline/baseline_model.pt")
    argparser.add_argument("--input", "-i", type=str, default="./input.json")
    argparser.add_argument("--output", "-o", type=str, default="./output.json")
    
    args = argparser.parse_args()
    
    if not os.path.isfile(args.input):
        print("Input file does not exist")
        print("Creating input file template")
        with open(args.input, 'w') as fd:
            json.dump([
                {
                    "input" : "Input sentence",
                    "is_cuda" : True
                }
            ], fd, indent=2)
        return

    elif not os.path.isfile(args.tokenizer):
        print("Tokenizer vocab file does not exist")
        return

    elif not os.path.isfile(args.model):
        print("Model file does not exist")
        return
    
    with open(args.input, 'r', encoding='latin-1') as fd:
        input_sentences = json.load(fd)
        
    tokenizer : Tokenizer = Tokenizer.from_file(args.tokenizer)
    PAD_ID = tokenizer.token_to_id(PAD_TOKEN)
    d = torch.load(args.model, map_location=DEVICE)
    embedd_dim = d["transformer_kwargs"]["embedd_dim"]
    del d["transformer_kwargs"]["embedd_dim"]
    model = Model(tokenizer.get_vocab_size(), embedd_dim, None, PAD_ID, **d["transformer_kwargs"]).to(DEVICE)
    model.load_state_dict(d["model_dict"])
    searcher = BeamSearch(model, tokenizer)
    
    dataset = TestDataset(input_sentences, tokenizer)
    collate_f = CollateFunctor(PAD_ID)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=collate_f)
    
    outputs = []
    pbar = tqdm(dataloader, leave=True)
    for ids, mask, cuda_mask, input_sentences in pbar:
        output_sentences = searcher(ids, mask, cuda_mask)
        
        for input_s, output_s, is_cuda in zip(input_sentences, output_sentences, cuda_mask):
            outputs.append({
                "input" : input_s,
                "output" : output_s,
                "is_cuda" : bool(is_cuda)
            })
            
    with open(args.output, "w") as fd:
        json.dump(outputs, fd, indent=2)
    
    print("Done!")
    
if __name__ == '__main__':
    main()