import argparse
import os
from typing import Dict, List
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordPiece
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
import tokenizers.decoders as decoders
from tokenizers.decoders import WordPiece
from tokenizers import pre_tokenizers, Regex, normalizers
from tqdm import tqdm

from src.datasets.interface.config import NEWLINE_TOKEN_TRANSLATION, SPACE_TOKEN_TRANSLATION, SPECIAL_TOKENS, SUBWORD_PREFIX, LOWERCASE, NEWLINE_TOKEN, SPACE_TOKEN, UNK_TOKEN
from src.datasets.interface.config import mongoDB

class CupydTokenizer(Tokenizer):
    
    def __init__(self, tokenizer_path : str = None, special_translations : dict = {NEWLINE_TOKEN : NEWLINE_TOKEN_TRANSLATION, SPACE_TOKEN : SPACE_TOKEN_TRANSLATION}):
        
        if tokenizer_path is not None:
            self.tokenizer : Tokenizer = Tokenizer.from_file(tokenizer_path)
        self.special_translations : dict = special_translations
        
    def translate(self, sentence : str) -> str:
        for key, value in self.special_translations.items():
            sentence = sentence.replace(key, value)
        return sentence
        
    def trainslate_back(self, sentence : str) -> List:
        for key, value in self.special_translations.items():
            sentence = sentence.replace(value, key)
        
        return sentence
        
    def encode(self, sentence : str) -> List:
        return self.tokenizer.encode(self.translate(sentence))
    
    def encode_batch(self, sentences : List[str]) -> List[List]:
        return map(lambda sentence : self.encode(sentence), sentences)
        
    def decode(self, tokens : list) -> str:
        return self.trainslate_back(self.tokenizer.decode(tokens))
    
    def decode_batch(self, sentences : List[List]) -> List[str]:
        return map(lambda sentence : self.decode(sentence), sentences)
    
    def token_to_id(self, token):
        return self.tokenizer.token_to_id(token)

def define_tokenizer(vocab_size: int):
    model = BPE(unk_token=UNK_TOKEN, continuing_subword_prefix=SUBWORD_PREFIX)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        continuing_subword_prefix=SUBWORD_PREFIX
    )
    tokenizer = Tokenizer(model)
    tokenizer.normalizer = normalizers.Sequence([
        # normalizers.Replace(Regex("\t"), " "),
        # normalizers.Replace(Regex(" {2,}"), " "),
        normalizers.BertNormalizer(
            lowercase=LOWERCASE,
            clean_text=True,
            strip_accents=False
        )
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.decoder = decoders.WordPiece(prefix=SUBWORD_PREFIX)  # we use WordPiece just because of the whitespace cleanup

    return tokenizer, trainer


def test(tokenizer, text):
    subwords = tokenizer.encode(text).tokens
    # print(tokenizer.encode(text).ids)
    return ' '.join(subwords)


def tokenizer_batch_generator():
    cuda_snippets = mongoDB["cuda_snippets"]
    train = cuda_snippets["train"]
    validation = cuda_snippets["validation"]
    pbar = tqdm({"train" : train, "validation" : validation}.items(), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    
    for name, collection in pbar:
        for kernel in collection.find():
            kernel_str = kernel.get("comment", "") + kernel.get("header", "") + kernel.get("body", "")
            yield CupydTokenizer().translate(kernel_str)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", "-s", type=int, default=10000)
    parser.add_argument("--output_file", "-o", type=str, default="tmp_vocab.json")
    args = parser.parse_args()

    tokenizer, trainer = define_tokenizer(args.vocab_size)
    tokenizer.train_from_iterator(tokenizer_batch_generator(), trainer)
    # tokenizer.train([args.input_file], trainer)
    tokenizer.save(args.output_file)

    print("\nTesting the tokenizer...\n")
    tokenizer = CupydTokenizer(args.output_file)  # this is how to load the save tokenizer
    texts = [
        """One of the most impressive long term hobby projects is Robert's Rocket Project.\n\n\n He started building a 100 lbf liquid engine in 2001, fired a regeneratively cooled version in 2007, started building a regen 250 lbf in 2008.""",
        """Det globale fondet har selv hatt finansielle utfordringer på grunn av finanskrisen og frys av penger fra givere etter mislighetssaker i enkelte mottakerland.""",
    ]
    for text in texts:
        print(f"INPUT:  {text}\n\nTOKENS: {test(tokenizer, text)}\n\nDECODED: {tokenizer.decode(tokenizer.encode(text).ids)}\n", flush=True)