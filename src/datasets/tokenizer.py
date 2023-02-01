import argparse
import json
import os
from typing import Dict, List
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordPiece
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
import tokenizers.decoders as decoders
from tokenizers.decoders import WordPiece
from tokenizers import pre_tokenizers, Regex, normalizers

from src.datasets.config import SPECIAL_TOKENS
from src.datasets.dataset_errors import WrongParameterError

LOWERCASE = False
SUBWORD_PREFIX = '$'


def define_tokenizer(vocab_size: int):
    model = BPE(unk_token="[UNK]", continuing_subword_prefix=SUBWORD_PREFIX)
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
    tokenizer.decoder = WordPiece(prefix=SUBWORD_PREFIX, cleanup=True)  # we use WordPiece just because of the whitespace cleanup

    return tokenizer, trainer


def test(tokenizer, text):
    subwords = tokenizer.encode(text).tokens
    # print(tokenizer.encode(text).ids)
    return ' '.join(subwords)


def tokenizer_batch_generator(input_folder : str):

    if not os.path.isdir(input_folder):
        raise WrongParameterError(f"Input folder \"{input_folder}\" must be a directory")
    
    files = os.listdir(input_folder)
    
    for file in files:
        full_path = os.path.join(input_folder, file)
        if os.path.isdir(full_path):
            try:
                gen = tokenizer_batch_generator(full_path)
                while gen:
                    yield next(gen)
            except:
                ...
                
        else:
            with open(full_path, "r") as fd:
                parsed_objs : List[Dict] = json.load(fd)
            
            batch = []
            for parsed_obj in parsed_objs:
                batch += [parsed_obj.get("comment", ""), parsed_obj.get("header", ""), parsed_obj.get("body", "")]
            
            yield batch




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", "-s", type=int, required=True)
    parser.add_argument("--input_folder", "-i", type=str, required=True)
    parser.add_argument("--output_file", "-o", type=str, required=True)
    args = parser.parse_args()

    tokenizer, trainer = define_tokenizer(args.vocab_size)
    tokenizer.train_from_iterator(tokenizer_batch_generator(args.input_folder), trainer)
    # tokenizer.train([args.input_file], trainer)
    tokenizer.save(args.output_file)

    print("\nTesting the tokenizer...\n")
    tokenizer = Tokenizer.from_file(args.output_file)  # this is how to load the save tokenizer
    texts = [
        """One of the most impressive long term hobby projects is Robert's Rocket Project.\n\n\n He started building a 100 lbf liquid engine in 2001, fired a regeneratively cooled version in 2007, started building a regen 250 lbf in 2008.""",
        """Det globale fondet har selv hatt finansielle utfordringer på grunn av finanskrisen og frys av penger fra givere etter mislighetssaker i enkelte mottakerland.""",
    ]
    for text in texts:
        print(f"INPUT:  {text}\n\nTOKENS: {test(tokenizer, text)}\n\nDECODED: {tokenizer.decode(tokenizer.encode(text).ids)}\n", flush=True)