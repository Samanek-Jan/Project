from copy import deepcopy
import os, sys
from tokenizers import Tokenizer
import torch
import torch.nn as nn
import torchmetrics
from tqdm import tqdm
import argparse
import transformers
import json
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from src.datasets.collate_functor import CollateFunctor
from src.datasets.config import BOS_TOKEN, DEVICE, EOS_TOKEN, MASK_TOKEN, MAX_SEQUENCE_SIZE, PAD_TOKEN

from src.model.bart.config import BATCH_SIZE, LR, MODELS_OUT_FOLDER, WARMUP_DURATION
from src.datasets.github_dataset.remote_dataset import RemoteDataset
from src.datasets.local_dataset.local_dataset import LocalDataset


def main():
    print(f"Using {DEVICE}")
    
    argument_parser = argparse.ArgumentParser("Training and testing script")
    argument_parser.add_argument("--epoch_n", "-n", type=int, default=1)
    argument_parser.add_argument("--pretraining", "-p", type=bool, default=False)
    argument_parser.add_argument("--epoch_size", "-i", type=int, default=20000)
    argument_parser.add_argument("--model_name", "-m", type=str, default="t5-small")
    argument_parser.add_argument("--tokenizer_name", "-t", type=str, default="t5-small")
    argument_parser.add_argument("--output_folder", "-o", type=str, default=MODELS_OUT_FOLDER)
    args = argument_parser.parse_args()
    
    # Initializing a GPT configuration
    configuration = AutoConfig.from_pretrained(args.model_name)
    global MAX_SEQUENCE_SIZE
    MAX_SEQUENCE_SIZE = configuration.n_positions
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, model_max_length=MAX_SEQUENCE_SIZE)
    
    # Initializing a model from the configuration
    model = AutoModelForSeq2SeqLM.from_config(configuration).to(DEVICE)
    model.resize_token_embeddings(len(tokenizer))
    
    collate_f = CollateFunctor(tokenizer, MAX_SEQUENCE_SIZE, MAX_SEQUENCE_SIZE)
    
    if args.pretraining:
        train_dataset = RemoteDataset(tokenizer, MAX_SEQUENCE_SIZE, MAX_SEQUENCE_SIZE, args.epoch_size)
        valid_dataset = LocalDataset(tokenizer, MAX_SEQUENCE_SIZE, MAX_SEQUENCE_SIZE, "valid")
    else:
        train_dataset = LocalDataset(tokenizer, MAX_SEQUENCE_SIZE, MAX_SEQUENCE_SIZE, "train")
        valid_dataset = LocalDataset(tokenizer, MAX_SEQUENCE_SIZE, MAX_SEQUENCE_SIZE, "valid")
        
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_f)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_f)
        
    param_n = get_n_params(model)
    print(f"Model params num. = {param_n}")
    
    train_and_test(model, train_dataloader, valid_dataloader, epoch_n=args.epoch_n, model_name=args.model_name, output_folder=args.output_folder)
    print("Done")


def train_and_test(model, 
                   train_dataloader, 
                   test_dataloader,
                   epoch_n,
                   model_name,
                   output_folder,
                   eval_every_n = 1):
    
    best_version = {"BLEU" : float("-inf")}
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # optimizer = transformer_AdamW_LLRD(model)
    # scheduler = LinearLR(optimizer)
    
    scheduler = transformers.get_constant_schedule_with_warmup(                
            optimizer = optimizer,
            num_warmup_steps = WARMUP_DURATION
    )
    
    max_bleu = best_version["BLEU"]
    bleu_score = torchmetrics.BLEUScore()
    tokenizer = train_dataloader.dataset.datasampler.tokenizer
    
    for epoch in range(1, epoch_n+1):
        pbar = tqdm(iter(train_dataloader), leave=True)
        model.train()

        for (x, x_str), (y, y_str) in pbar:
            optimizer.zero_grad()
            prediction = model(**x, labels=y["input_ids"])
            loss = prediction.loss
            prediction = torch.argmax(prediction.logits, -1)
            
            prediction_str = tokenizer.batch_decode(prediction.tolist(), skip_special_tokens=True)
            bleu_score.update(prediction_str, [[sentence] for sentence in y_str])
            
            pbar.set_description(f"epoch: [{epoch}/{epoch_n}], loss = {loss.item():.4f}")
            # print(f"loss = {loss.item():.4f}")
            loss.backward()
            optimizer.step()
            
        scheduler.step()
        bleu_score.reset()
                
        if epoch % eval_every_n == 0:
            pbar_prefix = f"[{epoch}/{epoch_n}]"
            bleu, rouge, (source_sentences, target_sentences, pred_sentences) = evaluate(model, test_dataloader, pbar_prefix=pbar_prefix)
            bv_bleu = best_version["BLEU"]
            print(f"{epoch}. best BLEU. = {bv_bleu:.3f}, cur. BLEU. = {bleu:.3f}, cur. Rouge = {rouge:.3f}")
            if bv_bleu < bleu:
                best_version = {
                    "model_dict" : deepcopy(model.state_dict()),
                    "optimizer_dict" : deepcopy(optimizer.state_dict()),
                    "BLEU" : bleu,
                    "ROUGE" : rouge,
                    "epoch" : epoch,
                    "source_sentences" : source_sentences,
                    "target_sentences" : target_sentences,
                    "pred_sentences" : pred_sentences,
                }
                    
                full_path = os.path.join(output_folder, f"{model_name}.pt")
                torch.save(best_version, full_path) 
        
        # print(f"Training bleu score = {score_val:.3f}")
        # print(f"Example of translations")
        # rand_inds = torch.randint(0, len(target_sentences), (3,))
        # for i in rand_inds:
        #     obj = {
        #         "source" : source_sentences[i],
        #         "target" : target_sentences[i][0],
        #         "predic" : pred_sentences[i]
        #     }
        #     print(json.dumps(obj, indent=2))
        # print("--------------------------------\n")
        
        if bleu > max_bleu:
            max_bleu = bleu
            
        torch.cuda.empty_cache()
           
    return best_version


@torch.no_grad()
def evaluate(model, test_dataloader, pbar_prefix=""):
    model.eval()
    sources_list = []
    sentences_target = []
    sentences_pred = []
    tokenizer : Tokenizer = test_dataloader.dataset.datasampler.tokenizer
    bleu_score_metric = torchmetrics.BLEUScore()

    test_dataloader = tqdm(test_dataloader, leave=False)

    bleu_score = torchmetrics.BLEUScore()
    for i, ((x, x_str), (y, y_str)) in enumerate(test_dataloader):
        generated_ids = model.generate(x["input_ids"], num_beams=1, min_length=0, max_length=MAX_SEQUENCE_SIZE)
        y_pred = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        
        sources_list.extend(x_str)
        sentences_target.extend([[sentence] for sentence in y_str])
        sentences_pred.extend(y_pred)
        
        test_dataloader.set_description("{} BLEU: {:.3f}, ROUGE: {:.3f}".format(pbar_prefix, bleu_score_metric(sentences_pred, sentences_target), torchmetrics.functional.rouge_score(sentences_pred, sentences_target)["rougeL_fmeasure"]))
        bleu_score.update(y_pred, y_str)
        
        # break
        

    bleu_score = bleu_score_metric(sentences_pred, sentences_target)
    rouge_score = torchmetrics.functional.rouge_score(sentences_pred, sentences_target)["rougeL_fmeasure"]

    print(f"BLEU: {bleu_score:.3f}, ROUGE: {rouge_score:.3f}")
    
    return float(bleu_score), float(rouge_score), (sources_list, sentences_target, sentences_pred)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


if __name__ == "__main__":
    main()