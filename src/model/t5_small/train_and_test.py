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

from src.model.t5_small.config import BATCH_SIZE, LR, MODELS_OUT_FOLDER, WARMUP_DURATION
from src.datasets.github_dataset.remote_dataset import RemoteDataset
from src.datasets.local_dataset.local_dataset import LocalDataset

pretraining = False


def main():
    print(f"Using {DEVICE}")
    global pretraining
    argument_parser = argparse.ArgumentParser("Training and testing script")
    argument_parser.add_argument("--epoch_n", "-n", type=int, default=1)
    argument_parser.add_argument("--pretraining", "-p", action='store_const', default=pretraining, const=not(pretraining))
    argument_parser.add_argument("--epoch_size", "-i", type=int, default=20000)
    argument_parser.add_argument("--model_name", "-m", type=str, default="t5-small")
    argument_parser.add_argument("--tokenizer_name", "-t", type=str, default="t5-small")
    argument_parser.add_argument("--output_folder", "-o", type=str, default=MODELS_OUT_FOLDER)
    argument_parser.add_argument("--model", "-d", type=str, default=None)
    args = argument_parser.parse_args()
    
    pretraining = args.pretraining
    # Initializing a GPT configuration
    configuration = AutoConfig.from_pretrained(args.model_name)
    global MAX_SEQUENCE_SIZE
    MAX_SEQUENCE_SIZE = configuration.n_positions
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, model_max_length=MAX_SEQUENCE_SIZE)
    model = AutoModelForSeq2SeqLM.from_config(configuration).to(DEVICE)
    model.resize_token_embeddings(len(tokenizer))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        
    # Initializing a model from the configuration
    if args.model is not None:
        model_dict = torch.load(args.model)
        model.load_state_dict(model_dict["model_dict"])
        optimizer.load_state_dict(model_dict["optimizer_dict"])

    
    collate_f = CollateFunctor(tokenizer, MAX_SEQUENCE_SIZE, MAX_SEQUENCE_SIZE)
    
    if pretraining:
        train_dataset = RemoteDataset(tokenizer, MAX_SEQUENCE_SIZE, MAX_SEQUENCE_SIZE, args.epoch_size)
        valid_dataset = LocalDataset(tokenizer, MAX_SEQUENCE_SIZE, MAX_SEQUENCE_SIZE, "valid")
    else:
        train_dataset = LocalDataset(tokenizer, MAX_SEQUENCE_SIZE, MAX_SEQUENCE_SIZE, "train")
        valid_dataset = LocalDataset(tokenizer, MAX_SEQUENCE_SIZE, MAX_SEQUENCE_SIZE, "valid")
        
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_f)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_f)
        
    param_n = get_n_params(model)
    print(f"Model params num. = {param_n}")
    
    train_and_test(model, optimizer, train_dataloader, valid_dataloader, epoch_n=args.epoch_n, model_name=args.model_name, output_folder=args.output_folder)
    print("Done")


def train_and_test(model,
                   optimizer,
                   train_dataloader, 
                   test_dataloader,
                   epoch_n,
                   model_name,
                   output_folder,
                   eval_every_n = 1):
    
    global pretraining
    
    best_version = {"BLEU" : float("-inf")}
    scheduler = transformers.get_constant_schedule_with_warmup(                
            optimizer = optimizer,
            num_warmup_steps = WARMUP_DURATION
    )
    
    max_bleu = best_version["BLEU"]
    bleu_score = torchmetrics.BLEUScore()
    tokenizer = train_dataloader.dataset.datasampler.tokenizer
    
    for epoch in range(1, epoch_n+1):
        pbar = tqdm(iter(train_dataloader), leave=False)
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
            if bleu >= bv_bleu:
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
                    
                full_path = os.path.join(output_folder, "{}{}.pt".format(model_name, "_pretraining" if pretraining else "_finetunning"))
                torch.save(best_version, full_path) 
        
        print(f"Training bleu score = {bleu:.3f}")
        print(f"Examples")
        rand_inds = torch.randint(0, len(target_sentences), (3,))
        for i in rand_inds:
            obj = {
                "SOURCE" : source_sentences[i],
                "TARGET" : target_sentences[i][0],
                "PREDIC" : pred_sentences[i]
            }
            print(json.dumps(obj, indent=2))
        print("--------------------------------\n")
        
        if bleu >= max_bleu:
            max_bleu = bleu
            
        torch.cuda.empty_cache()
           
    return best_version


@torch.no_grad()
def evaluate(model, test_dataloader, pbar_prefix=""):
    model.eval()
    sources_list = []
    sentences_target = []
    sentences_pred = []
    tokenizer = test_dataloader.dataset.datasampler.tokenizer

    test_dataloader = tqdm(test_dataloader, leave=False)

    bleu_score = torchmetrics.BLEUScore(tokenizer=tokenizer)
    rouge_score = torchmetrics.text.rouge.ROUGEScore(tokenizer=tokenizer)
    for i, ((x, x_str), (y, y_str)) in enumerate(test_dataloader):
        generated_ids = model.generate(x["input_ids"], num_beams=1, min_length=0, max_length=MAX_SEQUENCE_SIZE)
        y_pred = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        rouge_score.update(y_pred, y_str)
        sources_list.extend(x_str)
        sentences_target.extend(y_str)
        sentences_pred.extend(y_pred)

        y_str = [[y_sentence] for y_sentence in y_str]
        bleu_score.update(y_pred, y_str)
        cur_bleu_score = bleu_score.compute()
        
        test_dataloader.set_description("{} BLEU: {:.3f}, ROUGE: {:.3f}".format(pbar_prefix, cur_bleu_score, rouge_score.compute()["rougeL_fmeasure"]))
        
        # break
        
    print("BLEU: {:.3f}, ROUGE: {:.3f}".format(bleu_score.compute(),  rouge_score.compute()["rougeL_fmeasure"]))
    
    return float(bleu_score.compute()), float(rouge_score.compute()["rougeL_fmeasure"]), (sources_list, sentences_target, sentences_pred)

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