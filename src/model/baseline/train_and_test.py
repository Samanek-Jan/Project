from copy import deepcopy
import os, sys
from tokenizers import Tokenizer
import torch
import torch.nn as nn
import torchmetrics
from tqdm import tqdm
import argparse
import transformers

from src.datasets.config import CUDA_BOS_TOKEN, PAD_TOKEN
from src.datasets.dataset import CollateFunctor, Dataset
from model.baseline.config import BATCH_SIZE, DEVICE, LR, MAX_X, MAX_Y, MIN_X, MIN_Y, MODELS_OUT_FOLDER, WARMUP_DURATION
from model.baseline.linear_lr import LinearLR
from model.baseline.models import Model
from model.baseline.search import GreedySearch


def main():
    print(f"Using {DEVICE}")
    
    argument_parser = argparse.ArgumentParser("Training and testing script")
    argument_parser.add_argument("--tokenizer", "-t", required=True, type=str)
    argument_parser.add_argument("--train_folder", "-f", required=True, type=str)
    argument_parser.add_argument("--valid_folder", "-v", required=True, type=str)
    argument_parser.add_argument("--embedd_dim", "-e", type=int, default=300)
    argument_parser.add_argument("--num_encoder_layers", "-c", type=int, default=4)
    argument_parser.add_argument("--num_decoder_layers", "-o", type=int, default=3)
    argument_parser.add_argument("--num_heads", "-s", type=int, default=5)
    argument_parser.add_argument("--dropout", "-d", type=float, default=0.1)
    argument_parser.add_argument("--epoch_n", "-n", type=int, default=3)
    argument_parser.add_argument("--epoch_size", "-i", type=int, default=20000)
    args = argument_parser.parse_args()
    
    data_sampler_kwargs = {
        "min_x" : MIN_X,
        "max_x" : MAX_X,
        "min_y" : MIN_Y,
        "max_y" : MAX_Y,
        "tokenizer_path" : args.tokenizer
    }
    
    transformer_kwargs = { 
                        "num_encoder_layers" : args.num_encoder_layers, 
                        "num_decoder_layers" : args.num_decoder_layers,
                        "hidden_size" : args.embedd_dim,
                        "num_heads" : args.num_heads,
                        "dropout" : args.dropout
                        }
    
    train_dataset = Dataset(in_folder=args.train_folder, epoch_len=args.epoch_size, samples_per_obj=10, **data_sampler_kwargs)
    valid_dataset = Dataset(args.valid_folder, epoch_len=args.epoch_size//10, shuffle=True, samples_per_obj=10, **data_sampler_kwargs)
    PAD_ID = train_dataset.get_token_id(PAD_TOKEN)
    collate_f = CollateFunctor(PAD_ID)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True, collate_fn=collate_f)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=collate_f)
    
    loss_fce = nn.CrossEntropyLoss(ignore_index=-1)
    
    model = Model(train_dataset.get_vocab_size(), args.embedd_dim, loss_fce, PAD_ID, **transformer_kwargs).to(DEVICE)
    param_n = get_n_params(model)
    print(f"Model params num. = {param_n}")
    
    best_model = train_and_test(model, train_dataloader, valid_dataloader, epoch_n=args.epoch_n)
    best_model["transformer_config"] = transformer_kwargs

    model_name = f"baseline_model.pt"
    full_path = os.path.join(MODELS_OUT_FOLDER, model_name)
    torch.save(best_model, full_path) 
    print("Done")


def train_and_test(model, 
                   train_dataloader, 
                   test_dataloader, 
                   eval_every_n     = 1, 
                   epoch_n          = 1):
    
    best_version = {"BLEU" : float("-inf")}
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # optimizer = transformer_AdamW_LLRD(model)
    scheduler = LinearLR(optimizer)
    
    scheduler = transformers.get_constant_schedule_with_warmup(                
            optimizer = optimizer,
            num_warmup_steps = WARMUP_DURATION,
            # num_training_steps = train_dataloader.__len__()
    )
    
    max_bleu = best_version["BLEU"]
    bleu_score = torchmetrics.BLEUScore()
    
    for epoch in range(1, epoch_n+1):
        pbar = tqdm(train_dataloader, leave=False)
        model.train()

        for x, y, (x_str, y_str) in pbar:
            optimizer.zero_grad()            
            prediction, loss = model(x, y)
            prediction = prediction.argmax(-1)
            prediction_str = train_dataloader.dataset.decode_batch(prediction.tolist(), skip_special_tokens=True)
            bleu_score.update(prediction_str, [[sentence] for sentence in y_str])
            
            pbar.set_description(f"epoch: [{epoch}/{epoch_n}], loss = {loss.item():.4f}")
            # print(f"loss = {loss.item():.4f}")
            loss.backward()
            optimizer.step()
            
        scheduler.step()
        score_val = bleu_score.compute()
        bleu_score.reset()
                
        if epoch % eval_every_n == 0:
            bleu, (source_sentences, target_sentences, pred_sentences) = evaluate(model, test_dataloader)
            bv_bleu = best_version["BLEU"]
            print(f"{epoch}. best ver. BLEU. = {bv_bleu:.3f}, currect ver. BLEU. = {bleu:.3f}")
            if bv_bleu < bleu:
                best_version = {
                    "model_dict" : deepcopy(model.state_dict()),
                    "BLEU" : bleu,
                    "epoch" : epoch,
                    "source_sentences" : source_sentences,
                    "target_sentences" : target_sentences,
                    "pred_sentences" : pred_sentences,
                }
        
        print(f"Training bleu score = {score_val:.3f}")
        print(f"Example of translations")
        rand_inds = torch.randint(0, len(target_sentences), (2,))
        for i in rand_inds:
            print("source:\n\t{}\n\ntarget:\n\t{}\nprediction:\n\t{}\n".format(source_sentences[i], target_sentences[i][0], pred_sentences[i]))
            # print(source_sentences[i])
            # print(target_sentences[i][0])
            # print(pred_sentences[i])
            # print()
        print("--------------------------------\n")
        
        if bleu > max_bleu:
            max_bleu = bleu
            
        torch.cuda.empty_cache()
           
    return best_version


@torch.no_grad()
def evaluate(model, test_dataloader, search_class=GreedySearch, pbar : bool=True):
    model.eval()
    sources_list = []
    sentences_target = []
    sentences_pred = []
    tokenizer : Tokenizer = test_dataloader.dataset.get_tokenizer()
    searcher = search_class(model, tokenizer)
    bleu_score_metric = torchmetrics.BLEUScore()

    if pbar:
        test_dataloader = tqdm(test_dataloader, leave=False)

    # bleu_score = torchmetrics.BLEUScore()
    for i, ((sources, sources_mask), (y_ids, _), (sources_str, targets_str)) in enumerate(test_dataloader):
        
        y_bos = y_ids[:,0]
        cuda_bos = tokenizer.token_to_id(CUDA_BOS_TOKEN)
        cuda_mask = torch.where(y_bos == cuda_bos, True, False)
        predictions_str = searcher(sources, sources_mask, cuda_mask)
        # bleu_score.update(predictions_str, targets_str)
        
        sources_list.extend(sources_str)
        sentences_target.extend([[sentence] for sentence in targets_str])
        sentences_pred.extend(predictions_str)
        
        test_dataloader.set_description("BLEU score: {:.3f}".format(bleu_score_metric(sentences_pred, sentences_target)))

    bleu_score = bleu_score_metric(sentences_pred, sentences_target)

    # print(f"BLEU score: {bleu_score.compute() * 100.0}")
    # score_val = bleu_score.compute()
    # bleu_score.reset()
    
    return float(bleu_score), (sources_list, sentences_target, sentences_pred)

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