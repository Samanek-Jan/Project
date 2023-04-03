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
from torchmetrics.functional.text.rouge import rouge_score
from transformers import AutoConfig, AutoModelWithLMHead, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, Seq2SeqTrainer, TrainingArguments
from src.t5_small.datasets.config import DEVICE
from src.t5_small.datasets.collate_functor import CollateFunctor

from src.t5_small.config import BATCH_SIZE, BOS_TOKEN, EOS_TOKEN, LR, MODELS_OUT_FOLDER, PAD_TOKEN, SEED, UNK_TOKEN, WARMUP_DURATION
from src.t5_small.datasets.github_dataset.remote_dataset import RemoteDataset
from src.t5_small.datasets.local_dataset.local_dataset import LocalDataset

pretraining = False


def main():
    print(f"Using {DEVICE}")
    global pretraining
    argument_parser = argparse.ArgumentParser("Training and testing script")
    argument_parser.add_argument("--epoch_n", "-n", type=int, default=1)
    argument_parser.add_argument("--pretraining", "-p", action='store_const', default=pretraining, const=not(pretraining))
    argument_parser.add_argument("--epoch_size", "-i", type=int, default=20000)
    argument_parser.add_argument("--model_name", "-m", type=str, default="facebook/bart-large")
    argument_parser.add_argument("--tokenizer_name", "-t", type=str, default="facebook/bart-large")
    argument_parser.add_argument("--output_folder", "-o", type=str, default=MODELS_OUT_FOLDER)
    argument_parser.add_argument("--model", "-d", type=str, default=None)
    args = argument_parser.parse_args()

    pretraining = args.pretraining
    # Initializing a GPT configuration
    configuration = AutoConfig.from_pretrained(args.model_name)
    global MAX_SEQUENCE_SIZE
    # MAX_SEQUENCE_SIZE = configuration.n_positions
    MAX_SEQUENCE_SIZE = configuration.max_position_embeddings

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=False, model_max_length=MAX_SEQUENCE_SIZE, add_bos_token=True)
    tokenizer.add_special_tokens({
        # "bos_token" : BOS_TOKEN,
        # "eos_token" : EOS_TOKEN,
        "pad_token" : tokenizer.eos_token,
    })
    # tokenizer.pad_token = PAD_TOKEN
    # tokenizer.bos_token = BOS_TOKEN
    # tokenizer.eos_token = EOS_TOKEN
    # tokenizer.unk_token = UNK_TOKEN
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(DEVICE)
    model.config.problem_type = "multi_label_classification"
    # model.resize_token_embeddings(len(tokenizer))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Initializing a model from the configuration
    model_dict = {}
    if args.model is not None:
        model_dict = torch.load(args.model)
        model.load_state_dict(model_dict["model_dict"])
        optimizer.load_state_dict(model_dict["optimizer_dict"])


    collate_f = CollateFunctor(tokenizer)

    if pretraining:
        train_dataset = RemoteDataset(tokenizer, args.epoch_size)
        valid_dataset = LocalDataset(tokenizer, "valid")
    else:
        train_dataset = LocalDataset(tokenizer, "train")
        valid_dataset = LocalDataset(tokenizer, "valid")

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_f) # type: ignore
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_f) # type: ignore

    param_n = get_n_params(model)
    print(f"Model params num. = {param_n}")

    training_args = TrainingArguments(
        output_dir="./",
        evaluation_strategy="epoch",
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=WARMUP_DURATION,
        seed=SEED,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=args.epoch_n
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_f,
        tokenizer=tokenizer
    )


    # trainer.train()

    train_and_test(model, optimizer, train_dataloader, valid_dataloader, epoch_n=args.epoch_n, model_name=args.model_name, output_folder=args.output_folder, model_d=model_dict)
    print("Done")


def train_and_test(model,
                   optimizer,
                   train_dataloader,
                   test_dataloader,
                   epoch_n,
                   model_name,
                   output_folder,
                   eval_every_n = 5,
                   model_d = {}):

    global pretraining

    best_version = {"bv_BLEU" : model_d.get("bv_BLEU", float("-inf"))}
    scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer = optimizer,
            num_warmup_steps = WARMUP_DURATION
    )

    max_bleu = best_version["bv_BLEU"]
    bleu_score = torchmetrics.BLEUScore()
    tokenizer = train_dataloader.dataset.datasampler.tokenizer

    loss_list = model_d.get("loss_list", [])
    epoch_loss = []
    bleu_list = model_d.get("bleu_list", [])
    rouge_list = model_d.get("rouge_list", [])

    for epoch in range(1, epoch_n+1):
        pbar = tqdm(iter(train_dataloader), leave=False)
        model.train()

        for (x, x_str), (y, y_str) in pbar:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            prediction = model(**x, labels=y["input_ids"])
            loss = prediction.loss
            prediction = torch.argmax(prediction.logits, -1)

            prediction_str = tokenizer.batch_decode(prediction.tolist(), skip_special_tokens=True)
            bleu_score.update(prediction_str, [[sentence] for sentence in y_str])

            pbar.set_description(f"epoch: [{epoch}/{epoch_n}], loss = {loss.item():.4f}")
            epoch_loss.append(float(loss.item()))
            loss.backward()
            optimizer.step()
            # break

        loss_list.append(float(torch.mean(torch.tensor(epoch_loss)).item()))
        epoch_loss.clear()

        scheduler.step()
        bleu_score.reset()

        if epoch % eval_every_n == 0:
            pbar_prefix = f"[{epoch}/{epoch_n}]"
            bleu, rouge, (source_sentences, target_sentences, pred_sentences) = evaluate(model, test_dataloader, pbar_prefix=pbar_prefix)
            bv_bleu = best_version["bv_BLEU"]
            bleu_list.append(bleu)
            rouge_list.append(bleu)
            print(f"{epoch}. best BLEU. = {bv_bleu:.3f}, cur. BLEU. = {bleu:.3f}, cur. Rouge = {rouge:.3f}")
            if bleu >= bv_bleu:
                best_version = {
                    "model_dict" : deepcopy(model.state_dict()),
                    "optimizer_dict" : deepcopy(optimizer.state_dict()),
                    "loss_list" : loss_list,
                    "BLEU_list" : bleu_list,
                    "ROUGE_list" : rouge_list,
                    "epoch" : epoch,
                    "source_sentences" : source_sentences,
                    "target_sentences" : target_sentences,
                    "pred_sentences" : pred_sentences,
                    "bv_BLEU" : bleu,
                    "eval_every_n" : eval_every_n
                }

                # full_path = os.path.join(output_folder, "{}{}.pt".format(model_name, "_pretraining" if pretraining else "_finetunning"))
                full_path = os.path.join(output_folder, "{}{}.pt".format(model_name, "_pretrained"))
                torch.save(best_version, full_path)

            print(f"Training bleu score = {bleu:.3f}")
            print(f"Examples")
            rand_inds = torch.randint(0, len(target_sentences), (3,))
            for i in rand_inds:
                obj = {
                    "SOURCE" : source_sentences[i],
                    "TARGET" : target_sentences[i],
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
    cur_bleu_score = 0
    cur_rouge_score = 0

    # rouge_score = torchmetrics.text.rouge.ROUGEScore(tokenizer=tokenizer, rouge_keys="rougeL")
    for i, ((x, x_str), (y, y_str)) in enumerate(test_dataloader):
        generated_ids = model.generate(x["input_ids"], num_beams=1, min_length=0, max_length=MAX_SEQUENCE_SIZE)
        y_pred = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        sources_list.extend(x_str)
        sentences_target.extend(y_str)
        sentences_pred.extend(y_pred)

        y_str = [[y_sentence] for y_sentence in y_str]
        bleu_score.update(y_pred, y_str)
        cur_bleu_score = bleu_score.compute()
        cur_rouge_score = rouge_score(sentences_pred, sentences_target, tokenizer=tokenizer, rouge_keys="rougeL")["rougeL_fmeasure"]

        test_dataloader.set_description("{} BLEU: {:.3f}, ROUGE: {:.3f}".format(pbar_prefix, cur_bleu_score, cur_rouge_score))

        # break

    print("BLEU: {:.3f}, ROUGE: {:.3f}".format(cur_bleu_score, cur_rouge_score))

    return float(cur_bleu_score), float(cur_rouge_score), (sources_list, sentences_target, sentences_pred)

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