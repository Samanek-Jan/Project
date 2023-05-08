import random
from typing import Callable
import torch
import torch.nn as nn
import math
from baseline.config import BATCH_SIZE

from src.baseline.datasets.config import DEVICE

from src.baseline.transformer import Transformer

class Model(nn.Module):
    def __init__(self, 
                 num_embedding : int, 
                 embedding_dim : int, 
                 loss_fce : Callable, 
                 pad_id : int,
                 bos_id : int,
                 transformer_kwargs):
        super().__init__()
        self.embedding = nn.Embedding(num_embedding, embedding_dim, padding_idx=pad_id)
        self.embedding.weight.data /= math.sqrt(embedding_dim)  # descale the weights
        self.transformer = torch.nn.Transformer(dim_feedforward=embedding_dim, **transformer_kwargs, batch_first=True, norm_first=True)
        # self.transformer = Transformer(transformer_kwargs)
        self.head = nn.Linear(embedding_dim, num_embedding)
        # self.head.weight = self.embedding.weight
        self.head.weight.data /= math.sqrt(embedding_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.loss_fce = loss_fce
        self.pad_id = pad_id
        self.bos_id = bos_id

    def forward(self, x : tuple, y : tuple, teacher_forcing=0):
        x_ids, x_mask = x
        y_ids, y_mask = y
        
        x_ids = self.embedding(x_ids)
        y_tgt_ids = self.embedding(y_ids[:,:-1])
        y_tgt_mask = y_mask[:,:-1].to(DEVICE)
        
        target = y_ids[:,1:].to(x_ids.device)  
        
        prediction = None
        if self.training and random.random() < teacher_forcing:
            context = torch.full((len(x_mask),), self.bos_id).unsqueeze(-1).to(x_ids.device)

            for _ in range(len(y_ids[0])):
                pred_step = self.decode_step(x_ids, x_mask, context)[:,-1,:]
                if prediction is None:
                    prediction = pred_step.unsqueeze(1)
                else:
                    prediction = torch.cat([prediction, pred_step.unsqueeze(1)], dim=1).to(x_ids.device)
                context = torch.cat([context, pred_step.argmax(-1).unsqueeze(1)], dim=1).to(x_ids.device)

            prediction = prediction[:,1:]
            
        else: # Enforcing teacher forcing
            prediction = self.transformer(x_ids, y_tgt_ids, src_key_padding_mask=x_mask, tgt_key_padding_mask=y_tgt_mask)
            # preds = self.transformer(x_ids, x_mask, y_tgt_ids, y_tgt_mask)
            prediction = self.head(prediction)
            # source_encoding = self.encode_source(x_ids, x_mask)
                      
            
        loss = self.loss_fce(prediction.permute(0, 2, 1), target)
                
        return prediction, loss


    def encode_source(self, source, source_mask) -> torch.Tensor:
        embeddings = self.embedding(source)
        return self.transformer.encoder(embeddings, src_key_padding_mask=source_mask.type(torch.bool))

    def decode_step(self, source_encoding, source_mask, target_prefix) -> torch.Tensor:
        if type(target_prefix) is torch.Tensor:
            embeddings = self.embedding(target_prefix)
            target_mask = target_prefix == self.pad_id
            # target_mask = None
            target = self.transformer.decoder(embeddings, source_encoding, tgt_key_padding_mask=target_mask.type(torch.bool), memory_key_padding_mask=source_mask.type(torch.bool))
        else:
            embeddings = self.embedding(target_prefix[0])
            target_mask = target_prefix[1]
            target = self.transformer.decoder(embeddings, source_encoding, tgt_key_padding_mask=target_mask.type(torch.bool), memory_key_padding_mask=source_mask.type(torch.bool))
        return self.head(target)
    
    def get_input_embeddings(self):
        return self.embedding
        
    
# if __name__ == "__main__":
    
#     transformer_kwargs = {
#                         "num_encoder_layers" : 1, 
#                         "num_decoder_layers" : 1,
#                         "hidden_size" : 100,
#                         "num_heads" : 1,
#                         "dropout" : 0.3
#                         }
    
#     num_embedds = 15000
#     model = Model(num_embedds, 100, **transformer_kwargs).to(DEVICE)
#     dataset = TranslationDataset("../data/train_government.txt", "../data/vocab1000.json")
#     collate_f = CollateFunctor(PAD_ID)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False, collate_fn=collate_f)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
#     tokenizer = dataset.tokenizer
    
#     param_n = get_n_params(model)
#     print(f"Model param n = {param_n}")
    
#     # state_dict = torch.load("models/model_test.pt")
#     # model.load_state_dict(state_dict)
#     model.train()
    
#     bleu_score, (tg_sentences, pred_sentences) = evaluate(model, dataloader)
#     print(f"Bleu score = {bleu_score}")
#     rand_inds = torch.randint(0, len(tg_sentences), (3,))
#     for i in rand_inds:
#         print(tg_sentences[i])
#         print(pred_sentences[i])
#         print()
    
    # for i, (x, y, (src_s, trg_s)) in enumerate(dataloader, 1): 
    #     # optimizer.zero_grad()
    #     preds, loss = model(x, y)
    #     print(f"{i}. preds shape = {preds.shape}, Loss = {loss.item():.4f}")
    #     print(x[0])
    #     print(y[0][:,1:])
    #     print(preds.argmax(-1))
    #     print(trg_s)
    #     print(tokenizer.decode_batch(preds.argmax(-1).tolist(), skip_special_tokens=True))
    #     print()
        
        # loss.backward()
        # optimizer.step()
        # if i == 3:
        #     break
        
    # torch.save(model.state_dict(), "models/model_test.pt")
    