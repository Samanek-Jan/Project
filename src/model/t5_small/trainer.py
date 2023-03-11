import random
from typing import Union
import torch

from src.datasets.config import MAX_SEQUENCE_SIZE

class Trainer:
    
    def __init__(self, model, optimizer, scheduler, eos_id, teacher_forcing : Union(float, None) = None):
        self.teacher_forcing = teacher_forcing
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.model = model
        self.eos_id = eos_id
        
    def train_batch(self, batch, max_iterations=MAX_SEQUENCE_SIZE):
        
        ids = batch["input_ids"]
        mask = batch["attention_mask"]
        labels = batch["labels"]
        
        batch_size = len(ids)
        
        if self.teacher_forcing is None or random.random() > self.teacher_forcing:
            sentence_last_token = torch.zeros([batch_size])
            pred_ids = torch.empty([batch_size])
            for _ in range(MAX_SEQUENCE_SIZE):
                output = self.model(input_ids=ids, attention_mask=mask)
                
                pred_ids = torch.argmax(output, -1)
                sentence_last_token = torch.where(sentence_last_token == self.eos_id or pred_ids == self.eos_id, self.eos_id, 0)
                
                
                
                
            torch.argmax(output)
        
        else:
            ...
            
        