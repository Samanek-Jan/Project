import torch
import os

if __name__ == '__main__':
    file = "../models/baseline/baseline_model.pt"
    
    model_d = torch.load(file)
    # print("dropout: " + model_d["dropout"])
    # print("num_heads: " + model_d["num_heads"])
    # print("hidden_size: " + model_d["hidden_size"])
    # print("num_decoder_layers: " + model_d["num_decoder_layers"])
    # print("num_encoder_layers: " + model_d["num_encoder_layers"])
    
    print(model_d["transformer_kwargs"])
    
    for source, target, prediction in zip(model_d["source_sentences"], model_d["target_sentences"], model_d["pred_sentences"]):
        print(source + "\n")
        print("\n-------------------------\n")
        print(target[0])
        print("\n-------------------------\n")
        print(prediction)
        print("\n-------------------------\n")
        print("-------------------------\n")
        input()