from transformers import AutoConfig, AutoTokenizer, GPT2Model, GPT2Tokenizer

TOKENIZER_NAME = "t5-small"

DATA = """"
using T = int;

T** _data;
int* _size;
int* _numOrigEntries;
int* _numEntries;
#include <cstdint>
int main() {
    return 0;
}

__host__ __inline__ void init()
{
    T* data = nullptr;
    getInstance().acquireMemory<T*>(1, _data);
    getInstance().acquireMemory<int>(1, _numEntries);
    getInstance().acquireMemory<int>(1, _numOrigEntries);
    getInstance().acquireMemory<int>(1, _size);

    CHECK_FOR_CUDA_ERROR(cudaMemcpy(_data, &data, sizeof(T*), cudaMemcpyHostToDevice));
    CHECK_FOR_CUDA_ERROR(cudaMemset(_numEntries, 0, sizeof(int)));
    CHECK_FOR_CUDA_ERROR(cudaMemset(_numOrigEntries, 0, sizeof(int)));
    CHECK_FOR_CUDA_ERROR(cudaMemset(_size, 0, sizeof(int)));
}
"""

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.add_tokens(["{", "}", "<", ">", ";", "[", "]", "&", "*"])
    
    print(len(tokenizer))
    x = tokenizer(DATA)
    
    print(decoded := tokenizer.batch_decode([x["input_ids"]])[0].replace(";", ";\n").replace("{", "{\n").replace("}", "}\n"))
    print("sequence is same: ", decoded == DATA)
    
    