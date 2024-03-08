import time
from util import nethook
from util.globals import *
import numpy as np
from scipy.linalg import svd

from transformers import LlamaForCausalLM, LlamaTokenizer
from sklearn.utils.extmath import randomized_svd

model = LlamaForCausalLM.from_pretrained("openlm-research/open_llama_3b_v2")
tok = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b_v2")

# model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").cuda()
# tok = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

threshold = 0.99

all_indices = []

for idx in range(1, len(model.model.layers)):
    # openllama3b-v2, idx=0: 3107
    # llama-2, idx=0: 3982

    # w = nethook.get_parameter(model, w_name.format(idx))
    weight = model.model.layers[idx].mlp.down_proj.weight.data.cpu().numpy()


    # SVD decomposition:
    # Compute SVD
    start_time = time.time()
    U, s, Vt = svd(weight)
    end_time = time.time()
    print(f"Time spent for {idx}: {end_time - start_time} seconds")

    # U, Sigma, VT = randomized_svd(weight, n_components=200)

    # # Calculate the percentage of the total energy captured by the top 200 singular values
    # energy_captured = np.sum(Sigma**2) / np.linalg.norm(W, 'fro')**2
    # if energy_captured >= 0.99:
    #     print("A low-rank approximation with k=200 can effectively represent the matrix.")
    # else:
    #     print("Consider increasing k for a better approximation.")

    # Determine a reasonable k for approximation
    for i in range(len(s)):
        if sum(s[:i+1]) / sum(s) >= threshold:
            break
    
    all_indices.append(i+1)

    print("weight:", weight.shape, "99%:", i+1)

print(all_indices)
