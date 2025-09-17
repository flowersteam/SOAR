# can be use to merge and deduplicate the results multiple models (for train only)
# dedup correct response base on embedding space and incorrect response based on output grid
import pickle
import numpy as np
from soar.llm_utils import merge_results,to_formated_list
from tqdm import tqdm
from soar.preprocess import get_dataset


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--base_path", type=str, help="Path to soar git repo")
for i in range(20):
    parser.add_argument(f"--path_{i}", type=str, default="",
                        help=f"Path to a result file (can be empty if not used)")

parser.add_argument("--path_save", type=str)
parser.add_argument("--path_embed_model", type=str, default="/home/flowers/work/hf/CodeRankEmbed")
parser.add_argument("--arc2", action=argparse.BooleanOptionalAction, help="arc 2",default=False)
parser.add_argument("--split", type=str, help="split train val",default="train")

args = parser.parse_args()

train_data, val_data, test_data = get_dataset(args.base_path,arc_2=args.arc2)

all_data = train_data
if args.split == "val":
    all_data = val_data
# all_data.update(val_data)

list_k = list(all_data.keys())

from sentence_transformers import SentenceTransformer
model = SentenceTransformer(args.path_embed_model, trust_remote_code=True)

def rm_duplicate_resps(list_feat, list_resp, threshold = 0.9):
    if len(list_resp) == 0:
        return []
    if isinstance(list_feat, list):
        list_feat = np.array(list_feat)
    
    # Compute the cosine similarity matrix
    cosine_similarity_matrix = np.dot(list_feat, list_feat.T)
    duplicate_indices = []
    
    # Only check each pair once (upper triangle)
    for i in range(len(cosine_similarity_matrix)):
        if i in duplicate_indices:
            continue
        for j in range(i+1, len(cosine_similarity_matrix)):

            if cosine_similarity_matrix[i, j] > threshold:
                # Keep the first document, mark the second as duplicate
                duplicate_indices.append(j)
    # Get unique duplicates
    duplicate_indices = list(set(duplicate_indices))
    unique_idx_resp = [idx for idx in range(len(list_feat)) if idx not in duplicate_indices]
    return unique_idx_resp

def get_deduplicate_output_one_task(task):
    """dedup based on output grid"""
    dic_unique_output = {}
    for result in task:
        list_out = (
            result["predicted_train_output"] +
            result["predicted_test_output"]
            )
        list_out_str = str(list_out)
        if list_out_str in dic_unique_output:
            dic_unique_output[list_out_str] = result
    return list(dic_unique_output.values())

def rm_exact_code(list_resp):
    """remove exact duplicate code"""
    duplicate_indices = []
    list_codes = [resp["code"].strip() for resp in list_resp]
    # Only check each pair once (upper triangle)
    list_unique_code=[]
    for i in range(len(list_codes)):
        if list_codes[i] not in list_unique_code:
            list_unique_code.append(list_codes[i])
        else:
            duplicate_indices.append(i)

    # Get unique duplicates
    duplicate_indices = list(set(duplicate_indices))
    unique_idx_resp = [idx for idx in range(len(list_codes)) if idx not in duplicate_indices]
    list_resp = [list_resp[i] for i in unique_idx_resp]
    return list_resp

def filter(list_resp,list_embed):
    filtered_list_resp = []
    filtered_list_embed = []
    # deduplicate correct responses on similarity 
    assert len(list_resp) == len(list_embed), f"The length of responses and embeddings must match. Got {len(list_resp)} and {len(list_embed)}."
    list_is_correct = [id_resp for id_resp,resp in enumerate(list_resp) if all(resp["correct_test_input"] + resp["correct_train_input"])]
    list_resp_correct = [list_resp[id_resp] for id_resp in list_is_correct]
    list_embed_correct = [list_embed[id_resp] for id_resp in list_is_correct]
    list_id_keep_correct = rm_duplicate_resps(list_embed_correct, list_resp_correct, threshold=0.9)
    
    filtered_list_resp += [list_resp_correct[id_resp] for id_resp in list_id_keep_correct]
    filtered_list_embed += [list_embed_correct[id_resp] for id_resp in list_id_keep_correct]

    # deduplicate incorrect responses on outputs
    list_is_incorrect = [id_resp for id_resp,resp in enumerate(list_resp) if not all(resp["correct_test_input"] + resp["correct_train_input"])]
    list_resp_incorrect = [list_resp[id_resp] for id_resp in list_is_incorrect]

    list_id_keep_incorrect = get_deduplicate_output_one_task(list_resp_incorrect)
    list_embed_incorrect = [list_embed[id_resp] for id_resp in list_is_incorrect]

    filtered_list_resp += [list_resp_incorrect[id_resp] for id_resp in list_id_keep_incorrect]
    filtered_list_embed += [list_embed_incorrect[id_resp] for id_resp in list_id_keep_incorrect]

    return filtered_list_resp, filtered_list_embed

def dedup_incorrect_task_output(list_resp):
    """dedup based on output grid"""
    # first remove exact code duplicates
    list_is_incorrect = [id_resp for id_resp,resp in enumerate(list_resp) if not all(resp["correct_test_input"] + resp["correct_train_input"])]
    list_resp_incorrect = [list_resp[id_resp] for id_resp in list_is_incorrect]
    list_id_keep_incorrect = get_deduplicate_output_one_task(list_resp_incorrect)
    list_resp_incorrect = [list_resp_incorrect[id_resp] for id_resp in list_id_keep_incorrect]
    list_correct = [resp for resp in list_resp if all(resp["correct_test_input"] + resp["correct_train_input"])]
    list_all_resp = list_correct + list_resp_incorrect
    return list_all_resp

all_resp = {k: [] for k in list_k}
all_embed = {k: [] for k in list_k}

list_path = [getattr(args, f"path_{i}") for i in range(20) if getattr(args, f"path_{i}") != ""]
count_unique_dedup = 0
for path_file in tqdm(list_path):
    data_i = {}
    try:
        print(f"try opening {path_file}")
        with open(path_file, "rb") as f:
            res = merge_results(pickle.load(f))
        for k in list_k:
            all_resp[k] += res[k]
    except Exception as e:
        print(f"Error processing path: {path_file}: {e}")
        continue
count_unique_dedup = 0


for k in tqdm(list_k):

    list_resp = all_resp[k]
    if len(list_resp) == 0:
        print(f"No responses for {k}, skipping.")
        continue
    # compute embeddings for the responses
    list_resp = rm_exact_code(list_resp)
    list_resp = dedup_incorrect_task_output(list_resp)
    list_code = [resp["code"].strip() for resp in list_resp]
    list_feat = model.encode(list_code,batch_size=1,transfer_to_cpu=True,convert_to_numpy=True)#encode_in_batches(model, list_code)
    # Normalize the vectors (required for cosine similarity)
    normalized_embeddings = list_feat / np.linalg.norm(list_feat, axis=1, keepdims=True)
    all_embed[k] = normalized_embeddings
    resp_f, emb_f = filter(list_resp, all_embed[k])
    all_resp[k] = resp_f
    all_embed[k] = np.array(emb_f)
    count_unique_dedup += len(resp_f)
print(f"final : {count_unique_dedup} unique deduplicated responses")

with open(args.path_save, "wb") as f:
    pickle.dump(to_formated_list(all_resp), f)

