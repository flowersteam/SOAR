import numpy as np
import pickle
from soar.llm_utils import merge_results,get_dic_task_solved, to_formated_list
from soar.llm_utils import get_number_of_solved_tasks_bis, get_info
from tqdm import tqdm
import argparse
import os
import copy
from soar.preprocess import get_dataset
from soar.post_process.filtering import filter_invalid_grid, rm_duplicate_resps
parser = argparse.ArgumentParser()
parser.add_argument("--base_path", type=str, help="Path to soar git repo")
parser.add_argument("--path_embed_model", type=str, default="/home/flowers/work/hf/CodeRankEmbed")
parser.add_argument("--split", type=str, default="train", help="")
parser.add_argument("--N_sample_task", type=int, default=50)
parser.add_argument("--path_folder_refinement", type=str, default="/home/flowers/work/arc/soar/save_results/full_pipeline_scratch/Qwen2.5-Coder-14B-Instruct/gen-0/refinement/")
parser.add_argument("--sample_mode", type=str, default="diverse", help="[diverse, uniform, sample_close_1, sample_close_2, her_synth,greedy_div_ref]")
parser.add_argument("--use_prev_gen",action=argparse.BooleanOptionalAction ,default=False,help="merge solution of this generation with previous generation")
parser.add_argument("--arc2", action=argparse.BooleanOptionalAction, help="arc 2",default=False)
parser.add_argument("--dedup",action=argparse.BooleanOptionalAction ,default=False,help="deduplication of the responses")
parser.add_argument("--use_gt",action=argparse.BooleanOptionalAction ,default=False,help="force the use of gt for to upload to hf")

args = parser.parse_args()
train_data, val_data, test_data = get_dataset(args.base_path,arc_2=args.arc2)
data2test=copy.deepcopy(train_data)
data2test.update(val_data)

if args.dedup:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.path_embed_model, trust_remote_code=True)

# use path join instead
path_save = os.path.join(args.path_folder_refinement, f"data4train-{args.split}-n{args.N_sample_task}.pkl")
if args.sample_mode != "diverse":
    path_save = os.path.join(args.path_folder_refinement, f"data4train-{args.split}-n{args.N_sample_task}-{args.sample_mode}.pkl")

path_save_only_correct = os.path.join(args.path_folder_refinement, f"data4train_{args.split}_only_correct.pkl")
def get_category_correctness(response,split="train"): 
    cat = np.mean(response["correct_train_input"]) 
    if cat==0: # Zero correct response 
        return "zero"
    elif cat<=0.34: # Low correctness ]0,0.34] 
        return "low" 
    elif cat <0.98: # Medium correctness ]0.34,0.98[ 
        return "medium" 
    else: # High correctness ]0.98,1] 
        if split=="train": 
            if np.mean(response["correct_test_input"])!=1: 
                return "max_zero"
            else:
                return "max"
        else:
            return "max" 
        
def sampling_given_initial_correctness(data, N,split="train"):
    """only used to sample data on train data because we need reliable correctness metric"""
    new_data = {}
    for k, v in data.items():
        # Group responses by correctness category
        responses_by_correctness = {
            "zero": [],
            "low": [],
            "medium": [],
            "max_zero": [],
            "max": []
        }
        
        # Categorize responses
        # list_unique_id_2_idx = {v[id_resp]["unique_id"]: id_resp  for id_resp in range(len(v))}
        for response in v:
            # only sample from correct refined responses 
            correct_to_use = response["correct_train_input"]
            if split == "train" :
                correct_to_use = response["correct_train_input"] + response["correct_test_input"]
            if all(correct_to_use):
                if response["type"] == "refined" :
                    # parent_id  = response["parents"][0]
                    
                    correctness = get_category_correctness(response["parent_response"],split=split)
                    try:
                        responses_by_correctness[correctness].append(response)
                    except Exception as e:
                        print("correctness:", correctness)
                        print("response:", response["parent_response"])
                        raise e
        # Filter out the 'max' category get only parent that have incorrect response
        valid_responses = {cat: resp for cat, resp in responses_by_correctness.items() 
                         if resp and cat != 'max'}
        
        
        if not valid_responses:
            continue
            
        sampled_responses = []
        remaining_samples = N
        count_responses = sum(len(responses) for responses in valid_responses.values())
        if args.dedup and count_responses > N:
            for cat, responses in valid_responses.items():
                # Remove duplicates based on cosine similarity
                if args.dedup and len(responses) > 1:
                    responses = [resp for resp in responses if len(resp["code"].strip()) < 8000]
                    responses = rm_exact_code(responses)
                    responses = rm_duplicate_resps(model, responses)
                    
                    valid_responses[cat] = responses
        # Create a pool of available responses for each category
        available_responses = {cat: responses.copy() for cat, responses in valid_responses.items()}
        
        # First pass: try to sample evenly from each category
        samples_per_category = N // len(valid_responses)
        for cat, responses in valid_responses.items():
            # Sample without replacement
            n_samples = min(len(responses), samples_per_category)
            if n_samples > 0:
                indices = np.random.choice(len(responses), n_samples, replace=False)
                category_samples = [responses[i] for i in indices]
                sampled_responses.extend(category_samples)
                remaining_samples -= len(category_samples)
                
                # Remove sampled responses from available pool
                available_responses[cat] = [r for i, r in enumerate(responses) 
                                         if i not in indices]
        
        # Second pass: fill remaining samples from categories that still have examples
        while remaining_samples > 0:
            # Get categories that still have samples
            available_categories = [cat for cat, resp in available_responses.items() 
                                 if resp]
            if not available_categories:
                # If no more unique samples available, fall back to sampling with replacement
                all_samples = [sample for samples in valid_responses.values() 
                             for sample in samples]
                if all_samples:
                    additional_samples = np.random.choice(all_samples, 
                                                        min(remaining_samples,len(all_samples)), 
                                                        replace=False).tolist()
                    sampled_responses.extend(additional_samples)
                break
                
            # Randomly choose a category to sample from
            chosen_category = np.random.choice(available_categories)
            responses = available_responses[chosen_category]
            
            # Sample one example
            idx = np.random.choice(len(responses))
            sampled_responses.append(responses[idx])
            # Remove sampled response
            available_responses[chosen_category].pop(idx)
            remaining_samples -= 1
            
        new_data[k] = sampled_responses
        
    return new_data

def sampling_uniform(data, N):
    new_data = {}
    for k, v in data.items():
        if len(v) > N:
            new_data[k] = np.random.choice(v, min(len(v),N), replace=False).tolist()
        else:
            new_data[k] = v
    return new_data


def get_set_io_pairs(function, use_synth_test=True):
    if use_synth_test:
        io_pairs = function["predicted_train_output"] + function["predicted_test_output"]
    else:
        io_pairs = function["predicted_train_output"]
    set_i = set(str(out_ii) for out_ii in io_pairs)
    return set_i

def sample_close_1(data, N, max_attempts=5000):
    """
    For each key in data, randomly sample pairs of functions meeting the conditions:
    - They share at least one common input-output pair
    - They are not identical
    
    Parameters:
    - N: maximum number of function pairs to sample per key
    - max_attempts: maximum number of random sampling attempts before moving on
    """
    new_data = {}
    for k, function_list in tqdm(data.items()):
        if len(function_list) < 2:  # Skip if not enough functions to form pairs
            new_data[k] = []
            continue
            
        candidate_pairs = []
        attempts = 0
        
        # Keep sampling until we find N pairs or reach max_attempts
        while len(candidate_pairs) < N and attempts < max_attempts:
            # Randomly sample two different indices
            i, j = np.random.choice(len(function_list), size=2, replace=False)
            
            # Convert outputs to sets
            out_i = function_list[i]["predicted_train_output"] + function_list[i]["predicted_test_output"]
            out_j = function_list[j]["predicted_train_output"] + function_list[j]["predicted_test_output"]
            set_i = set(str(out_ii) for out_ii in out_i)
            set_j = set(str(out_jj) for out_jj in out_j)
            
            # Check conditions
            common = set_i.intersection(set_j)
            if len(common) >= 1 and len(set_j)-len(common) != 0:
                pair = (function_list[i], function_list[j])
                if pair not in candidate_pairs:  # Avoid duplicates
                    candidate_pairs.append(pair)
            attempts += 1
        
        # Format the sampled pairs
        sampled_pairs_formated = []
        for candidate in candidate_pairs:
            idx_child = np.random.choice([0,1])
            idx_parent = 1 - idx_child
            new_pair = copy.deepcopy(candidate[idx_child])
            new_pair["parent_response"] = candidate[idx_parent]
            sampled_pairs_formated.append(new_pair)
        
        new_data[k] = sampled_pairs_formated

    return new_data


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

def sample_close_2(data, N, max_attempts=5000):
    """
    For each key in data, randomly sample pairs of functions meeting the conditions:
    - They share at least one common input-output pair
    - They are not identical
    
    Parameters:
    - N: maximum number of function pairs to sample per key
    - max_attempts: maximum number of random sampling attempts before moving on
    """
    new_data = {}
    for k, function_list in tqdm(data.items()):
        if len(function_list) < 2:  # Skip if not enough functions to form pairs
            new_data[k] = []
            continue
            
        candidate_pairs = []
        attempts = 0
        
        # Keep sampling until we find N pairs or reach max_attempts
        while len(candidate_pairs) < N and attempts < max_attempts:
            # Randomly sample two different indices
            i, j = np.random.choice(len(function_list), size=2, replace=False)
            
            # Convert outputs to sets
            out_i = function_list[i]["predicted_train_output"] + function_list[i]["predicted_test_output"]
            out_j = function_list[j]["predicted_train_output"] + function_list[j]["predicted_test_output"]
            set_i = set(str(out_ii) for out_ii in out_i)
            set_j = set(str(out_jj) for out_jj in out_j)
            
            # Check conditions
            common = set_i.intersection(set_j)
            if len(common) >= 2 and len(set_j)-len(common) != 0:
                pair = (function_list[i], function_list[j])
                if pair not in candidate_pairs:  # Avoid duplicates
                    candidate_pairs.append(pair)
            
            attempts += 1
        
        # Format the sampled pairs
        sampled_pairs_formated = []
        for candidate in candidate_pairs:
            idx_child = np.random.choice([0,1])
            idx_parent = 1 - idx_child
            new_pair = copy.deepcopy(candidate[idx_child])
            new_pair["parent_response"] = candidate[idx_parent]
            sampled_pairs_formated.append(new_pair)
        
        new_data[k] = sampled_pairs_formated

    return new_data


def sample_her_synth(data, N):
    new_data = {}
    for k, function_list in data.items():
        list_cand = []
        for function in function_list:
            set_child = get_set_io_pairs(function)
            set_parent = get_set_io_pairs(function["parent_response"])
            common = set_child.intersection(set_parent)
            if len(common) >= 1  and len(set_parent)-len(common)!=0:
                list_cand.append(function)
        if len(list_cand)!=0:
            if len(list_cand) > N:
                indices = np.random.choice(len(list_cand), N, replace=False)
                sampled_pairs = [list_cand[idx] for idx in indices]
            else:
                sampled_pairs = list_cand

            new_data[k] = sampled_pairs
        else:
            new_data[k] = []
    return new_data


def diverse_greedy_repair(data,N):
    dic_sample = {}
    N_greedy = N//2
    N_diverse = N//2
    min_number_sample = 0
    for k, function_list in data.items():
        list_cand = []
        for function in function_list:
            set_child = get_set_io_pairs(function,False)
            set_parent = get_set_io_pairs(function["parent_response"],False)
            common = set_child.intersection(set_parent)
            if len(set_parent)-len(common)!=0:
                list_cand.append(function)
        if len(list_cand)!=0:
            dic_resp_sorted  = sorted(data[k], key=lambda x: np.mean(x["correct_train_input"]), reverse=True)
            dic_sample[k] = dic_resp_sorted[:N_greedy]

            list_diverse=[]
            
            # take example wo correct train input
            dic_other_resp = dic_resp_sorted[N_greedy:]
            for resp in dic_other_resp:
                if not any(resp["correct_train_input"]):
                    list_diverse.append(resp)
            
            if len(list_diverse)<=N_diverse:
                # in case we have not enough example without correct and example wo correct train input 
                for resp in dic_other_resp:
                    if np.sum(resp["correct_train_input"])==1:
                        list_diverse.append(resp)
            dic_sample[k] += np.random.choice(list_diverse, min(N_diverse,len(list_diverse)), replace=False).tolist()
            min_number_sample = min(min_number_sample,len(dic_sample[k]))

        else:
            dic_sample[k] = []
    print("min_number_sample: ",min_number_sample)
    return dic_sample
# TODO: later from code that have != output sample code that are similar but not too much (maybe around 0.8-0.85)?




def clean_repair_data(data,split="train", keep_correct=True):
    data_merged = merge_results(data)
    data_merged = filter_invalid_grid(data_merged,data2test)

    n_resp_max=0
    for k, v in data_merged.items():
        v_filtered = [f for f in v if f["type"] == "refined"]
        n_resp_max = max(n_resp_max,len(v_filtered))
    print("n resp max = ",n_resp_max)
    n_resp_max=0
    data_clean = {}
    for k, v in data_merged.items():
        list_unique_id_2_idx = {v[id_resp]["unique_id"]: id_resp  for id_resp in range(len(v))}

        list_resp = []
        for response in v:
            # only sample from correct refined responses 
            if split == "train":
                flag_correct = all(response["correct_train_input"] + response["correct_test_input"])
            else:
                flag_correct = all(response["correct_train_input"])
            if keep_correct:
                if not flag_correct:
                    continue
            if response["type"] == "refined" :
                parent_is_set = False
                if "parent_response" in response:
                    parent_response = response["parent_response"]
                    # check if parent_response is full: (code,predicted_test_output,predicted_train_output) 
                    if "code" in parent_response and "predicted_test_output" in parent_response and "predicted_train_output" in parent_response:
                        parent_is_set = True
                if parent_is_set:
                    list_resp.append(response)
                else:
                    parent_id  = response["parents"][0]
                    if parent_id in list_unique_id_2_idx:
                        idx_parent = list_unique_id_2_idx[parent_id]
                        parent = v[idx_parent]
                        response["parent_response"] = parent
                        list_resp.append(response)  
        data_clean[k] = list_resp
        n_resp_max = max(n_resp_max,len(list_resp))
    print("n resp max = ",n_resp_max)
    return [{"dict_response":data_clean}]

def get_previous_paths(path):
    """
    .../soar/save_results/full_pipeline_scratch/Qwen2.5-Coder-32B-Instruct/gen-1/solution/
    -> [.../soar/save_results/full_pipeline_scratch/Qwen2.5-Coder-32B-Instruct/gen-0/solution/, .../soar/save_results/full_pipeline_scratch/Qwen2.5-Coder-32B-Instruct/gen-1/solution/]
    """
    # Find the pattern 'gen-X' in the path
    
    import re
    match = re.search(r'gen-(\d+)/', path)
    if not match:
        return [path]
    if not args.use_prev_gen:
        print("use_prev_gen is False, return only current path")
        return [path]
    current_gen = int(match.group(1))
    base_path = path[:match.start()]
    suffix = path[match.end():]
    
    # Generate all paths from gen-0 to current gen - 1
    paths = []
    for i in range(current_gen):

        new_path = f"{base_path}gen-{i}/{suffix}"
        paths.append(new_path)
    return paths

# load files
files = os.listdir(args.path_folder_refinement)
list_files_keep=[]
for i in files:
    if (not "data4train" in i) and ".pkl" in i and i.startswith(args.split+"_"):
        list_files_keep.append(i)


keep_correct = True
if args.sample_mode != "diverse":
    keep_correct = False

all_results = []
print("files to load:")
print(list_files_keep)
for filename in tqdm(list_files_keep):
    file2open = args.path_folder_refinement + filename
    # check if the file exists
    if not os.path.exists(file2open):
        raise ValueError(f"File {file2open} does not exist")
    try:
        with open(file2open, 'rb') as f:
            print('--------')
            print("loading",filename)
            data = pickle.load(f)
            data = to_formated_list(data)
            split = args.split 
            if args.use_gt:
                split = "train"
            all_results += clean_repair_data(data,split=split,keep_correct=keep_correct)
    except:
        print(f"error: can't load {file2open}")



with open(path_save_only_correct, "wb") as f:
    pickle.dump(all_results, f)


list_prev_path = get_previous_paths(path_save_only_correct)
print("found previous paths:")
for prev_path in list_prev_path:
    print("- ",prev_path)
    try:
        with open(prev_path, "rb") as f:

            all_results += to_formated_list(pickle.load(f))
    except:
        print(f"error: can't load {prev_path}")
all_results = merge_results(all_results)
list_len = []
for k,v in all_results.items():
    list_len.append(len(v))
print("max repair = ",max(list_len))

get_info(get_number_of_solved_tasks_bis(all_results,mv=False))

print("sampling mode:",args.sample_mode)
if args.sample_mode == "uniform":
    data_train = sampling_uniform(all_results, args.N_sample_task)
elif args.sample_mode == "diverse":
    split = args.split
    if args.use_gt:
        split = "train"
    data_train = sampling_given_initial_correctness(all_results, args.N_sample_task, split = split)
elif args.sample_mode == "sample_close_1":
    data_train = sample_close_1(all_results, args.N_sample_task)
elif args.sample_mode == "sample_close_2":
    data_train = sample_close_2(all_results, args.N_sample_task)
elif args.sample_mode == "her_synth":
    data_train = sample_her_synth(all_results, args.N_sample_task)
elif args.sample_mode == "greedy_div_ref":
    data_train = diverse_greedy_repair(all_results, args.N_sample_task)
else:
    raise ValueError(f"Unknown sample mode: {args.sample_mode}")

get_info(get_number_of_solved_tasks_bis(data_train,mv=False))

with open(path_save, "wb") as f:
    pickle.dump(data_train, f)


get_info(get_number_of_solved_tasks_bis(data_train,mv=True))
