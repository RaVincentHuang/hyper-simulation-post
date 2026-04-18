import json
from hyper_simulation.llm.prompt.vmdit import rewrite
from hyper_simulation.llm.chat_completion import get_invoke_prompt
from tqdm import tqdm
def read_evi(file_path):
    query=[]
    ctxs=[]
    with open(file_path, "r") as fin:
        for _, example in enumerate(fin):
            example = json.loads(example)
            query.append(example["question"])
            ctxs.append(example["ctxs"][0]['title']+example["ctxs"][0]['text'])
    return query,ctxs
def rewrite_with_llm(query):
    res = get_invoke_prompt(
        {"query": query},
        rewrite,
        top_p=0.7,
            temperature=0.9,
    )
    return res
def get_new_query(q,ct):
    ins = "Give a question [{q}] and its possible answering passages [{ct}].".format(q=q,ct=ct)
    return ins
def get_query(file_path):
    query, ctxs= read_evi(file_path)
    new_q=[]
    error_q=[]
    for i in tqdm(range(len(query)), desc="Processing", unit="example"):
        q=query[i]
        ct=ctxs[i]
        nq=get_new_query(q,ct)
        new_q.append(nq)
    return new_q, error_q
def calc_rewrite(file_path, out_path):
    new_q, _ = get_query(file_path)
    json_data = json.dumps(new_q)
    with open(out_path, "w") as file:
        file.write(json_data)
if __name__=='__main__':
    file_path= "../../data/eval_data/popqa_longtail.jsonl"
    new_q,error_q=get_query(file_path)
    json_data_1=json.dumps(new_q)
    json_data_2 = json.dumps(error_q)
    with open("new_query/popqa_q_0.jsonl", "w") as file:
        file.write(json_data_1)