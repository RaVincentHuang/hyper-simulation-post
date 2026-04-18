import tqdm
import torch
import json
import matplotlib.pyplot as plt
from hyper_simulation.llm.prompt.vmdit import same_statements, same_sentences
from hyper_simulation.llm.chat_completion import get_invoke_prompt
from tqdm import tqdm
def get_ctxs_em(file_path):
    ctxs=[]
    query=[]
    with open(file_path, "r") as fin:
        for k, example in enumerate(fin):
            example = json.loads(example)
            cts=[]
            for ct in example["ctxs"]:
                if "id" in ct:
                    cts.append(ct)
            ctxs.append(cts)
            query.append(example["question"])
    return ctxs,query
def same_statements_with_llm(evi, query):
    res = get_invoke_prompt({
        "evi" : evi,
        "query" : query,
    }, same_statements, top_p=0.7, temperature=0.9, )
    return res
def same_sentences_with_llm(evi, query):
    res = get_invoke_prompt({
        "evi" : evi,
        "query" : query,
    }, same_sentences, top_p=0.7, temperature=0.9, )
    return res
def context_relation(t_context_result,f_context_result):
    r=[]
    for idx in t_context_result['id']:
        for idy in f_context_result['id']:
            r.append((idx,idy))
    return r
def get_enquery(file_path):
    ctxs,query=get_ctxs_em(file_path)
    unsim_s=[]
    error_d=[]
    for i in tqdm(range(len(ctxs)), desc="get enquery:"):
        t_context_result={"id":[],"result":[]}
        f_context_result={"id":[],"result":[]}
        evidences = ["[{}] ".format(i+1) + ctx["title"]+"\n" + ctx["text"] for i, ctx in enumerate(ctxs[i])]
        q=query[i]
        for j in range(len(ctxs[i])):
            try:
                r= same_statements_with_llm(evidences[j],q)
            except:
                error_d.append((i,j))
                continue
            if "True" in r:
                t_context_result['result'].append(r)
                t_context_result['id'].append(int(ctxs[i][j]['id']))
            else:
                f_context_result['result'].append(r)
                f_context_result['id'].append(int(ctxs[i][j]['id']))
        unsim=context_relation(t_context_result,f_context_result)
        unsim_s.append(unsim)
    return unsim_s,error_d
def calc_relations(file_path, rel_path):
    relation_c,error_d = get_enquery(file_path)
    print(len(relation_c))
    json_data = json.dumps(relation_c)
    json_data_1 = json.dumps(error_d)
    with open(rel_path, "w") as file:
        file.write(json_data)
if __name__=='__main__':
    file_path = "../retr_result/L2/popqa_longtail.jsonl"
    relation_c, error_d = get_enquery(file_path)
    print(len(relation_c))
    json_data = json.dumps(relation_c)
    json_data_1 = json.dumps(error_d)
    with open("relation_context/L2/relation_context_popqa_0_50.json", "w") as file:
        file.write(json_data)
    with open("relation_context/L2/relation_context_popqa_error.json", "w") as file:
        file.write(json_data_1)