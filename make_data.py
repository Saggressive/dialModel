import json
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext", use_fast=True)
def main(load_path,save_path):
    proc_lines=[]
    with open(load_path) as f:
        data=json.load(f)
        for line in data:
            text=line[0]
            text="".join(text)
            text=tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            if len(text)<=2:
                continue 
            act=line[-2]
            intent,domain,solt,value = set(),set(),set(),set()
            for item in act:
                if item[0]!='' and item[0]!='none':
                    intent.add(item[0])
                if item[1]!='' and item[1]!='none':
                    domain.add(item[1])
                if item[2]!='' and item[2]!='none':
                    solt.add(item[2])
                if item[3]!='' and item[3]!='none':
                    value.add(item[3])
            proc_line = {"text":text,"intent":list(intent), \
                            "domain":list(domain),"solt":list(solt),"value":list(value)}
            proc_lines.append(proc_line)
    with open(save_path,"w",encoding='utf-8') as f:
        for proc_line in proc_lines:
            f.write(json.dumps(proc_line) + '\n')

if __name__=="__main__":
    load_path = "/mmu_nlp/wuxing/suzhenpeng/dialModel/crosswoz/train_data.json"
    save_path = "/mmu_nlp/wuxing/suzhenpeng/dialModel/crosswoz/proc_data/train_data.json"
    main(load_path,save_path)
