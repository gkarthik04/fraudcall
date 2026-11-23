import json
cells = [
("markdown", "# T-rob: distilled multilingual transformer on recognised transcripts (Colab, GPU)\nRun on a free Colab GPU runtime. **Before running**, zip `data/manifest/manifest.jsonl` and `data/features/vocab.json`... only the manifest is needed here; upload `manifest.jsonl` to Colab. **Outputs to copy back:** `results/runs/T-rob/s{seed}/preds_val.csv` and `preds_test.csv` into the same path in the local project, then run `python -m src.eval.run_eval`."),
("code", "!pip -q install transformers torch scikit-learn pandas\nfrom google.colab import files\nfiles.upload()  # choose manifest.jsonl"),
("code", """import json, numpy as np, pandas as pd, torch, torch.nn as nn, os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
rows=[json.loads(l) for l in open('manifest.jsonl')]
NAME='distilbert/distilbert-base-multilingual-cased'  # Apache-2.0
tok=AutoTokenizer.from_pretrained(NAME)
def samples(split, conds, sets=(1,)):
    out=[]
    for r in rows:
        if r['split']==split and r['tts_set'] in sets and (conds is None or r['cond'] in conds):
            for k,s in enumerate(r['steps']): out.append((r['tts_set'],r['cond'],r['cid'],k,s['t'],r['label'],s['asr']))
    return out
def enc(batch): return tok([b[6] or ' ' for b in batch], truncation=True, max_length=256, padding=True, return_tensors='pt')
def predict(model, data, bs=64):
    model.eval(); sc=[]
    with torch.no_grad():
        for i in range(0,len(data),bs):
            b=data[i:i+bs]; x={k:v.cuda() for k,v in enc(b).items()}
            sc+=torch.sigmoid(model(**x).logits[:,1]-model(**x).logits[:,0]).cpu().tolist()
    return sc
from sklearn.metrics import f1_score
for seed in (0,1,2):
    torch.manual_seed(seed); np.random.seed(seed)
    model=AutoModelForSequenceClassification.from_pretrained(NAME,num_labels=2).cuda()
    train=samples('train',['clean','rand0','rand1']); val=samples('val',['clean','rand0','rand1'])
    opt=torch.optim.AdamW(model.parameters(),lr=3e-5); best=-1
    for ep in range(3):
        model.train(); idx=np.random.permutation(len(train))
        for i in range(0,len(idx),32):
            b=[train[j] for j in idx[i:i+32]]; x={k:v.cuda() for k,v in enc(b).items()}
            loss=nn.functional.cross_entropy(model(**x).logits, torch.tensor([t[5] for t in b]).cuda())
            opt.zero_grad(); loss.backward(); opt.step()
        f1=f1_score([v[5] for v in val], np.array(predict(model,val))>0.5)   # validation only
        print(seed,ep,'val f1',f1)
        if f1>best: best=f1; torch.save(model.state_dict(),'best.pt')
    model.load_state_dict(torch.load('best.pt'))
    out=f'results/runs/T-rob/s{seed}'; os.makedirs(out,exist_ok=True)
    for part,data in (('val',samples('val',None)),('test',samples('test',None,(1,2)))):
        sc=predict(model,data)
        pd.DataFrame([dict(set=d[0],cond=d[1],cid=d[2],step=d[3],t=d[4],label=d[5],score=s) for d,s in zip(data,sc)]).to_csv(f'{out}/preds_{part}.csv',index=False)
    json.dump({'name':'T-rob','seed':seed,'exp':{'train_conds':['clean','rand0','rand1']}},open(f'{out}/summary.json','w'))
!zip -r T-rob-results.zip results
files.download('T-rob-results.zip')"""),
]
nb={"cells":[{"cell_type":t,"metadata":{},"source":s.splitlines(True),**({"outputs":[],"execution_count":None} if t=="code" else {})} for t,s in cells],
    "metadata":{"accelerator":"GPU","kernelspec":{"name":"python3","display_name":"Python 3"}},"nbformat":4,"nbformat_minor":5}
json.dump(nb,open("notebooks/T-rob_colab.ipynb","w"),indent=1)
