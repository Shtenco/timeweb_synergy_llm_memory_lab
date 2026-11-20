#!/usr/bin/env python3
"""HIAI V6: audited retrieval + optional REAL Ollama LLM generation.
Does not simulate LLM responses; failures abort instead of being scored as misses.
Synthetic examples are demo fixtures, NOT independent external empirical validation.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, math, os, platform, random, re, statistics, sys, time, urllib.error, urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize
import psutil
from corpora import REGISTRY, external_jsonl

TOKEN = re.compile(r'[\w-]+',re.UNICODE)
HERE=Path(__file__).resolve().parents[1]

def toks(s):return TOKEN.findall(str(s).casefold())
def norm(s):return ' '.join(toks(s))
def safe_f1(pred,gold):
    p=toks(pred);g=toks(gold)
    common=sum((collections.Counter(p)&collections.Counter(g)).values())
    if not p and not g:return 1.
    if not p or not g or not common:return 0.
    precision=common/len(p); recall=common/len(g)
    return 2*precision*recall/(precision+recall)

def http_json(url,payload=None,timeout=120):
    data=None if payload is None else json.dumps(payload,ensure_ascii=False).encode('utf-8')
    r=urllib.request.Request(url,data=data,headers={'Content-Type':'application/json'},method='POST' if payload is not None else 'GET')
    with urllib.request.urlopen(r,timeout=timeout) as f:return json.load(f)

class Ollama:
    def __init__(self,base='http://127.0.0.1:11434',timeout=120):
        self.base=base.rstrip('/');self.timeout=timeout
    def available(self):
        j=http_json(self.base+'/api/tags',timeout=10)
        return [x['name'] for x in j.get('models',[])]
    def check(self,names):
        names=list(dict.fromkeys(x for x in names if x))
        have=self.available()
        missing=[x for x in names if x not in have and not any(y.split(':')[0]==x for y in have)]
        if missing:raise RuntimeError(f'Unavailable Ollama models: {missing}. Have {have}. Please: ollama pull NAME')
    def embed(self,texts,model):
        # Model-generated dense embeddings; no fallback to LSA disguised as pretrained semantic embedding.
        out=[]
        for i in range(0,len(texts),32):
            j=http_json(self.base+'/api/embed',{'model':model,'input':texts[i:i+32],'truncate':False},self.timeout)
            out.extend(j['embeddings'])
        if len(out)!=len(texts): raise RuntimeError('embedding batch size mismatch')
        return np.asarray(out,dtype=np.float32)
    def generate(self,model,prompt,num_ctx=4096,max_tokens=64):
        j=http_json(self.base+'/api/generate',{'model':model,'prompt':prompt,'stream':False,'think':False,
                  'options':{'temperature':0,'num_ctx':num_ctx,'num_predict':max_tokens},'keep_alive':'2m'},self.timeout)
        if not j.get('done'):raise RuntimeError('Ollama incomplete response')
        return j

class Index:
    def __init__(self,docs,enable_lsa=True,ollama=None,embed_model=None,lsa_dim=48):
        self.docs={d['id']:d for d in docs}
        if len(self.docs)!=len(docs):raise ValueError('Duplicate IDs')
        self.ids=list(self.docs); self.n=len(docs)
        self.terms={k:collections.Counter(toks(v['text'])) for k,v in self.docs.items()}
        self.df=collections.Counter()
        for c in self.terms.values():self.df.update(c.keys())
        self.avglen=sum(sum(c.values()) for c in self.terms.values())/max(1,self.n)
        self.vectorizer=None;self.tfidf_matrix=None;self.lsa=None;self.doc_lsa=None;self.ollama=ollama;self.embed_model=embed_model;self.doc_embeddings=None
        self.index_seconds={};
        t=time.perf_counter()
        if enable_lsa:
            self.vectorizer=TfidfVectorizer(token_pattern=r'(?u)\b[\w-]+\b',lowercase=True,ngram_range=(1,2),sublinear_tf=True)
            X=self.vectorizer.fit_transform([self.docs[x]['text'] for x in self.ids])
            self.tfidf_matrix=X
            dims=min(lsa_dim,X.shape[0]-1,X.shape[1]-1)
            if dims >= 2:
                self.lsa=TruncatedSVD(n_components=dims,n_iter=5,random_state=17)
                self.doc_lsa=sk_normalize(self.lsa.fit_transform(X),norm='l2')
            else:
                self.doc_lsa=None
        self.index_seconds['lsa']=time.perf_counter()-t
        if ollama and embed_model:
            t=time.perf_counter()
            arr=ollama.embed([self.docs[x]['text'] for x in self.ids],embed_model)
            self.doc_embeddings=sk_normalize(arr,norm='l2')
            self.index_seconds['dense_ollama']=time.perf_counter()-t

    def bm25_scores(self,q):
        ts=toks(q); scores=np.zeros(self.n)
        for n,k in enumerate(self.ids):
            tf=self.terms[k];dl=sum(tf.values())
            for w in ts:
                f=tf.get(w,0);df=self.df.get(w,0)
                if f:
                    idf=math.log(1+(self.n-df+.5)/(df+.5))
                    scores[n]+=idf*f*2.2/(f+1.2*(.25+.75*dl/self.avglen))
        return scores
    def lsa_scores(self,q):
        if self.doc_lsa is None:raise ValueError('LSA unavailable')
        query=self.vectorizer.transform([q]);v=sk_normalize(self.lsa.transform(query))
        return (self.doc_lsa@v.T).reshape(-1)
    def dense_scores(self,q):
        if self.doc_embeddings is None:raise ValueError('real embedding model missing; pass --embed-model and run Ollama')
        query=sk_normalize(self.ollama.embed([q],self.embed_model))
        return (self.doc_embeddings@query.T).reshape(-1)
    def rank(self,scores):
        # Tie-break is stable by ID, not by accidental shuffled document order.
        return [self.ids[i] for i in sorted(range(self.n),key=lambda i:(-float(scores[i]),self.ids[i]))]
    def retrieve(self,q,method,k):
        if method=='bm25':return self.rank(self.bm25_scores(q))[:k]
        if method=='tfidf':
            if self.vectorizer is None:raise ValueError('tfidf index unavailable')
            qv=self.vectorizer.transform([q])
            return self.rank((self.tfidf_matrix@qv.T).toarray().reshape(-1))[:k]
        if method=='dense_lsa':return self.rank(self.lsa_scores(q))[:k]
        if method=='dense_ollama':return self.rank(self.dense_scores(q))[:k]
        if method=='hybrid_lsa':return self._rrf(q,'dense_lsa',k)
        if method=='hybrid_ollama':return self._rrf(q,'dense_ollama',k)
        if method=='graph':
            # No gold labels, entity-id hints or regex-oracle are used: expand links from top BM25 hits.
            seeds=self.rank(self.bm25_scores(q))[:3]
            ret=[];visited=set()
            for seed in seeds:
                # Exhaust explicit links of the highest ranked seed before lower-ranked seeds.
                # Edges come from source documents ONLY, never from q['gold'] or answer.
                queue=[seed]
                while queue and len(ret)<k:
                    did=queue.pop(0)
                    if did in visited or did not in self.docs:continue
                    visited.add(did);ret.append(did)
                    queue.extend(self.docs[did].get('links',[]))
            return (ret+[d for d in self.rank(self.bm25_scores(q)) if d not in visited])[:k]
        raise ValueError('Unknown method '+method)
    def _rrf(self,q,second,k,constant=60):
        a=self.rank(self.bm25_scores(q))
        b=self.rank(self.lsa_scores(q) if second=='dense_lsa' else self.dense_scores(q))
        rank_a={v:i+1 for i,v in enumerate(a)};rank_b={v:i+1 for i,v in enumerate(b)}
        return sorted(self.ids,key=lambda d:(-1/(constant+rank_a[d])-1/(constant+rank_b[d]),d))[:k]

METHODS=['bm25','tfidf','dense_lsa','hybrid_lsa','graph']

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def load_corpus(which,path=None):
    if path:return external_jsonl(path)
    return REGISTRY[which](noisy=True)

def prepare(which='incidents',data=None,enable_embedding=False,client=None,embed_model=None):
    docs,questions,meta=load_corpus(which,data)
    index=Index(docs,ollama=client if enable_embedding else None,embed_model=embed_model if enable_embedding else None)
    return docs,questions,meta,index

def retrieval_suite(datasets,out,include_embedding=False,client=None,embed_model=None,k=3,external=None):
    rows=[];indexes=[]
    methods=METHODS+(['dense_ollama','hybrid_ollama'] if include_embedding else [])
    for ds in datasets:
        data=external if ds=='external' else None
        docs,questions,meta,index=prepare(ds,data,include_embedding,client,embed_model)
        indexes.append({'dataset':meta['dataset'],'n_docs':len(docs),'n_questions':len(questions),'synthetic':meta['synthetic'],
                        'index_lsa_ms':round(index.index_seconds['lsa']*1000,3),
                        'index_embed_ms':round(index.index_seconds.get('dense_ollama',0)*1000,3),
                        'doc_sha256':hashlib.sha256(json.dumps(docs,sort_keys=True,ensure_ascii=False).encode()).hexdigest()})
        for q in questions:
            gold=set(q['gold'])
            if not gold:continue
            for method in methods:
                st=time.perf_counter(); found=index.retrieve(q['question'],method,k); elapsed=(time.perf_counter()-st)*1000
                overlap=gold.intersection(found)
                ranks=[i+1 for i,d in enumerate(found) if d in gold]
                rows.append({'dataset':meta['dataset'],'synthetic':int(meta['synthetic']), 'split':q['split'],'kind':q['kind'],
                  'qid':q['id'],'method':method,'k':k,'n_gold':len(gold),'n_found':len(overlap),
                  'all_found':int(gold.issubset(found)),'gold_recall':len(overlap)/len(gold),
                  'any_hit':int(bool(overlap)),'mrr_first_gold':1/min(ranks) if ranks else 0.,
                  'retrieval_ms':round(elapsed,4),'retrieved_ids':';'.join(found),'gold_ids':';'.join(q['gold'])})
        print('RETRIEVAL',meta['dataset'],'docs=',len(docs),'qa=',len(questions),flush=True)
    summaries=[]
    groups=sorted(set((r['dataset'],r['split'],r['kind'],r['method']) for r in rows))
    for ds,split,kind,method in groups:
        rr=[r for r in rows if (r['dataset'],r['split'],r['kind'],r['method'])==(ds,split,kind,method)]
        summaries.append({'dataset':ds,'split':split,'kind':kind,'method':method,'n':len(rr),'all_gold_found':round(statistics.mean(x['all_found'] for x in rr),4),
          'recall':round(statistics.mean(x['gold_recall'] for x in rr),4),'mrr':round(statistics.mean(x['mrr_first_gold'] for x in rr),4),
          'p50_ms':round(float(np.median([x['retrieval_ms'] for x in rr])),4),
          'p95_ms':round(float(np.percentile([x['retrieval_ms'] for x in rr],95)),4)})
    write_csv(out/'retrieval_raw.csv',rows);write_csv(out/'retrieval_summary.csv',summaries)
    (out/'index_manifest.json').write_text(json.dumps(indexes,ensure_ascii=False,indent=2),encoding='utf-8')
    return rows,summaries

def prompt_for(q,docs,idlist):
    evidence='\n'.join(f'[{i}] {docs[i]["text"]}' for i in idlist)
    language='Russian' if bool(re.search('[А-Яа-яЁё]',q['question'])) else 'English'
    if language=='Russian':
        pre='Ответь строго на основании документов ниже. Если информации нет, скажи только «НЕТ ДАННЫХ». Одно короткое значение без пояснений.'
    else:pre='Answer strictly from the documents below. If unknown, reply only "NO DATA". Give one short value without explanation.'
    return f'{pre}\nДокументы / Documents:\n{evidence if evidence else "[none]"}\nВопрос / Question: {q["question"]}\nОтвет / Answer:'

def paired_ci(rows,kind='exact',a='small_bm25',b='large_bm25',repeats=2000,seed=2026):
    by=collections.defaultdict(lambda:collections.defaultdict(list))
    for r in rows:
        if r['configuration'] in {a,b} and not r.get('error'):
            by[(r['dataset'],r['qid'])][r['configuration']].append(r[kind])
    # Bootstrap over *questions*; repeated generations of the same question are dependent.
    paired=[(statistics.mean(v[a]),statistics.mean(v[b])) for v in by.values() if a in v and b in v]
    if len(paired)<2:return {'n_pairs':len(paired),'ci95':None}
    arr=np.asarray(paired,dtype=float); rng=np.random.default_rng(seed)
    obs=float(np.mean(arr[:,0]-arr[:,1])); indices=rng.integers(0,len(arr),(repeats,len(arr)))
    draws=np.mean(arr[indices,0]-arr[indices,1],axis=1)
    return {'n_pairs':len(arr),'small_minus_large':obs,'ci95':[float(x) for x in np.percentile(draws,[2.5,97.5])]}

def model_suite(dataset,out,client,small,large,embed_model=None,k=3,limit=0,repeats=1,data=None,
                hourly_usd=None,input_per_million=None,output_per_million=None,billing='none',seed=17):
    if billing=='hourly' and hourly_usd is None:raise ValueError('--hourly-usd needed')
    if billing=='token' and (input_per_million is None or output_per_million is None):raise ValueError('--input-per-million and --output-per-million needed')
    client.check([small,large]+([embed_model] if embed_model else []))
    installed=http_json(client.base+'/api/tags',timeout=10).get('models',[])
    referenced=[small,large]+([embed_model] if embed_model else [])
    installed_used={x['name']:{'digest':x.get('digest'),'size':x.get('size'), 'details':x.get('details')} for x in installed if x.get('name') in referenced}
    try: runtime_version=http_json(client.base+'/api/version',timeout=10)
    except Exception:runtime_version={'version':'unknown'}
    (out/'model_provenance.json').write_text(json.dumps({'installed':installed_used,'runtime_version':runtime_version,'requested':referenced},ensure_ascii=False,indent=2),encoding='utf-8')
    docs,qs,meta,index=prepare(dataset,data=data,enable_embedding=bool(embed_model),client=client,embed_model=embed_model)
    docs_map=index.docs
    questions=[x for x in qs if x['split']=='test']
    if limit: questions=questions[:limit]
    methods=[None,'bm25','dense_lsa','hybrid_lsa','graph']
    if embed_model:methods+=['dense_ollama','hybrid_ollama']
    configs=[(key,model,m) for key,model in (('small',small),('large',large)) for m in methods]
    rows=[];rand=random.Random(seed)
    for repeat in range(repeats):
        for q in questions:
            order=configs[:];rand.shuffle(order)
            for label,model,method in order:
                ret_start=time.perf_counter()
                found=index.retrieve(q['question'],method,k) if method else []
                retrieval_ms=(time.perf_counter()-ret_start)*1000
                prompt=prompt_for(q,docs_map,found)
                start=time.perf_counter()
                try:
                    result=client.generate(model,prompt)
                except Exception as e:
                    # Abort, don't misrepresent networking/auth/model error as an incorrect answer.
                    raise RuntimeError(f'Generation failed for {meta["dataset"]} {label} {q["id"]}: {e}') from e
                llm_seconds=time.perf_counter()-start
                total_seconds=llm_seconds+retrieval_ms/1000
                answer=result.get('response','').strip()
                expect=q['answer']
                if norm(expect)=='нет данных' and not re.search('[А-Яа-яЁё]',q['question']):expect='NO DATA'
                in_tok=result.get('prompt_eval_count');out_tok=result.get('eval_count')
                if billing=='hourly':cost=total_seconds/3600*hourly_usd
                elif billing=='token':cost=(in_tok or 0)*input_per_million/1e6+(out_tok or 0)*output_per_million/1e6
                else:cost=None
                row={'dataset':meta['dataset'],'synthetic':int(meta['synthetic']),'qid':q['id'],'kind':q['kind'],
                  'repeat':repeat,'configuration':f'{label}_{method or "closed"}', 'model':model,'retrieval_method':method or 'closed',
                  'k':k,'gold':expect,'answer':answer,'exact':int(norm(answer)==norm(expect)),'token_f1':round(safe_f1(answer,expect),4),
                  'abstention':int(norm(answer) in ['нет данных','no data']),
                  'evidence_all_found':int(set(q['gold']).issubset(found)) if q['gold'] else None,
                  'retrieved_ids':';'.join(found),'retrieval_ms':round(retrieval_ms,4),'inference_s':round(llm_seconds,5),
                  'total_seconds':round(total_seconds,5),'input_tokens':in_tok,'output_tokens':out_tok,
                  'load_seconds':round(result.get('load_duration',0)/1e9,5),
                  'eval_seconds':round(result.get('eval_duration',0)/1e9,5),
                  'bill_mode':billing,'estimated_cost_usd':round(cost,8) if cost is not None else '',
                  'error':''}
                rows.append(row)
                print('MODEL',q['id'],row['configuration'],'EXACT',row['exact'],flush=True)
    write_csv(out/'llm_raw.csv',rows)
    summaries=[]
    for name in sorted(set(r['configuration'] for r in rows)):
        rr=[r for r in rows if r['configuration']==name]
        costs=[r['estimated_cost_usd'] for r in rr if r['estimated_cost_usd']!='']
        correct=sum(r['exact'] for r in rr)
        summaries.append({'configuration':name,'n':len(rr),'n_correct':correct,'accuracy':round(correct/len(rr),4),
          'f1':round(statistics.mean(r['token_f1'] for r in rr),4),
          'p50_seconds':round(float(np.median([r['total_seconds'] for r in rr])),4),
          'p95_seconds':round(float(np.percentile([r['total_seconds'] for r in rr],95)),4),
          'mean_input_tokens':round(statistics.mean(r['input_tokens'] or 0 for r in rr),2),
          'mean_output_tokens':round(statistics.mean(r['output_tokens'] or 0 for r in rr),2),
          'observed_model_cost_usd_total':round(sum(costs),7) if costs else 'NOT_MEASURED',
          'estimated_usd_per_correct':round(sum(costs)/correct,7) if costs and correct else 'NOT_MEASURED',
          'cost_exclusions':'indexing, electricity, storage, network, amortization' if costs else 'no billing rate supplied'})
    write_csv(out/'llm_summary.csv',summaries)
    ci=paired_ci(rows,a='small_bm25',b='large_bm25')
    (out/'llm_paired_ci.json').write_text(json.dumps(ci,ensure_ascii=False,indent=2),encoding='utf-8')
    return rows,summaries

def system_manifest(out,args):
    os.makedirs(out,exist_ok=True)
    payload={'run_time_utc':datetime.now(timezone.utc).isoformat(),'argv':sys.argv,
      'python':sys.version,'platform':platform.platform(),'processor':platform.processor(),
      'logical_cpus':os.cpu_count(),'memory_total_bytes':psutil.virtual_memory().total,
      'rss_start_bytes':psutil.Process(os.getpid()).memory_info().rss,
      'numpy':np.__version__,'sklearn':__import__('sklearn').__version__,
      'git_commit':os.environ.get('GIT_COMMIT','unknown'),'run_mode':args.mode,
      'llm_inference_requested':args.mode=='llm','llm_inference_completed':False,
      'data_origin':'synthetic demo unless explicitly external corpus supplied'}
    (out/'execution_manifest.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['retrieval','llm'],default='retrieval')
    p.add_argument('--datasets',default='projects,incidents,revisions',help='comma separated; external requires --external-file')
    p.add_argument('--external-file')
    p.add_argument('--out',default=str(HERE/'results'/'run_latest'))
    p.add_argument('--k',type=int,default=3)
    p.add_argument('--ollama-url',default='http://127.0.0.1:11434')
    p.add_argument('--small-model',default='qwen3:0.6b')
    p.add_argument('--large-model',default='qwen3:4b')
    p.add_argument('--embed-model',default=None,help='Must be pretrained Ollama model. Example nomic-embed-text-v2-moe')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--billing-mode',choices=['none','hourly','token'],default='none')
    p.add_argument('--hourly-usd',type=float)
    p.add_argument('--input-per-million',type=float)
    p.add_argument('--output-per-million',type=float)
    a=p.parse_args(argv)
    if a.k<1 or a.repeats<1: p.error('k and repeats must be positive')
    if any(x is not None and x<0 for x in [a.hourly_usd,a.input_per_million,a.output_per_million]):p.error('negative costs')
    datasets=a.datasets.split(',')
    if any(x not in REGISTRY and x!='external' for x in datasets):p.error('bad datasets')
    if 'external' in datasets and not a.external_file:p.error('external requires --external-file')
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    system_manifest(out,a)
    if a.mode=='retrieval':
        client=Ollama(a.ollama_url) if a.embed_model else None
        if client:client.check([a.embed_model])
        _,s=retrieval_suite(datasets,out,bool(a.embed_model),client,a.embed_model,k=a.k,external=a.external_file)
        print('Saved',len(s),'summary rows:',out)
    else:
        client=Ollama(a.ollama_url)
        for dataset in datasets:
            dest=out/dataset;dest.mkdir(exist_ok=True)
            data=a.external_file if dataset=='external' else None
            _,summary=model_suite(dataset,dest,client,a.small_model,a.large_model,embed_model=a.embed_model,data=data,
                    k=a.k,limit=a.limit,repeats=a.repeats,hourly_usd=a.hourly_usd,
                    input_per_million=a.input_per_million,output_per_million=a.output_per_million,billing=a.billing_mode)
            print('Saved',len(summary),'LLM configs:',dest)
        path=out/'execution_manifest.json'
        actual=json.loads(path.read_text(encoding='utf-8'))
        actual['llm_inference_completed']=True
        path.write_text(json.dumps(actual,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__': main()
