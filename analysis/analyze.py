import numpy as np
import json
import torch
import vg.scorer as S
import logging


PREFIX= '../experiments'

def valid_results():
    """Table 1 (tab:core-results)"""
    out = []
    for run in valid_runs():
        result = run.copy()
        result['recall@10'] = np.mean(run['recall@10'])
        result['speaker_id'] = np.mean(run['speaker_id'])
        result['medr'] = np.mean(run['medr'])
        out.append(result)
    return out



def test_results():
    """Table 2 (tab:core-results-test)"""
    runs = [ dict(cond="",         tasks=1, s=2, t='.', s2i=2, s2t='.', t2s='.', t2i='.'),
             dict(cond="joint",    tasks=3, s=2, t=1,   s2i=2, s2t=0,   t2s=0,   t2i=1) ]
    out = []
    for run in runs:
        spec = "{}/s{s}-t{t}-s2i{s2i}-s2t{s2t}-t2s{t2s}-t2i{t2i}".format(PREFIX, **run)
        suffix = bestrun(spec, suffixes='efg', cond=run['cond'])[1]
        scores = testscores("{}/s{s}-t{t}-s2i{s2i}-s2t{s2t}-t2s{t2s}-t2i{t2i}".format(PREFIX, **run), 
                        suffix=suffix, cond=run['cond'])
        out.append({**run, **scores})
    return out
        

def inv_results():
    """Figure 2 (fig:speaker-inv)"""
    from subprocess import call
    data = list(melt(valid_runs()))
    json.dumps("valid_runs.json")
    call(["Rscript", "inv_results.R"])

def rsa_results():
    """Table 3 (tab:rsa)"""
    from sklearn.metrics.pairwise import cosine_similarity

    import vg.flickr8k_provider as dp_f
    logging.getLogger().setLevel('INFO')
    logging.info("Loading data")
    prov_flickr = dp_f.getDataProvider('flickr8k', root='..', audio_kind='mfcc')
    logging.info("Setting up scorer")
    scorer = S.Scorer(prov_flickr, 
                    dict(split='val', 
                         tokenize=lambda sent: sent['audio'], 
                         batch_size=16
                         ))
    # SIMS
    logging.info("Computing MFCC similarity matrix")
    mfcc     = np.array([ audio.mean(axis=0) for audio in scorer.sentence_data])    
    sim = {}
    sim['mfcc']  = cosine_similarity(mfcc)
    sim['text']  = scorer.string_sim
    sim['image'] = scorer.sim_images
    # PRED 1 s2i
    logging.info("Computing M1,s2i similarity matrix")
    net = load_best_run('{}/s2-t.-s2i2-s2t.-t2s.-t2i.'.format(PREFIX), cond='')
    pred = S.encode_sentences(net, scorer.sentence_data, batch_size=scorer.config['batch_size'])
    sim['m1,s2i'] = cosine_similarity(pred)
    # PRED 6 s2i  
    logging.info("Computing M6,s2i similarity matrix")
    net = load_best_run('{}/s2-t1-s2i2-s2t0-t2s0-t2i1'.format(PREFIX), cond='joint')
    pred = S.encode_sentences(net, scorer.sentence_data, batch_size=scorer.config['batch_size'])
    sim['m6,s2i']  = cosine_similarity(pred)
    # PRED 6 s2t
    logging.info("Computing M6,s2t similarity matrix")
    pred = S.encode_sentences_SpeechText(net, scorer.sentence_data, batch_size=scorer.config['batch_size'])
    sim['m6,s2t'] =  cosine_similarity(pred)
    out = []
    logging.info("Computing RSA scores")
    for row in ['m1,s2i', 'm6,s2i', 'm6,s2t', 'image']:
        for col in ['mfcc', 'text', 'image']:
            out.append((row, col, S.RSA(sim[row], sim[col])))
    return out




         

def load_best_run(spec, cond='joint'):
    _, suffix, epoch = bestrun(spec, suffixes='efg', cond=cond)
    net = torch.load("{}-{}-{}/model.{}.pkl".format(spec, cond, suffix, epoch))
    return net


def valid_runs():   
    base = [ dict(cond="",    tasks=1, s=2, t='.', s2i=2, s2t='.', t2s='.', t2i='.'),
             dict(cond="joint",    tasks=2, s=2, t=1,   s2i=2, s2t=0,   t2s=0,   t2i='.'),
             dict(cond="disjoint", tasks=2, s=2, t=1,   s2i=2, s2t=0,   t2s=0,   t2i='.'),
           ]

    S2I = [1, 2]
    ST  = [0, 1]
    rest = []
    metrics = ('recall@10', 'medr', 'speaker_id')
    for cond in ["joint", "disjoint"]:
        for s2i in S2I:
            for st in ST:
                 rest.append(dict(cond=cond, tasks=3, s=2, t=1, s2i=s2i, s2t=st, t2s=st, t2i=1))
        
        rest.append(         dict(cond=cond, tasks=3, s=4, t=1, s2i=0,   s2t=0,  t2s=0,  t2i=0)) 

    for run in base + rest:
        scores = validscores("{}/s{s}-t{t}-s2i{s2i}-s2t{s2t}-t2s{t2s}-t2i{t2i}".format(PREFIX, **run), 
                           suffixes="efg", 
                           cond=run['cond'])

        for metric in metrics:
           run[metric] = scores[metric]
        yield run

def melt(runs):
    for run in runs:
        assert len(run['recall@10']) == len(run['medr'])
        for i in range(len(run['recall@10'])):
            yield {**run, 'recall@10': run['recall@10'][i], 'medr': run['medr'][i], 'speaker_id': run['speaker_id'][i]}

    

KEYS = ("cond", "tasks", "s", "t", "s2i", "s2t", "t2s", "t2i", "recall@10", "medr")
def json2latex(data, keys=KEYS):
    if keys is None:
        keys = data.keys()
    def format(x):
        if type(x) in [float, np.float, np.float64]:
            return "{:5.3f}".format(x)
        else:
            return "{}".format(x)
    return "\\\\\n".join(" & ".join(format(datum[key]) for key in keys) for datum in data)

def bestrec(path, criterion='recall@10'):   
         #print(path)
         R = [ json.loads(line) for line in open(path) ]        
         besti = np.argmax([Ri['retrieval'][criterion] for Ri in R])            
         return R[besti]

def bestrun(spec, suffixes, cond='joint'):
    rec = []
    for suffix in suffixes:
        for epoch, line in zip(range(1, 26), open("{}-{}-{}/result.json".format(spec, cond, suffix))):
            data = json.loads(line)
            rec.append((data['retrieval']['recall@10'], suffix, epoch))
    return sorted(rec)[-1]

def testscores(spec, suffix, cond):
    R = json.load(open("{}-{}-{}/test/result.json".format(spec, cond, suffix)))
    return {'recall@10': R['retrieval']['recall@10'], 'medr': R['retrieval']['medr']}
    

def validscores(spec, suffixes, cond='joint'):
    result = {'recall@10': [], 'medr': [], 'speaker_id': []}    
    for suffix in suffixes:
        data = bestrec("{}-{}-{}/result.json".format(spec, cond, suffix))    
        # display key results averaged over runs, with min and max
        # recall@10
        # medr
        # speaker accuracy
        result['recall@10'].append(data['retrieval']['recall@10'])
        result['medr'].append(data['retrieval']['medr'])
        result['speaker_id'].append(data['speaker_id']['rep'])
    return result
