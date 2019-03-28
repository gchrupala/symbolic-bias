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
    json.dump(make_json_happy(out), open("rsa.json", "w"), indent=2)


def phoneme_decoding_results():
    """Table 5 (tab:phoneme-decoding)"""
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    try:
        data = np.load("phoneme_decoding_data.npy").item()
    except FileNotFoundError:
        nets = get_nets()
        data = phoneme_decoding_data(nets)
        np.save("phoneme_decoding_data.npy", data)
    result = {}
    for rep in data.keys():
        scaler = StandardScaler()
        X_train, X_test, y_train, y_test = train_test_split(data[rep]['features'], data[rep]['labels'], test_size=1/2, random_state=123)       
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
        #m = GridSearchCV(LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=200), {'C': [ 10**n for n in range(-1, 1) ]}, cv=3, n_jobs=-1)
        logging.info("Fitting Logistic Regression for {}".format(rep))
        m = LogisticRegression(solver="lbfgs", multi_class='auto', max_iter=300, random_state=123, C=1.0)
        m.fit(X_train, y_train)
        result[rep] = float(m.score(X_test, y_test)) #dict(score=m.score(X_test, y_test), cv_results=m.cv_results_)
        print(result[rep])
    json.dump(make_json_happy(result), open("phoneme-decoding.json", "w"))

def get_nets():
    import vg.defn.three_way2 as D
    net_base = load_best_run('{}/s2-t.-s2i2-s2t.-t2s.-t2i.'.format(PREFIX), cond='')
    net_mt = load_best_run('{}/s2-t1-s2i2-s2t0-t2s0-t2i1'.format(PREFIX), cond='joint')
    config = dict(TextImage=dict(ImageEncoder=dict(size=1024, size_target=4096),
                                   lr=0.0002,
                                   margin_size=0.2,
                                   max_norm=2.0, 
                                   TextEncoderTop=dict(size=1024, size_feature=1024, depth=1, size_attn=128)),
                    SpeechImage=dict(ImageEncoder=dict(size=1024, size_target=4096),
                                     lr=0.0002,
                                     margin_size=0.2,
                                     max_norm=2.0, 
                                     SpeechEncoderTop=dict(size=1024, size_input=1024, depth=2, size_attn=128)),
                    SpeechText=dict(TextEncoderTop=dict(size_feature=1024,
                                                        size=1024,
                                                        depth=0,
                                                        size_attn=128),
                                    SpeechEncoderTop=dict(size=1024,
                                                          size_input=1024,
                                                          depth=0,
                                                          size_attn=128), 
                                    lr=0.0002,
                                    margin_size=0.2,
                                    max_norm=2.0),
                    
                    SpeechEncoderBottom=dict(size=1024, depth=2, size_vocab=13, filter_length=6, filter_size=64, stride=2),
                    TextEncoderBottom=dict(size_feature=net_mt.TextEncoderBottom.size_feature,
                                           size_embed=128,
                                           size=1024,
                                           depth=1)
                   )
    net_base = load_best_run('{}/s2-t.-s2i2-s2t.-t2s.-t2i.'.format(PREFIX), cond='')
    net_mt = load_best_run('{}/s2-t1-s2i2-s2t0-t2s0-t2i1'.format(PREFIX), cond='joint')
    net_init = D.Net(config).cuda()
    return [('m6_init', net_init), ('m1', net_base), ('m6', net_mt) ]

def phoneme_decoding_data(nets, alignment_path="../data/flickr8k/dataset.val.fa.json",
                          dataset_path="../data/flickr8k/dataset.json",
                          max_size=5000,
                          directory="."):
    """Generate data for training a phoneme decoding model."""
    import vg.flickr8k_provider as dp

    logging.getLogger().setLevel('INFO')
    logging.info("Loading alignments")
    data = {}
    for line in open(alignment_path):
        item = json.loads(line)
        data[item['sentid']] = item
    logging.info("Loading audio features")
    prov = dp.getDataProvider('flickr8k', root='..', audio_kind='mfcc')
    val = list(prov.iterSentences(split='val'))
    data_filter = [ (data[sent['sentid']], sent) for sent in val
                        if np.all([word.get('start', False) for word in data[sent['sentid']]['words']]) ]
    data_filter = data_filter[:max_size]
    data_state =  [phoneme for (utt, sent) in data_filter for phoneme in slices(utt, sent['audio']) ]
    result = {}
    logging.info("Extracting MFCC examples")
    result['mfcc'] = fa_data(data_state)
    for name, net in nets:
        result[name] = {}
        L =  1
        S =  net.SpeechEncoderBottom.stride
        logging.info("Extracting recurrent layer states")
        audio = [ sent['audio'] for utt,sent in data_filter ]
        states = get_layer_states(net, audio, batch_size=32)
        layer = 0
        def aggregate(x):
                return x[:,layer,:].mean(axis=0)
        data_state =  [phoneme for i in range(len(data_filter))
                         for phoneme in slices(data_filter[i][0], states[i], index=lambda x: index(x, stride=S), aggregate=aggregate) ]
        result[name] = fa_data(data_state)
    return result

def get_layer_states(net, audios, batch_size=128):
    import onion.util as util
    from vg.simple_data import vector_padder
    """Pass audios through the model and for each audio return the state of each timestep and each layer."""
    result = []
    lens = inout(np.array(list(map(len, audios))))
    rs = (r for batch in util.grouper(audios, batch_size) 
                for r in layer_states(net, torch.from_numpy(vector_padder(batch)).cuda()).cpu().numpy()
         )
    for (r,l) in zip(rs, lens):
        result.append(np.expand_dims(r[-l:,:], axis=1))
    return result

def get_state_stack(net, audios, batch_size=128):
    import onion.util as util
    from vg.simple_data import vector_padder
    """Pass audios through the model and for each audio return the state of each timestep and each layer."""
    result = []
    lens = inout(np.array(list(map(len, audios))))
    rs = (r for batch in util.grouper(audios, batch_size) 
                for r in state_stack(net, torch.from_numpy(vector_padder(batch)).cuda()).cpu().numpy()
         )
    for (r,l) in zip(rs, lens):
        result.append(r[-l:,:])
    return result


def inout(L, pad=6, ksize=6, stride=2): # Default Flickr8k model parameters 
    return np.floor( (L+2*pad-1*(ksize-1)-1)/stride + 1).astype(int)

def index(t, stride=2, size=6):
    """Return index into the recurrent state of speech model given timestep
    `t`.
    See: https://pytorch.org/docs/stable/nn.html#torch.nn.Conv1d
    """
    return inout(t//10, pad=size, ksize=size, stride=stride) # sampling audio every 10ms


def layer_states(net, audio):
    from vg.scorer import testing
    with testing(net):
        states = net.SpeechImage.SpeechEncoderBottom(audio)
    return states

def state_stack(net, audio):
    from vg.scorer import testing
    with testing(net):
        states_bot = net.SpeechImage.SpeechEncoderBottom.states(audio)
        states_top = net.SpeechImage.SpeechEncoderTop.states(states_bot[-1])
    states = torch.cat([states_bot, states_top], dim=0).permute(1, 2, 0, 3) #batch x length x layer x feature
    return states
        


def fa_data(data_state):
    y, X = zip(*data_state)
    X = np.vstack(X)
    y = np.array(y)
    # Get rid of NaNs
    ix = np.isnan(X.sum(axis=1))
    X = X[~ix]
    y = y[~ix]
    return dict(features=X, labels=y)

def slices(utt, rep, index=lambda ms: ms//10, aggregate=lambda x: x.mean(axis=0)):
    """Return sequence of slices associated with phoneme labels, given an
       alignment object `utt`, a representation array `rep`, and
       indexing function `index`, and an aggregating function\
       `aggregate`.

    """
    for phoneme in phones(utt):
        phone, start, end = phoneme
        assert index(start)<index(end)+1, "Something funny: {} {} {} {}".format(start, end, index(start), index(end))
        yield (phone, aggregate(rep[index(start):index(end)+1]))

def phones(utt):
    """Return sequence of phoneme labels associated with start and end
     time corresponding to the alignment JSON object `utt`.
    
    """
    for word in utt['words']:
        pos = word['start']
        for phone in word['phones']:
            start = pos
            end = pos + phone['duration']
            pos = end
            label = phone['phone'].split('_')[0]
            if label != 'oov':
                yield (label, int(start*1000), int(end*1000))

         

def load_best_run(spec, cond='joint'):
    _, suffix, epoch = bestrun(spec, suffixes='efg', cond=cond)
    path = "{}-{}-{}/model.{}.pkl".format(spec, cond, suffix, epoch)
    logging.info("Laoding model from {}".format(path))
    net = torch.load(path)
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


def make_json_happy(x):
    if isinstance(x, np.float32) or isinstance(x, np.float64):
        return float(x)
    elif isinstance(x, dict):
        return {key: make_json_happy(value) for key, value in x.items() }
    elif isinstance(x, list):
        return [ make_json_happy(value) for value in x ]
    elif isinstance(x, tuple):
        return tuple(make_json_happy(value) for value in x)
    else:
        return x

