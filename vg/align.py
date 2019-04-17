import json
import multiprocessing
nthreads = multiprocessing.cpu_count()
import logging
import gentle
resources = gentle.Resources()
import vg.activations
import numpy as np

def phoneme_activations(activations, alignment):
    """Return array of phoneme labels and array of corresponding mean-pooled activation states."""
    labels, states = zip(*list(slices(alignment, activations, index=vg.activations.index)))
    return np.array(labels), np.stack(states)
    
def on_progress(p):
    for k,v in p.items():
        logging.debug("%s: %s" % (k, v))


def align(audiopath, transcript):
    logging.info("converting audio to 8K sampled wav")
    with gentle.resampled(audiopath) as wavfile:
        logging.info("starting alignment")
        aligner = gentle.ForcedAligner(resources, transcript, nthreads=nthreads, disfluency=False, 
                                   conservative=False)
        return json.loads(aligner.transcribe(wavfile, progress_cb=on_progress, logging=logging).to_json())


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
