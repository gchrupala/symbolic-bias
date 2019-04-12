import json
import multiprocessing
nthreads = multiprocessing.cpu_count()
import logging
import gentle
resources = gentle.Resources()


def on_progress(p):
    for k,v in p.items():
        logging.debug("%s: %s" % (k, v))


def align(audiopath, transcript):
    logging.info("converting audio to 8K sampled wav")
    with gentle.resampled(audiopath) as wavfile:
        logging.info("starting alignment")
        aligner = gentle.ForcedAligner(resources, transcript, nthreads=nthreads, disfluency=False, 
                                   conservative=False)
        return aligner.transcribe(wavfile, progress_cb=on_progress, logging=logging)

