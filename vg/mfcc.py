import soundfile as sf
import python_speech_features as psf
import numpy as np

def extract_mfcc(f, truncate=20, nfft=512):
    (sig, rate) = sf.read(f)
    max_len = truncate*rate
    mfcc_feat = psf.mfcc(sig[:max_len], samplerate=rate, nfft=nfft)    
    return np.asarray(mfcc_feat, dtype='float32')
