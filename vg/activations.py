import numpy as np
import torch
import vg.mfcc as C

def from_audio(model_path, paths, device='cpu'):
    """Return per-layer activation states for audio files stored in paths, from model stored in model_path. 
    The output is a list of arrays of shape (time, layer, feature).
    """
    net = torch.load(model_path, device)
    audio = [ C.extract_mfcc(path) for path in paths ]
    return get_state_stack(net, audio)


def get_state_stack(net, audios, batch_size=128):
    import onion.util as util
    from vg.simple_data import vector_padder
    """Pass audios through the model and for each audio return the state of each timestep and each layer."""
    device = next(net.parameters()).device
    result = []
    lens = inout(np.array(list(map(len, audios))))
    rs = (r for batch in util.grouper(audios, batch_size) 
                for r in state_stack(net, torch.from_numpy(vector_padder(batch)).to(device)).cpu().numpy()
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


def state_stack(net, audio):
    from vg.scorer import testing
    with testing(net):
        states_bot = net.SpeechImage.SpeechEncoderBottom.states(audio)
        states_top = net.SpeechImage.SpeechEncoderTop.states(states_bot[-1])
    states = torch.cat([states_bot, states_top], dim=0).permute(1, 2, 0, 3) #batch x length x layer x feature
    return states
