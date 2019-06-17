import numpy
import random
seed = 103
random.seed(seed)
numpy.random.seed(seed)

import vg.simple_data as sd
import vg.flickr8k_provider as dp_f

import vg.defn.three_way2 as D
import vg.scorer

batch_size = 16
epochs=25

prov_flickr = dp_f.getDataProvider('flickr8k', root='../..', audio_kind='mfcc')

data_flickr = sd.SimpleData(prov_flickr, tokenize=sd.characters, min_df=1, scale=False,
                            batch_size=batch_size, shuffle=True)

model_config = dict(TextImage=dict(ImageEncoder=dict(size=1024, size_target=4096),
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
                    TextEncoderBottom=dict(size_feature=data_flickr.mapper.size(),
                                           size_embed=128,
                                           size=1024,
                                           depth=1)
                   )






def audio(sent):
    return sent['audio']

net = D.Net(model_config)
net.batcher = None
net.mapper = None

scorer = vg.scorer.Scorer(prov_flickr, 
                    dict(split='val', 
                         tokenize=audio, 
                         batch_size=batch_size
                         ))
                  

run_config = dict(epochs=epochs,
                  validate_period=400,
                  tasks=[ ('SpeechText', net.SpeechText),
                          ('SpeechImage', net.SpeechImage),
                          ('TextImage', net.TextImage)],
                  Scorer=scorer)
D.experiment(net=net, data=dict(SpeechText=data_flickr, SpeechImage=data_flickr, TextImage=data_flickr), run_config=run_config)

