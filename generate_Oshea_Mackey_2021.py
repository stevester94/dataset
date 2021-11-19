#!/usr/bin/env python
import transmitters_2021 as transmitters
from source_alphabet import source_alphabet
import analyze_stats
from gnuradio import channels, gr, blocks
import numpy as np
import numpy.fft, cPickle, gzip
import random

# dynamic_channel_model is not deterministic even with seed :(
SEED = 1337
np.random.seed(SEED)
random.seed(SEED)


'''
Generate dataset with dynamic channel model across range of SNRs
'''

apply_channel = True

dataset = {}

# The output format looks like this
# {('mod type', SNR): np.array(nvecs_per_key, 2, vec_length), etc}

# CIFAR-10 has 6000 samples/class. CIFAR-100 has 600. Somewhere in there seems like right order of magnitude
nvecs_per_key = 1000
vec_length = 128
snr_vals = range(-20,20,2)
samps_per_symbol_options = [2,4,6,8,10,12,14,16,18,20]

for samps_per_symbol in samps_per_symbol_options:
    transmitters.sps = samps_per_symbol
    for snr in snr_vals:
        print "snr is ", snr
        for alphabet_type in transmitters.transmitters.keys():
            for i,mod_type in enumerate(transmitters.transmitters[alphabet_type]):
                dataset[(mod_type.modname, snr, samps_per_symbol)] = np.zeros([nvecs_per_key, 2, vec_length], dtype=np.float32)
                # moar vectors!
                insufficient_modsnr_vectors = True
                modvec_indx = 0
                while insufficient_modsnr_vectors:
                    tx_len = int(10e3)
                    if mod_type.modname == "QAM16":
                        tx_len = int(20e3)
                    if mod_type.modname == "QAM64":
                        tx_len = int(30e3)
                    src = source_alphabet(alphabet_type, tx_len, True)
                    mod = mod_type()
                    fD = 1
                    delays = [0.0, 0.9, 1.7]
                    mags = [1, 0.8, 0.3]
                    ntaps = 8
                    noise_amp = 10**(-snr/10.0)
                    chan = channels.dynamic_channel_model( 200e3, 0.01, 50, .01, 0.5e3, 8, fD, True, 4, delays, mags, ntaps, noise_amp, SEED )  
                    snk = blocks.vector_sink_c()    
                    tb = gr.top_block() 
                    # connect blocks
                    if apply_channel:
                        tb.connect(src, mod, chan, snk)
                    else:
                        tb.connect(src, mod, snk)
                    tb.run()    
                    raw_output_vector = np.array(snk.data(), dtype=np.complex64)
                    # start the sampler some random time after channel model transients (arbitrary values here)
                    sampler_indx = random.randint(50, 500)
                    while sampler_indx + vec_length < len(raw_output_vector) and modvec_indx < nvecs_per_key:
                        sampled_vector = raw_output_vector[sampler_indx:sampler_indx+vec_length]
                        # Normalize the energy in this vector to be 1
                        energy = np.sum((np.abs(sampled_vector)))
                        sampled_vector = sampled_vector / energy
                        dataset[(mod_type.modname, snr, samps_per_symbol)][modvec_indx,0,:] = np.real(sampled_vector)
                        dataset[(mod_type.modname, snr, samps_per_symbol)][modvec_indx,1,:] = np.imag(sampled_vector)
                        # bound the upper end very high so it's likely we get multiple passes through
                        # independent channels
                        sampler_indx += random.randint(vec_length, round(len(raw_output_vector)*.05))
                        modvec_indx += 1    
                    if modvec_indx == nvecs_per_key:
                        # we're all done
                        insufficient_modsnr_vectors = False

print "all done. writing to disk"
cPickle.dump( dataset, file("RML2016.10a_dict.dat", "wb" ) )
