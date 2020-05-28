import brian2.numpy_ as np
import brian2 as bb
from brian2 import ms, second, Hz, mV, pA, nS, pF
from collections import defaultdict

from matplotlib import pyplot as plt
from tqdm import tqdm
import assemblyseq.assemblyseq as asq
import importlib

def network_sim(config):
    # import brian2.numpy_ as np
    # import brian2.only as bb
    # from brian2 import ms, second, Hz, mV, pA, nS, pF
    # bb.core.tracking.InstanceFollower.instance_sets = defaultdict(bb.core.tracking.InstanceTrackerSet)
    bb.devices.reinit_devices()
    i = config['n_sim']
    faster_run = True
    if faster_run:
        bb.set_device('cpp_standalone', directory='PETH_standalone_'+str(i), build_on_run=False)

    bb.start_scope()
    if faster_run:
        bb.get_device().reinit()
        bb.get_device().activate(build_on_run=False, directory='PETH_standalone_'+str(i))

    import assemblyseq.assemblyseq as asq


    nn = asq.Nets(config)

    nn.generate_ps_assemblies('gen_no_overlap')
    nn.set_net_connectivity()

    nn.set_spike_monitor()
    nn.set_rate_monitor()

    for gr_num in range(nn.n_ass):
        gr = nn.p_ass_index[0][gr_num]
        t_inp = (20.55 + gr_num * .1) * second
        nn.set_noisy_input(gr, t_inp, sigma=0 * ms)

    nn.set_syn_input(nn.p_ass_index[0][0], np.arange(26, 31, 1))
#     nn.balance(10 * second, 5.)
#     nn.balance(10 * second, .1)
#     nn.balance(5 * second, .01)
#     nn.run_sim(12*second)
    nn.run_sim(1*second)
    # nn.Pe.I -= .0 * pA

    if faster_run:
        bb.get_device().build(directory='PETH_standalone_'+str(i), compile=True, run=True, debug=False)

    return nn

config = {'Ne': 20000, 'Ni': 5000, 'cp_ee': .01, 'cp_ie': .01, 'cp_ei': 0.01, 'cp_ii': .01,
              'n_ass': 10, 's_ass': 500, 'pr': .15, 'pf': .03, 'symmetric_sequence': True, 'p_rev': .03,
              'g_ee': 0.1 * nS, 'g_ie': 0.1 * nS, 'g_ei': 0.4 * nS, 'g_ii': 0.4 * nS, 'n_sim': 0}


configs = []
for i, pf in enumerate([0.01, 0.03, 0.1]):
    cf = config.copy()
    cf['pf'] = pf
    cf['n_sim'] = i
    configs.append(cf)

nn = network_sim(configs[0])

del nn
nn1 = network_sim(configs[0])