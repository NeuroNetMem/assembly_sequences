import brian2.numpy_ as np
import brian2 as bb
from brian2 import ms, second, Hz, mV, pA, nS, pF
from collections import defaultdict

eqs_exc = '''dv/dt = (g_l*(v_r-v)+Ie+Ii+I)/(C_m) : volt
            dge/dt = -ge/(tau_exc) : siemens
            dgi/dt = -gi/(tau_inh) : siemens
            Ie = ge*(v_e-v) : amp
            Ii = gi*(v_i-v) : amp
            I : amp '''
eqs_inh = '''dv/dt = (g_l*(v_r-v)+Ie+Ii+I)/(C_m) : volt
            dge/dt = -ge/(tau_exc) : siemens
            dgi/dt = -gi/(tau_inh) : siemens
            Ie = ge*(v_e-v) : amp
            Ii = gi*(v_i-v) : amp
            I : amp '''

params ={'Ne': 10000, 'Ni': 2500, 'cp_ee': .02, 'cp_ie': .02, 'cp_ei': .02,
                              'cp_ii': .02, 'pr': .05, 'pf': .05, 'g_ee': 0.19 * nS, 'g_ie': 0.2 * nS, 'g_ei': 1.0 * nS,
                              'g_ii': 1.0 * nS, 'n_ass': 10, 's_ass': 500, 'n_chains': 0, 'cf_ffn': 1., 'cf_rec': 1.,
                              'g_l': 10. * nS, 'C_m': 200 * pF, 'v_r': -60. * mV, 'v_e': 0. * mV, 'v_i': -80. * mV,
                              'tau_m_exc': 20. * ms, 'tau_m_inh': 20. * ms, 'tau_inh': 10 * ms,
                              'tau_fast_inh': 10 * ms, 'tau_exc': 5. * ms, 'tau_stdp': 20. * ms,
                              'alpha': .2, 'g_min': 0 * nS, 'g_max': 50 * nS, 'eta_p': 0.001}

faster_run = True
if faster_run:
    bb.set_device('cpp_standalone', directory='PETH_standalone', build_on_run=False)

def network_sim():
    # this line resets Brian's object tracking system, without it, the simulation will crash when run a second time
    # bb.core.tracking.InstanceFollower.instance_sets = defaultdict(bb.core.tracking.InstanceTrackerSet)
    bb.start_scope()
    if faster_run:
        bb.get_device().reinit()
        bb.get_device().activate(build_on_run=False, directory='PETH_standalone')

    network = bb.Network()
    Pe = bb.NeuronGroup(20000, eqs_exc, threshold='v > -50 * mV',
                        reset='v = -60 * mV', refractory=2. * ms, method='euler',
                        namespace=params)
    # Pi = bb.NeuronGroup(5000, eqs_inh, threshold='v > -50 * mV',
    #                          reset='v = -60 * mV', refractory=2. * ms, method='euler',
    #                          namespace=params)
    C_ee = bb.Synapses(Pe, Pe, model='w:siemens', on_pre='ge+=w')
    #C_ie = bb.Synapses(Pe, Pi, model='w:siemens', on_pre='ge+=w')

    i = np.random.randint(20000)
    j = np.random.randint(20000)
    C_ee.connect(i=i, j=j)
    C_ee.w = params['g_ee']
    network.add(Pe)
    network.add(C_ee)
    network.run(1 * second)


    if faster_run:
        bb.get_device().build(directory='PETH_standalone', compile=True, run=True, debug=False)

    return network


print("run 1")
nn = network_sim()
del nn
# the second run is where the bug triggers
print("run 2")
nn1 = network_sim()
