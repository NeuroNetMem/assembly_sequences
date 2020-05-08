import numpy as np
import brian2 as bb
from brian2 import ms, second, Hz, mV, pA, nS, pF
from time import time, asctime

# some custom modules
from assemblyseq import plotter, calc_spikes

g_l = 10. * nS
C_m = 200 * pF
v_r = -60. * mV
v_e = 0. * mV
v_i = -80. * mV
tau_m_exc = 20. * ms
tau_m_inh = 20. * ms
tau_inh = 10 * ms
tau_fast_inh = 10 * ms
tau_exc = 5. * ms
tau_stdp = 20. * ms
alpha = .2
g_min = 0 * nS
g_max = 50 * nS

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
eq_stdp = '''dapost/dt = -apost/tau_stdp : 1 (event-driven)
            dapre/dt = -apre/tau_stdp : 1 (event-driven)
            w: siemens '''

eq_pre = '''gi_post+=w
            w=clip(w+eta_p*(apost-alpha)*g_ei,g_min,g_max)
            apre+=1'''
eq_post = '''w=clip(w+eta_p*apre*g_ei,g_min,g_max)
            apost+=1'''


def if_else(condition, a, b):
    if condition:
        return a
    else:
        return b


class Pointless(object):
    """a hackaround changing learning rate eta"""
    pass


eta = Pointless()
eta.v = .001
eta.eta = 1. * eta.v

# defines an extra clock according to which some extra input currents 
# can be injected; 
# one can play with changing conductances etc...

syn_input_freq = 1. * Hz  # frequency of current input oscillation
myclock = bb.Clock(dt=10 * ms)  # create an extra clock


@bb.network_operation(myclock)
def inject():
    """
        Injects currents into neuronal populations...off by default
    """
    if myclock.t > 25000 * ms:
        nn.Pe.I = nn.ext_input + \
            nn.Isine * (1. + 0 * np.sin(2 * np.pi * myclock.t * syn_input_freq))
        nn.Pi.I = nn.ext_input + \
            nn.Isini * (1. + 0 * np.sin(2 * np.pi * myclock.t * syn_input_freq))


# noinspection PyUnusedLocal,PyPep8Naming
class Nets:
    # FIX
    def __init__(self, Ne=10000, Ni=2500, cp_ee=.02, cp_ie=.02, cp_ei=.02,
                 cp_ii=.02, pr=.05, pf=.05, g_ee=0.19 * nS, g_ie=0.2 * nS, g_ei=1.0 * nS,
                 g_ii=1.0 * nS, n_ass=10, s_ass=500, n_chains=0, cf_ffn=1., cf_rec=1.,
                 type_ext_input='curr', ext_input=200 * pA, synapses_per_nrn=250,
                 inject_some_extra_i=False, g_ff_coef=1,
                 symmetric_sequence=False, p_rev=0., extra_recorded_nrns=False,
                 limit_syn_numbers=False, continuous_ass=False,
                 use_random_conn_ff=False, modified_contin=False):
        """
            Ne: number of excitatory neurons
            r_ie: ration of Ni/Ne
            cp_yx: connection probability from x to y
            if type_ext_input=='pois': ext_input={'N_p:10000','f_p':25,
                                        'coef_ep':1., 'sp':.02}
            !!!
            due to current limitations (that I wanna set all g_ee once and
            not to care of which how much it is), currently g_ff_coef can take
            only integer values, if I want a strong synapse, I just put several
            normal ones!
        """
        ########################################################################
        # define a bunch of consts
        self.timestep = .1 * ms  # simulation time step
        self.D = 2 * ms  # AP delay
        self.m_ts = 1. * ms  # monitors time step

        if Ne > 0:
            self.r_ie = (Ni + .0) / Ne  # ratio Ni/Ne
        else:
            self.r_ie = .0
        self.Ne = Ne
        self.Ni = Ni
        self.N = self.Ne + self.Ni
        # set some random connectivity for all E,I neurons
        self.cp_ee = cp_ee
        self.cp_ie = cp_ie
        self.cp_ei = cp_ei
        self.cp_ii = cp_ii
        # conductances
        self.g_ee = g_ee
        self.g_ie = g_ie
        self.g_ei = g_ei
        self.g_ii = g_ii
        self.g_max = g_max

        self.g_ff_coef = int(g_ff_coef)
        self.g_l = g_l
        self.use_random_conn_ff = use_random_conn_ff

        self.type_ext_input = type_ext_input
        self.ext_input = ext_input

        self.limit_syn_numbers = limit_syn_numbers
        self.n_chains = n_chains
        self.n_ass = n_ass  # number of assemblies in the ffn/minimum 2
        self.s_ass = s_ass  # neurons in an assembly
        self.s_assinh = int(self.s_ass * self.r_ie)

        self.cf_ffn = cf_ffn  # strength of ffn synaptic connections
        self.cf_rec = cf_rec  # strength of rec synaptic connections

        # recurrent connection probabilities into a group
        self.pr_ee = pr  # e to e
        self.pr_ie = pr  # e to i
        self.pr_ei = pr
        self.pr_ii = pr
        # FF connection probabilities
        self.pf_ee = pf
        self.pf_ie = 0  # pf
        self.pf_ei = 0  # pf
        self.pf_ii = 0  # pf
        # FB maybe?
        self.symmetric_sequence = symmetric_sequence
        self.continuous_ass = continuous_ass
        self.synapses_per_nrn = synapses_per_nrn

        self.modified_contin = modified_contin

        self.sh_e = 0
        self.sh_i = 0

        # neurons and groups to measure from
        self.nrn_meas_e = []
        self.nrn_meas_i = []
        # neuron groups for spike time measure (for cv and ff)
        if True:
            self.nrngrp_meas = [0, 5, self.n_ass - 1]
            self.n_spike_m_gr = min(50, int(self.s_ass))

            # temporal recording from ps neurons
            self.nrn_meas_e.append(0 * self.s_ass)
            self.nrn_meas_e.append(1 * self.s_ass)
            self.nrn_meas_e.append(2 * self.s_ass)
            self.nrn_meas_e.append(3 * self.s_ass)
            self.nrn_meas_e.append((self.n_ass - 1) * self.s_ass - 1)
            self.nrn_meas_e.append((self.n_ass - 1) * self.s_ass + 1)
            self.nrn_meas_e.append(self.n_ass * self.s_ass - 1)

            # put a few neurons to measure for F2 plots
            for i in range(50):
                self.nrn_meas_e.append(self.n_ass * self.s_ass - 50 - i)
            self.nrn_meas_i.append(1 * self.s_assinh - 1)

        self.nrn_meas_e.append(self.Ne - 1)
        self.nrn_meas_i.append(self.Ni - 1)

        if extra_recorded_nrns:
            # record extra all nrns in second, last assembly and random nrns
            for i in range(self.s_ass):
                self.nrn_meas_e.append(1 * self.s_ass + i)
            for i in range(self.s_ass):
                self.nrn_meas_e.append((self.n_ass - 1) * self.s_ass + i)
            for i in range(self.s_ass):
                self.nrn_meas_e.append(self.n_ass * self.s_ass + i)

        self.p_ass = []
        self.p_assinh = []
        self.p_ass_index = []
        self.p_assinh_index = []

        self.dummy_ass_index = []  # index of non-PS neurons, size is s_ass
        # then function to apply them (later)

        self.dummy_group = []
        self.C_ed = []

        self.inject_some_extra_i = inject_some_extra_i
        self.p_rev = p_rev
        # define variables..needed??
        self.network = None
        self.Pe = None
        self.Pi = None
        self.C_ee = None
        self.C_ie = None
        self.C_ei = None
        self.C_ii = None

        self.cee = None
        self.cie = None
        self.cei = None
        self.cii = None
        self.mon_rate_e = None
        self.mon_rate_i = None

        self.Isine = None
        self.Isini = None

        self.pf_ee_new = None
        self.C_ee_ff = None

        self.P_poisson = None

        self.mon_spike_e = None
        self.mon_spike_i = None

        self.mon_spike_sngl = None
        self.mon_spike_gr = None

        self.mon_volt_e = None
        self.mon_volt_i = None
        self.mon_econd_e = None
        self.mon_icond_e = None
        self.mon_econd_i = None
        self.mon_icond_i = None
        self.mon_ecurr_e = None
        self.mon_icurr_e = None
        self.mon_ecurr_i = None
        self.mon_icurr_i = None

        self.create_net()

        print('initiated ', asctime())

    def create_net(self):
        """ create a network with and connect it"""
        self.network = bb.Network()
        self.network.clock = bb.Clock(dt=self.timestep)

        # create a couple of groups
        # noinspection PyTypeChecker
        self.Pe = bb.NeuronGroup(self.Ne, eqs_exc, threshold='v > -50 * mV',
                                 reset='v = -60 * mV', refractory=2. * ms, method='euler')
        # noinspection PyTypeChecker
        self.Pi = bb.NeuronGroup(self.Ni, eqs_inh, threshold='v > -50 * mV',
                                 reset='v = -60 * mV', refractory=2. * ms, method='euler')

        self.Pe.v = (-65 + 15 * np.random.rand(self.Ne)) * mV
        self.Pi.v = (-65 + 15 * np.random.rand(self.Ni)) * mV
        # noinspection PyTypeChecker
        self.network.add(self.Pe, self.Pi)
        if self.inject_some_extra_i:
            self.network.add(inject)

        if self.type_ext_input == 'curr':
            self.set_in_curr([self.Pe, self.Pi])
        else:
            raise NotImplementedError('no input, sure about it?')

        self.C_ee = bb.Synapses(self.Pe, self.Pe, model='w:siemens', on_pre='ge+=w')
        self.C_ie = bb.Synapses(self.Pe, self.Pi, model='w:siemens', on_pre='ge+=w')
        self.C_ii = bb.Synapses(self.Pi, self.Pi, model='w:siemens', on_pre='gi+=w')
        stdp_on = True
        if stdp_on:
            namespace = {'g_ei': self.g_ei, 'eta_p': eta.eta}
            self.C_ei = bb.Synapses(self.Pi, self.Pe,
                                    model=eq_stdp, on_pre=eq_pre, on_post=eq_post,
                                    namespace=namespace)
        else:
            self.C_ei = bb.Synapses(self.Pi, self.Pe,
                                    model='w:siemens', on_pre='gi_post+=w')

    def gen_ordered(self):
        """
            Generate n assemblies where neurons are ordered
            sh_e, sh_i : shift of e/i neurons (by default order starts at 0)
        """
        if self.n_chains:
            self.sh_e += self.s_ass * self.n_ass
            self.sh_i += self.s_assinh * self.n_ass
        nrn_e = np.arange(self.sh_e, self.Ne)
        nrn_i = np.arange(self.sh_i, self.Ni)
        p_ind_e = [nrn_e[n * self.s_ass:(n + 1) * self.s_ass] for n in range(self.n_ass)]
        p_ind_i = [nrn_i[n * self.s_assinh:(n + 1) * self.s_assinh] for n in range(self.n_ass)]
        print('An ordered sequence is created')
        return p_ind_e, p_ind_i

    def gen_no_overlap(self):
        """
            Generate n assemblies with random neurons
            no repetition of a neuron is allowed
        """
        nrn_perm_e = np.random.permutation(self.Ne)
        nrn_perm_i = np.random.permutation(self.Ni)
        p_ind_e = [nrn_perm_e[n * self.s_ass:(n + 1) * self.s_ass] for n in range(self.n_ass)]
        p_ind_i = [nrn_perm_i[n * self.s_assinh:(n + 1) * self.s_assinh] for n in range(self.n_ass)]
        print('A random sequence without overlaps is created')
        return p_ind_e, p_ind_i

    def gen_ass_overlap(self):
        """
            Generate a n assemblies with random neurons
            repetitions of a neuron in different groups is allowed
        """
        # permutate and pick the first s_ass elements..
        p_ind_e = [np.random.permutation(self.Ne)[:self.s_ass]
                   for _ in range(self.n_ass)]
        p_ind_i = [np.random.permutation(self.Ni)[:self.s_assinh]
                   for _ in range(self.n_ass)]
        print('A random sequence without repetition in a group is created')
        return p_ind_e, p_ind_i

    def gen_random(self):
        """
            Generate a n assemblies with random neurons, repetitions in a
            group are allowed
        """
        p_ind_e = np.random.randint(self.Ne, size=(self.n_ass, self.s_ass))
        p_ind_i = np.random.randint(self.Ni, size=(self.n_ass, self.s_assinh))
        print('A sequence with completely random neurons is created')
        return p_ind_e, p_ind_i

    def gen_dummy(self, p_ind_e_out):
        dum = []
        indexes_flatten = np.array(p_ind_e_out).flatten()
        # not to generate a random number for each neurons
        permutated_numbers = np.random.permutation(self.Ne)
        dum_size = 0
        for nrn_f in permutated_numbers:
            if nrn_f not in indexes_flatten:
                dum.append(nrn_f)
                dum_size += 1
                if dum_size >= self.s_ass:
                    break
        return dum

    def generate_ps_assemblies(self, ass_randomness='gen_no_overlap'):
        """
            generates assemblies of random neurons,
            neurons can lie into several group, but once into the same group
            ass_randomness : how to pick the neurons
                    gen_ordered     : ordered assemblies
                    gen_no_overlap  : random assemblies, no overlap
                    gen_ass_overlap : random assemlies with overlap
                    gen_random      : totally random choise of neurons

        """
        p_ind_e_out, p_ind_i_out = eval("self." + ass_randomness)()
        self.p_ass_index.append(p_ind_e_out)
        self.p_assinh_index.append(p_ind_i_out)
        self.dummy_ass_index.append(self.gen_dummy(p_ind_e_out))
        self.n_chains += 1

    def create_random_matrix(self, pre_nrns, post_nrns, p, pre_is_post=True):
        """
            creates random connections between 2 populations of size
            pre_nrns and post_nrns (population sizes)
            might be slow but allows us to edit the connectivity matrix
            before throwing it into the ruthless synapse class
            ith element consists of the postsynaptic connection of ith nrn
            pre_is_post : flag that prevents a neuron to connect to itself
                if set to True
        """
        conn_mat = []
        for i in range(pre_nrns):
            conn_nrn = list(np.arange(post_nrns)
                            [np.random.random(post_nrns) < p])
            if i in conn_nrn and pre_is_post:  # no autosynapses
                conn_nrn.remove(i)
            conn_mat.append(conn_nrn)
        return conn_mat

    def make_connections_discrete(self):
        for n_ch in range(self.n_chains):  # iterate over sequences
            p_index = self.p_ass_index[n_ch]
            p_indexinh = self.p_assinh_index[n_ch]
            # iterate over the assemblies in the PS
            for n_gr in range(len(p_indexinh)):
                # iterate over E neurons in a group
                for p1 in p_index[n_gr]:
                    # E to E recurrent
                    p1_post = list(p_index[n_gr][
                                       np.random.random(len(p_index[n_gr])) < self.pr_ee])
                    if p1 in p1_post:  # no autosynapse
                        p1_post.remove(p1)
                    # if remove_old_conn_flag_ee:
                    #     cee[p1] = cee[p1][len(p1_post):]
                    #     if p1 < 5:
                    #         print(n_gr, p1, len(p1_post))
                    self.cee[p1].extend(p1_post)
                    # E to E feedforward
                    if n_gr < self.n_ass - 1:  # in case it's the last group
                        ###################################################
                        # flag for using the random connections for ff
                        # instead of embedding new ff synapses, strengthen
                        # the background connections proportionally
                        use_random_conn_ff = False
                        if use_random_conn_ff:
                            p1_post = np.intersect1d(self.cee[p1],
                                                     p_index[n_gr + 1])
                            for _ in range(int(self.pf_ee / self.cp_ee)):
                                # noinspection PyTypeChecker
                                self.cee[p1].extend(p1_post)  # FIX
                            # check for postsynaptic partners of p1 in cee
                            # do the same synapses pff/r_rand times?
                        else:
                            for _ in range(self.g_ff_coef):
                                p1_post = list(p_index[n_gr + 1]
                                               [np.random.random(len(p_index[n_gr + 1]))
                                                < self.pf_ee])
                                if p1 in p1_post:  # no autosynapse
                                    p1_post.remove(p1)

                                self.cee[p1].extend(p1_post)
                    # E to E reverse
                    if self.symmetric_sequence and n_gr:
                        p1_post = list(p_index[n_gr - 1][
                                           np.random.random(len(p_index[n_gr - 1])) <
                                           self.p_rev])
                        if p1 in p1_post:  # no autosynapse
                            p1_post.remove(p1)

                    # E to I recurrent
                    p1_post = list(p_indexinh[n_gr][
                                       np.random.random(len(p_indexinh[n_gr])) < self.pr_ie])

                    self.cie[p1].extend(p1_post)
                for i1 in p_indexinh[n_gr]:
                    # I to I recurrent
                    i1_post = list(p_indexinh[n_gr][
                                       np.random.random(len(p_indexinh[n_gr])) < self.pr_ii])
                    # np.random.random(len(p_indexinh[n_gr]))<pr_ii])
                    if i1 in i1_post:  # no autosynapse
                        i1_post.remove(i1)

                    self.cii[i1].extend(i1_post)

                    # I to E recurrent
                    i1_post = list(p_index[n_gr][
                                       np.random.random(len(p_index[n_gr])) < self.pr_ei])

                    self.cei[i1].extend(i1_post)

    @staticmethod
    def find_post(p_ind, ix, ran_be_cont, ran_af_cont, pr):
        """
            hw stands for half width (M/2) normally 250 neurons
            range variables specify the range of connectivity from
            neuron i,i.e., to how many neurons will neuron i project
                ran_be: range before neuron
                ran_af: range after

        """
        # rns from first group will have higher rc connection to
        # the following half group
        if ix < ran_be_cont:
            pr_n = (ran_be_cont + ran_af_cont) / (ran_af_cont + ix) * pr
            p1_post_f = p_ind[0:ix + ran_af_cont][
                np.random.random(ix + ran_af_cont) < pr_n]
        # last neurons also need some special care to connect
        elif ix > len(p_ind) - ran_af_cont:
            pr_n = pr * (ran_be_cont + ran_af_cont) / (ran_af_cont + len(p_ind) - ix - 1)
            p1_post_f = p_ind[ix - ran_be_cont:][
                np.random.random(len(p_ind) - ix + ran_be_cont) < pr_n]
            print('aa', len(p_ind), ix, ran_be_cont, ran_af_cont, pr_n)
            print(len(p_ind[ix - ran_be_cont:]), len(p_ind) - ix + ran_be_cont)
        # most neurons are happy
        else:
            pr_n = pr
            p1_post_f = p_ind[ix - ran_be_cont:ix + ran_af_cont][
                np.random.random(ran_be_cont + ran_af_cont) < pr_n]
        return p1_post_f

    def make_connections_continuous(self):


        for n_ch in range(self.n_chains):  # iterate over sequences
            p_index = np.array(self.p_ass_index[n_ch]).flatten()
            p_indexinh = np.array(self.p_assinh_index[n_ch]).flatten()
            ran_be = 1 * self.s_ass / 2  # here positive means before..to fix!
            ran_af = 1 * self.s_ass / 2
            ran_be_i = self.s_assinh / 2 + 1
            ran_af_i = self.s_assinh / 2 + 1
            if self.modified_contin:
                ran_ff_start = 1 * self.s_ass / 2
                ran_ff_end = 3 * self.s_ass / 2
            else:
                raise NotImplementedError("must define ran_ff_end")
            # iterate over the assemblies in the PS
            for i, p1 in enumerate(p_index):
                # E-to-E recurrent
                p1_post = self.find_post(p_index, i, ran_be, ran_af, self.pr_ee)
                # if p1 in p1_post: # no autosynapse
                # p1_post = list(p1_post).remove(p1)
                self.cee[p1].extend(p1_post)

                # E-to-I recurrent
                p1_post = self.find_post(p_indexinh, i / 4, ran_be_i, ran_af_i,
                                    self.pr_ie)
                self.cie[p1].extend(p1_post)

                # E-to-E feedforward
                if i < len(p_index) - ran_ff_end:
                    p1_post = p_index[i + ran_ff_start:i + ran_ff_end][
                        np.random.random(ran_ff_end - ran_ff_start)
                        < self.pf_ee]
                # here not to miss connections to the last group
                else:
                    p1_post = p_index[i:len(p_index)][
                        np.random.random(len(p_index) - i) < self.pf_ee]
                self.cee[p1].extend(p1_post)

            for i, i1 in enumerate(p_indexinh):
                # I-to-E recurrent
                i1_post = self.find_post(p_index, 4 * i,
                                    ran_be, ran_af, self.pr_ei)
                self.cei[i1].extend(i1_post)

                # I-to-I recurrent
                i1_post = self.find_post(p_indexinh, i, ran_be_i, ran_af_i,
                                    self.pr_ii)
                # if i1 in i1_post: # no autosynapse
                # i1_post = list(i1_post).remove(i1)
                self.cii[i1].extend(i1_post)

    def apply_connection_matrix(self, S, conn_mat, f_ee=False):
        """
            creates the synapses by applying conn_mat connectivity matrix
            to the synaptic class S
            basically does the following but fast!

            for i, conn_nrn in enumerate(conn_mat):
                for j in conn_nrn:
                    S[i,j]=True

            f_ee is a flag indicating e-e connections

        """
        presynaptic, postsynaptic = [], []
        synapses_pre = {}
        nsynapses = 0
        for i in range(len(conn_mat)):
            conn_nrn = conn_mat[i]
            k1 = len(conn_nrn)
            # too connected? get rid of older synapses
            if self.limit_syn_numbers and f_ee and (k1 > self.synapses_per_nrn):
                x = max(self.synapses_per_nrn, k1 - self.synapses_per_nrn)
                conn_nrn = conn_nrn[-x:]  # simply cut!
                '''
                # some exponential forgeting of old synapses
                tau = (k1-self.synapses_per_nrn)/2.
                conn_nrn = np.array(conn_nrn)[\
                    np.exp(-np.arange(k1)/tau)<np.random.random(k1)]
                '''
            k = len(conn_nrn)  # new number of postsynaptic connections
            # just print to keep an eye on what's going on
            # if i<20:
            # print '# synpapses before and after ', k1,k
            if k:
                synapses_pre[i] = nsynapses + np.arange(k)
                presynaptic.append(i * np.ones(k, dtype=int))
                postsynaptic.append(conn_nrn)
                nsynapses += k
        presynaptic = np.hstack(presynaptic)
        postsynaptic = np.hstack(postsynaptic)
        S.connect(i=presynaptic, j=postsynaptic)

    def set_net_connectivity(self):
        """sets connections in the network"""
        # creates randomly connected matrices
        self.cee = self.create_random_matrix(self.Ne, self.Ne, self.cp_ee, True)
        self.cie = self.create_random_matrix(self.Ne, self.Ni, self.cp_ie, False)
        self.cei = self.create_random_matrix(self.Ni, self.Ne, self.cp_ei, False)
        self.cii = self.create_random_matrix(self.Ni, self.Ni, self.cp_ii, True)

        # seems that these 2 flags are outdated and unusable; can't bother to
        # remove them now
        # remove_old_conn_flag_ee = False
        # remove_old_conn_flag = False

        if self.continuous_ass:
            self.make_connections_continuous()
        else:
            self.make_connections_discrete()

        self.apply_connection_matrix(self.C_ee, self.cee, True)
        self.apply_connection_matrix(self.C_ie, self.cie)
        self.apply_connection_matrix(self.C_ei, self.cei)
        self.apply_connection_matrix(self.C_ii, self.cii)

        self.C_ee.w = self.g_ee
        self.C_ie.w = self.g_ie
        self.C_ei.w = self.g_ei
        self.C_ii.w = self.g_ii
        self.C_ee.delay = self.D
        self.C_ie.delay = self.D
        self.C_ei.delay = self.D
        self.C_ii.delay = self.D
        self.network.add(self.C_ee)
        self.network.add(self.C_ie)
        self.network.add(self.C_ei)
        self.network.add(self.C_ii)

        print('connections imprinted! ', asctime())

    def boost_pff(self, pf_ee_new):
        """
            creates anew connectivity matrix and applies to code
            for new ff connections that should be added after some
            simulation time

        """

        def get_disc_conn():
            conn_mat_f = [[] for _ in range(self.Ne)]
            # E to E feedforward
            for ch in range(self.n_chains):
                p_index = self.p_ass_index[ch]
                for gr_f in range(self.n_ass - 1):
                    for p1 in p_index[gr_f]:
                        p1_post = list(p_index[gr_f + 1]
                                       [np.random.random(len(p_index[gr_f + 1]))
                                        < self.pf_ee_new])
                        conn_mat_f[p1].extend(p1_post)
            return conn_mat_f

        def get_cont_conn():
            conn_mat_f = [[] for _ in range(self.Ne)]
            if self.modified_contin:
                ran_ff_start = 1 * self.s_ass / 2
                ran_ff_end = 3 * self.s_ass / 2
            else:
                raise NotImplementedError("must define ran_ff_End")
            for ch in range(self.n_chains):
                p_index = np.array(self.p_ass_index[ch]).flatten()
                for ix, p1 in enumerate(p_index):
                    # E-to-E feedforward
                    if self.modified_contin:
                        if ix < len(p_index) - ran_ff_end:
                            p1_post = p_index[ix + ran_ff_start:ix + ran_ff_end][
                                np.random.random(ran_ff_end - ran_ff_start)
                                < self.pf_ee_new]
                        # here not to miss connections to the last group 
                        elif ix < len(p_index) - ran_ff_start:
                            p1_post = p_index[ix + ran_ff_start:len(p_index)][
                                np.random.random(len(p_index) - ix - ran_ff_start)
                                < self.pf_ee_new]
                        else:
                            p1_post = []
                    else:
                        if ix < len(p_index) - self.s_ass:
                            p1_post = p_index[ix:ix + self.s_ass][
                                np.random.random(self.s_ass) < self.pf_ee_new]
                        # here not to miss connections to the last group 
                        else:
                            p1_post = p_index[ix:len(p_index)][
                                np.random.random(len(p_index) - ix) < self.pf_ee_new]
                    conn_mat_f[p1].extend(p1_post)
            return conn_mat_f

        def get_rand_boost():
            ex_pre = np.array(self.C_ee.presynaptic)
            ex_post = np.array(self.C_ee.postsynaptic)

            conn_mat_f = [[] for _ in range(self.Ne)]
            for ch in range(self.n_chains):
                p_index = self.p_ass_index[ch]
                for gr_f in range(self.n_ass - 1):
                    for p1 in p_index[gr_f]:
                        p1_ex_post = ex_post[ex_pre == p1]
                        p1_post = np.intersect1d(
                            self.p_ass_index[0][gr_f + 1], p1_ex_post)
                        for _ in range(int(self.pf_ee_new / self.cp_ee)):
                            # noinspection PyTypeChecker
                            conn_mat_f[p1].extend(p1_post)  # FIX
                        if not gr_f and not p1:
                            print(p1, p1_post)
                            print()
                            # 1/0
            return conn_mat_f

        self.pf_ee_new = pf_ee_new
        self.C_ee_ff = bb.Synapses(self.Pe, self.Pe,
                                   model='w:siemens', on_pre='ge+=w')
        if self.continuous_ass:
            conn_mat = get_cont_conn()
        else:
            if self.use_random_conn_ff:
                conn_mat = get_rand_boost()
            else:
                conn_mat = get_disc_conn()

        presynaptic, postsynaptic = [], []
        synapses_pre = {}
        nsynapses = 0
        for i in range(len(conn_mat)):
            conn_nrn = conn_mat[i]
            k = len(conn_nrn)  # new number of postsynaptic connections
            if k:
                synapses_pre[i] = nsynapses + np.arange(k)
                presynaptic.append(i * np.ones(k, dtype=int))
                postsynaptic.append(conn_nrn)
                nsynapses += k
        presynaptic = np.hstack(presynaptic)
        postsynaptic = np.hstack(postsynaptic)

        self.C_ee_ff.create_synapses(presynaptic, postsynaptic, synapses_pre)
        self.C_ee_ff.w = self.g_ee
        self.C_ee_ff.delay = self.D
        self.network.add(self.C_ee_ff)
        print('pff boosted!')

    def balance(self, bal_time=2 * second, eta_c=1.):
        """
        balancing function: runs the network for bal_time and:
        1) sets the learning rate to eta
        2) !!! switches off the spike recorder (ap_record = False)

        """
        t0 = time()
        eta.eta = eta.v * eta_c
        self.network.run(bal_time)
        eta.eta = 0.0
        t1 = time()
        print('balanced: ', t1 - t0)

    def run_sim(self, run_time=1 * second):
        """ runs the network for run_time with I plasticity turned off"""
        t0 = time()
        eta.eta = 0.0
        self.network.run(run_time)
        t1 = time()
        print('run: ', t1 - t0)

    def set_in_curr(self, target, ext_input=None):
        """ ce,ci currents injected in E/I populations"""
        if ext_input is None:
            ext_input = self.ext_input
        for t in target:
            t.I = ext_input

    def set_in_poisson(self, target):
        """
            Set poissonian input to a group of neurons
            target: list of targert groups
            N_p: # of poissons inputs
            f_p: frequency of P
            sp: sparseness of connections
            coef_ep: factor of ep conductance to g_exc

        """
        # somehow PoissonInput is way slower! also leads to diff behaviour
        # for gr in target:
        # inp_poisson = bb.PoissonInput(gr,N=100,rate=f_p,
        # weight=2.*self.g_ee,state='ge')
        # self.network.add(inp_poisson)
        N_p = self.ext_input['N_p']
        f_p = self.ext_input['f_p']
        sp = self.ext_input['sp']
        coef_ep = self.ext_input['coef_ep']
        self.P_poisson = bb.PoissonGroup(N_p, f_p, clock=self.network.clock)
        self.network.add(self.P_poisson)
        for gr_f in target:
            Cep = bb.Synapses(self.P_poisson, gr_f, model='w:siemens', on_pre='ge+=w')
            Cep.connect(p=sp)
            Cep.w = coef_ep * self.g_ee
            self.network.add(Cep)

    def set_syn_input(self, target, time_f: bb.Quantity):
        """adding sync inputs at some time points"""
        # noinspection PyTypeChecker
        ext_in = bb.SpikeGeneratorGroup(1, indices=bb.array([0]), times=bb.array([time_f]), clock=self.network.clock)
        C_syne = bb.Synapses(ext_in, target, model='w:siemens', on_pre='ge+=w')
        C_syne.connect_random(ext_in, target, sparseness=1.)
        C_syne.w = 30. * self.g_ee
        self.network.add(ext_in, C_syne)

    def set_syn_input_ran(self, target, time_f):
        """adding sync inputs at some time points"""
        # noinspection PyTypeChecker
        ext_in = bb.SpikeGeneratorGroup(1, indices=[0], times=[time_f], clock=self.network.clock)
        C_syne = bb.Synapses(ext_in, self.Pe, model='w:siemens', on_pre='ge+=w')
        for n in target:
            C_syne.connect_random(ext_in, self.Pe[n], sparseness=1.)
        C_syne.w = 30. * self.g_ee
        self.network.add(ext_in, C_syne)

    def set_noisy_input(self, target, time_f, sigma=0., mcoef=30):
        """adding sync inputs at some time points with
            normal jitter distribution sigma

            mcoef is the strength of stimulation

        """
        t0 = time_f - 6. * sigma  # mean delay is set to 6*sigma
        # noinspection PyTypeChecker
        ext_in = bb.SpikeGeneratorGroup(1, indices=[0], times=[t0], clock=self.network.clock)
        C_syne = bb.Synapses(ext_in, self.Pe, model='w:siemens', on_pre='ge+=w')

        C_syne.connect(i=np.zeros_like(target), j=target)
        C_syne.w = mcoef * self.g_ee
        if sigma > 0.:
            C_syne.delay = np.random.normal(6. * sigma, sigma, len(target))
        else:
            C_syne.delay = np.zeros(len(target))
        self.network.add(ext_in, C_syne)

    def set_rate_monitor(self):
        """yep"""
        self.mon_rate_e = bb.PopulationRateMonitor(self.Pe)
        self.mon_rate_i = bb.PopulationRateMonitor(self.Pi)
        self.network.add(self.mon_rate_e, self.mon_rate_i)

    def set_spike_monitor(self):
        """yep"""
        self.mon_spike_e = bb.SpikeMonitor(self.Pe)
        self.mon_spike_i = bb.SpikeMonitor(self.Pi)
        self.network.add(self.mon_spike_e, self.mon_spike_i)

    def set_group_spike_monitor(self, ch=0):
        """
            !!!
            this would not work with random assemblies
            to be removed in the future
        """
        self.mon_spike_sngl = []  # measure spike times from a few single neurons
        for nrn_f in self.nrn_meas_e:
            self.mon_spike_sngl.append(bb.SpikeMonitor(self.Pe[nrn_f]))
        self.network.add(self.mon_spike_sngl)

        self.mon_spike_gr = []  # measure spike times from groups (for CV and FF)
        for gr_f in self.nrngrp_meas:
            self.mon_spike_gr.append(bb.SpikeMonitor(
                self.p_ass[ch][gr_f][0:self.n_spike_m_gr]))
        # also control group of neurons which is not included in the ps
        self.mon_spike_gr.append(bb.SpikeMonitor(
            self.Pe[self.n_ass * self.s_ass:(self.n_ass + 1) * self.s_ass]
            [0:self.n_spike_m_gr]))
        self.network.add(self.mon_spike_gr)
        # default spike easure is off
        for sp in self.mon_spike_gr:
            sp.record = False

    def set_voltage_monitor(self):
        """yep"""
        self.mon_volt_e = bb.StateMonitor(self.Pe, 'v', record=self.nrn_meas_e)
        self.mon_volt_i = bb.StateMonitor(self.Pi, 'v', record=self.nrn_meas_i)
        self.network.add(self.mon_volt_e, self.mon_volt_i)

    def set_conductance_monitor(self):
        """yep"""
        self.mon_econd_e = bb.StateMonitor(self.Pe, 'ge', record=self.nrn_meas_e)
        self.mon_icond_e = bb.StateMonitor(self.Pe, 'gi', record=self.nrn_meas_e)
        self.mon_econd_i = bb.StateMonitor(self.Pi, 'ge', record=self.nrn_meas_i)
        self.mon_icond_i = bb.StateMonitor(self.Pi, 'gi', record=self.nrn_meas_i)
        self.network.add(self.mon_econd_e, self.mon_icond_e,
                         self.mon_econd_i, self.mon_icond_i)

    def set_current_monitor(self):
        """yep"""
        self.mon_ecurr_e = bb.StateMonitor(self.Pe, 'Ie', record=self.nrn_meas_e)
        self.mon_icurr_e = bb.StateMonitor(self.Pe, 'Ii', record=self.nrn_meas_e)
        self.mon_ecurr_i = bb.StateMonitor(self.Pi, 'Ie', record=self.nrn_meas_i)
        self.mon_icurr_i = bb.StateMonitor(self.Pi, 'Ii', record=self.nrn_meas_i)
        self.network.add(self.mon_ecurr_e, self.mon_icurr_e,
                         self.mon_ecurr_i, self.mon_icurr_i)

    # noinspection PyUnresolvedReferences
    def run_full_sim(self, sim_times):
        self.generate_ordered_ps()  # FIX
        self.set_ffchain_new()

        self.set_rate_monitor()
        self.set_group_spike_monitor()

        stim_times = np.arange(sim_times['start_sim'], sim_times['stop_sim'], 1)
        for t in stim_times:
            self.set_syn_input(self.p_ass[0][0], t * second)

        # stimulation with a que (not full)
        for que in [80, 60, 40, 20]:
            start_que = sim_times['start_sim' + str(que)]
            stop_que = sim_times['stop_sim' + str(que)]
            que_res = que / 100.  # # 80,60,40,20% of pop stimulation

            for t in range(start_que, stop_que):
                n_sim_nrn = int(que_res * self.s_ass)
                self.set_syn_input(self.p_ass[0][0][0:n_sim_nrn], t * second)

        # set balance times with corresponding learning rates
        t0 = 0
        for t, r in zip(sim_times['balance_dur'], sim_times['balance_rate']):
            self.balance((t - t0) * second, r)
            t0 = t

        # run the simulations
        self.run_sim((sim_times['stop_sim20'] - sim_times['start_sim']) * second)
        # turn on the group spike monitor
        for sp in self.mon_spike_gr:
            sp.record = True
            # run for spontan activity
        self.run_sim((sim_times['stop_spont_recording'] -
                      sim_times['stop_sim20']) * second)

    def dummy(self):
        # noinspection PyDictCreation
        sim_times = {'balance_dur': [10, 15, 20, 25], 'balance_rate': [5, 1, .1, .01], 'start_sim': 16, 'stop_sim': 20,
                     'start_sim80': 20, 'stop_sim80': 20, 'start_sim60': 20, 'stop_sim60': 20, 'start_sim40': 20,
                     'stop_sim40': 20, 'start_sim20': 20, 'stop_sim20': 22, 'start_fr_recording': 16,
                     'stop_fr_recording': 25}
        sim_times['start_spont_recording'] = sim_times['stop_sim20']
        sim_times['stop_spont_recording'] = 25

        self.set_rate_monitor()
        self.set_spike_monitor()
        self.set_voltage_monitor()
        self.set_current_monitor()
        self.set_conductance_monitor()

        self.run_full_sim(sim_times)

    def plot_for_raster_curr_volt(self):
        num_ps = 1
        for n in range(num_ps):
            self.generate_ps_assemblies('gen_ass_overlap')
        self.set_net_connectivity()

        self.set_spike_monitor()
        self.set_rate_monitor()

        '''
        gr = self.p_ass_index[0][0]
        self.set_noisy_input(gr,.5*second,sigma=0*ms)
        #gr1 = self.p_ass_index[1][0]
        #self.set_noisy_input(gr1,.7*second,sigma=0*ms)
        self.balance(1.*second,5.)
        '''

        t0 = 30  # time offset for stimulation in secs
        n_stim = 5
        for n in range(num_ps):
            for i in range(n_stim):
                gr_num_f = int(self.n_ass / 5. * i)
                print('stim to ', gr_num_f)
                gr_f = self.p_ass_index[n][gr_num_f]
                t = (t0 + n + i * 3) * second
                self.set_noisy_input(gr_f, t, sigma=0 * ms)

        self.balance(10 * second, 5.)
        self.balance(10 * second, 1.)
        self.balance(5 * second, .1)
        self.balance(5 * second, .01)
        self.run_sim(16 * second)

        for n in range(num_ps):
            plt.figure(figsize=(12., 8.))
            plotter.plot_ps_raster(self, chain_n=n, frac=.01)

    def test_shifts(self, ie, ii, tr):
        self.generate_ps_assemblies('gen_no_overlap')
        self.set_net_connectivity()
        self.set_spike_monitor()
        self.set_rate_monitor()

        self.Isine = ie * pA
        self.Isini = ii * pA
        self.network.add(inject)

        gr_f = self.p_ass_index[0][0]
        for i in range(9):
            t = (21 + i) * second
            self.set_noisy_input(gr_f, t, sigma=0 * ms)

        self.balance(5 * second, 5.)
        self.balance(5 * second, 1.)
        self.balance(5 * second, .1)
        self.balance(5 * second, .01)
        self.run_sim(10 * second)

        pr, pf = self.pr_ee, self.pf_ee

        plt.figure(figsize=(12., 8.))
        plotter.plot_ps_raster(self, chain_n=0, frac=.1)
        # plt.xlim([6800,8300])

    def stim_curr(self, ps=0, gr_f=0, dur_stim=100, dur_relx=400,
                  curr=10 * pA):
        """
            stimulate group gr in ps with a continuous current

        """
        for nrn_f in self.p_ass_index[ps][gr_f]:
            self.Pe[nrn_f].I += curr
        self.run_sim(dur_stim * ms)
        for nrn_f in self.p_ass_index[ps][gr_f]:
            self.Pe[nrn_f].I -= curr
        self.run_sim(dur_relx * ms)


def test_symm():
    nn_f = Nets(Ne=20000, Ni=5000, cp_ee=.01, cp_ie=.01, cp_ei=0.01, cp_ii=.01,
                n_ass=10, s_ass=500, pr=.15, pf=.03, symmetric_sequence=True, p_rev=.03,
                g_ee=0.1 * nS, g_ie=0.1 * nS, g_ei=0.4 * nS, g_ii=0.4 * nS)

    nn_f.generate_ps_assemblies('gen_no_overlap')
    nn_f.set_net_connectivity()
    nn_f.set_spike_monitor()
    nn_f.set_rate_monitor()

    '''
    gr = nn.p_ass_index[0][0]
    t = 20*second
    nn.set_noisy_input(gr,t,sigma=0*ms)
    t = 20.5*second
    nn.set_noisy_input(gr,t,sigma=0*ms)

    gr = nn.p_ass_index[0][9]
    t = 21*second
    nn.set_noisy_input(gr,t,sigma=0*ms)
    t = 21.5*second
    nn.set_noisy_input(gr,t,sigma=0*ms)

    nn.balance(5*second,5.)
    nn.balance(5*second,1.)
    nn.balance(5*second,.1)
    nn.balance(5*second,.01)
    nn.run_sim(2*second)
    '''

    for gr_num_f in range(nn_f.n_ass):
        gr_f = nn_f.p_ass_index[0][gr_num_f]
        t = (20.55 + gr_num_f * .1) * second
        nn_f.set_noisy_input(gr_f, t, sigma=0 * ms)

    nn_f.balance(5 * second, 5.)
    nn_f.balance(5 * second, 1.)
    nn_f.balance(5 * second, .1)
    nn_f.balance(5 * second, .01)
    # nn.run_sim(4*second)
    nn_f.Pe.I -= .0 * pA

    for nrn_f in nn_f.p_ass_index[0][0]:
        nn_f.Pe[nrn_f].I += 3 * pA
    nn_f.run_sim(.5 * second)
    for nrn_f in nn_f.p_ass_index[0][0]:
        nn_f.Pe[nrn_f].I -= 3 * pA

    nn_f.Pe.I -= 9 * pA
    nn_f.run_sim(1. * second)
    nn_f.Pe.I += 9 * pA

    for nrn_f in nn_f.p_ass_index[0][9]:
        nn_f.Pe[nrn_f].I += 3 * pA
    nn_f.run_sim(.5 * second)
    for nrn_f in nn_f.p_ass_index[0][9]:
        nn_f.Pe[nrn_f].I -= 3 * pA

    plotter.plot_ps_raster(nn_f, chain_n=0, frac=.1)
    plt.xlim([20000, 22000])
    return nn_f


def test_fr():
    pr, pf = 0.06, 0.06

    nn_f = Nets(Ne=20000, Ni=5000, cp_ee=.01, cp_ie=.01, cp_ei=0.01, cp_ii=.01,
                n_ass=10, s_ass=500, pr=pr, pf=pf,
                g_ee=0.1 * nS, g_ie=0.1 * nS, g_ei=0.4 * nS, g_ii=0.4 * nS)

    nn_f.generate_ps_assemblies('gen_no_overlap')
    nn_f.set_net_connectivity()
    nn_f.set_spike_monitor()
    nn_f.set_rate_monitor()

    gr_f = nn_f.p_ass_index[0][0]
    t = 20 * second
    nn_f.set_noisy_input(gr_f, t, sigma=0 * ms)
    t = 21 * second
    nn_f.set_noisy_input(gr_f, t, sigma=0 * ms)

    '''
    nn.balance(.01*second,5.)
    nn.balance(.01*second,1.)
    '''

    nn_f.balance(1 * second, 5.)

    gr_fr_e = calc_spikes.make_fr_from_spikes(nn_f, ps=0, w=1, exc_nrns=True)
    gr_fr_i = calc_spikes.make_fr_from_spikes(nn_f, ps=0, w=1, exc_nrns=False)

    plt.subplot(211)
    for gr_f in range(nn_f.n_ass):
        plt.plot(calc_spikes.gaus_smooth(gr_fr_e[gr_f], 2))
    plt.subplot(212)
    for gr_f in range(nn_f.n_ass):
        plt.plot(calc_spikes.gaus_smooth(gr_fr_i[gr_f], 2))

    plt.show()

    return nn_f


def test_no_ps():
    pr, pf = 0.06, 0.06
    nn_f = Nets(Ne=20000, Ni=5000, cp_ee=.01, cp_ie=.01, cp_ei=0.01, cp_ii=.01,
                n_chains=0, n_ass=2, s_ass=500, pr=pr, pf=pf,
                g_ee=0.1 * nS, g_ie=0.1 * nS, g_ei=0.4 * nS, g_ii=0.4 * nS)

    nn_f.set_net_connectivity()
    nn_f.set_spike_monitor()
    nn_f.set_rate_monitor()

    nn_f.balance(5 * second, 5.)
    nn_f.balance(5 * second, 1.)
    nn_f.balance(5 * second, .2)
    nn_f.balance(5 * second, .05)
    nn_f.run_sim(1 * second)
    return nn_f


# noinspection PyPep8Naming
def test_diff_gff(Ne=20000):
    gfc = 1
    pr = 0.06
    pf = 0.06

    Ni = Ne // 4
    cp = .01

    # the default conductances used for Ne=20000
    ge0 = 0.1 * nS
    gi0 = 0.4 * nS

    gee = ge0 * (20000. / Ne) ** .5
    gii = gi0 * (20000. / Ne) ** .5

    pf = pf * (Ne / 20000.) ** .5
    pr = pr * (Ne / 20000.) ** .5

    continuous_ass = False
    nn_f = Nets(Ne=Ne, Ni=Ni, cp_ee=cp, cp_ie=cp, cp_ei=cp, cp_ii=cp,
                n_ass=10, s_ass=500, pr=pr, pf=pf, ext_input=200 * pA,
                g_ee=gee, g_ie=gee, g_ei=gii, g_ii=gii, g_ff_coef=gfc,
                continuous_ass=continuous_ass)

    # nn.generate_ps_assemblies('')
    nn_f.generate_ps_assemblies('gen_ordered')
    nn_f.set_net_connectivity()
    nn_f.set_spike_monitor()
    nn_f.set_rate_monitor()
    nn_f.set_voltage_monitor()
    nn_f.set_current_monitor()

    gr_f = nn_f.p_ass_index[0][0]

    t = 21 * second
    nn_f.set_noisy_input(gr_f, t, sigma=0 * ms)
    t = 23 * second
    nn_f.set_noisy_input(gr_f, t, sigma=0 * ms)

    nn_f.mon_spike_e.record = False
    nn_f.mon_spike_i.record = False

    nn_f.balance(5 * second, 5.)
    nn_f.balance(5 * second, 1.)
    nn_f.balance(5 * second, .1)
    nn_f.balance(5 * second, .01)
    nn_f.mon_spike_e.record = True
    nn_f.mon_spike_i.record = True
    nn_f.run_sim(5 * second)

    return nn_f


def test_psps():
    """
        test PSPs
    """
    ge0 = 0.1 * nS
    gi0 = 0.4 * nS
    cp = 0
    nn_f = Nets(Ne=10, Ni=2, cp_ee=cp, cp_ie=cp, cp_ei=cp, cp_ii=cp,
                n_ass=0, s_ass=1, pr=0, pf=0, ext_input=0 * pA,
                g_ee=ge0, g_ie=ge0, g_ei=gi0, g_ii=gi0)

    nn_f.C_ee = bb.Synapses(nn_f.Pe, nn_f.Pe, model='w:siemens', on_pre='ge+=w')
    nn_f.C_ee[0, 9] = True
    nn_f.C_ee.w = nn_f.g_ee
    nn_f.C_ee.delay = nn_f.D
    nn_f.network.add(nn_f.C_ee)

    '''
    '''
    target = nn_f.Pe[0]
    # noinspection PyTypeChecker
    ext_in = bb.SpikeGeneratorGroup(1, indices=[0], times=[300] * ms, clock=nn_f.network.clock)
    # noinspection PyPep8Naming
    C_syne = bb.Synapses(ext_in, target, model='w:siemens', on_pre='ge+=w')
    C_syne.connect_random(ext_in, target, sparseness=1.)
    C_syne.w = 130. * nn_f.g_ee
    nn_f.network.add(ext_in, C_syne)
    nn_f.nrn_meas_e = [0, 1, 9]
    nn_f.mon_volt_e = bb.StateMonitor(nn_f.Pe, 'v', record=nn_f.nrn_meas_e)  # ,timestep=1)
    nn_f.network.add(nn_f.mon_volt_e)

    nn_f.run_sim(500 * ms)

    plt.plot(nn_f.mon_volt_e.times / ms,
             nn_f.mon_volt_e[9] / mV)
    plotter.show()

    return nn_f


def test_longseq():
    nn_f = Nets(Ne=20000, Ni=5000, cp_ee=.01, cp_ie=.01, cp_ei=0.01, cp_ii=.01,
                n_ass=444, s_ass=150, pr=.19, pf=.19, synapses_per_nrn=200,
                ext_input=200 * pA, limit_syn_numbers=True,
                g_ee=0.1 * nS, g_ie=0.1 * nS, g_ei=0.4 * nS, g_ii=0.4 * nS
                )
    nn_f.generate_ps_assemblies('gen_ass_overlap')
    nn_f.set_net_connectivity()
    nn_f.set_spike_monitor()
    nn_f.set_rate_monitor()
    nn_f.set_voltage_monitor()
    nn_f.set_current_monitor()

    gr_f = nn_f.p_ass_index[0][0]
    t = 21 * second
    nn_f.set_noisy_input(gr_f, t, sigma=0 * ms, mcoef=30)

    nn_f.mon_spike_e.record = False
    nn_f.mon_spike_i.record = False
    nn_f.balance(5 * second, 5.)
    nn_f.balance(5 * second, 1.)
    nn_f.balance(5 * second, .1)
    nn_f.balance(5 * second, .01)
    nn_f.mon_spike_e.record = True
    nn_f.mon_spike_i.record = True
    nn_f.run_sim(6 * second)

    # plotter.plot_ps_raster(nn, frac=1./150)

    fname = 'longseq444.npz'
    spikes4save = calc_spikes.get_spike_times_ps(nn_f, frac=1. / 150)
    np.savez_compressed(fname, spikes4save)

    return nn_f


def test_2_ass(Ne=20000):
    pr = 0.1
    pf = 0.06

    # noinspection PyPep8Naming
    Ni = Ne // 4
    cp = .01

    # the default conductances used for Ne=20000
    ge0 = 0.1 * nS
    gi0 = 0.4 * nS

    nn_f = Nets(Ne=Ne, Ni=Ni, cp_ee=cp, cp_ie=cp, cp_ei=cp, cp_ii=cp,
                n_ass=10, s_ass=500, pr=pr, pf=pf, ext_input=200 * pA,
                g_ee=ge0, g_ie=ge0, g_ei=gi0, g_ii=gi0)

    nn_f.generate_ps_assemblies('gen_ordered')
    nn_f.generate_ps_assemblies('gen_ordered')
    nn_f.set_net_connectivity()
    nn_f.set_spike_monitor()
    nn_f.set_rate_monitor()
    nn_f.set_voltage_monitor()
    nn_f.set_current_monitor()

    gr0 = nn_f.p_ass_index[0][0]
    gr1 = nn_f.p_ass_index[1][0]

    t = 21 * second
    nn_f.set_noisy_input(gr0, t, sigma=0 * ms)
    t = 22 * second
    nn_f.set_noisy_input(gr1, t, sigma=0 * ms)

    nn_f.mon_spike_e.record = False
    nn_f.mon_spike_i.record = False

    '''
    nn.balance(20*second,5.)
    nn.balance(20*second,1.)
    nn.balance(10*second,.1)
    nn.balance(10*second,.01)
    '''
    nn_f.balance(5 * second, 5.)
    nn_f.balance(5 * second, 1.)
    nn_f.balance(5 * second, .1)
    nn_f.balance(5 * second, .01)
    nn_f.mon_spike_e.record = True
    nn_f.mon_spike_i.record = True
    nn_f.run_sim(5 * second)

    fname = '2asss.npz'
    spikes4save = calc_spikes.get_all_spikes(nn_f)
    np.savez_compressed(fname, np.array(spikes4save))
    return nn_f


def show_ass_frs():
    """
        Plots the firing of sequent assemblies

    """

    nn_f = Nets(Ne=20000, Ni=5000, cp_ee=.01, cp_ie=.01, cp_ei=0.01, cp_ii=.01,
                n_ass=10, s_ass=500, pr=.06, pf=.06,
                ext_input=200 * pA,
                g_ee=0.1 * nS, g_ie=0.1 * nS, g_ei=0.4 * nS, g_ii=0.4 * nS
                )
    nn_f.generate_ps_assemblies('gen_ordered')
    nn_f.set_net_connectivity()
    nn_f.set_spike_monitor()
    nn_f.set_rate_monitor()
    nn_f.set_voltage_monitor()
    nn_f.set_current_monitor()
    nn_f.set_conductance_monitor()

    gr0 = nn_f.p_ass_index[0][0]
    t = 21 * second
    nn_f.set_noisy_input(gr0, t, sigma=0 * ms)
    t = 21.5 * second
    nn_f.set_noisy_input(gr0, t, sigma=0 * ms)
    t = 22. * second
    nn_f.set_noisy_input(gr0, t, sigma=0 * ms)
    t = 22.5 * second
    nn_f.set_noisy_input(gr0, t, sigma=0 * ms)
    t = 23. * second
    nn_f.set_noisy_input(gr0, t, sigma=0 * ms)

    nn_f.mon_spike_e.record = False
    nn_f.mon_spike_i.record = False
    nn_f.balance(5 * second, 5.)
    nn_f.balance(5 * second, 1.)
    nn_f.balance(5 * second, .1)
    nn_f.balance(5 * second, .01)
    nn_f.mon_spike_e.record = True
    nn_f.mon_spike_i.record = True
    nn_f.run_sim(6 * second)
    plotter.plot_gr_fr2(nn_f, wbin=.2, ngroups=8)

    return nn_f


def test_tau():
    nn_f = Nets(Ne=20000, Ni=5000, cp_ee=.01, cp_ie=.01, cp_ei=0.01, cp_ii=.01,
                n_ass=1, s_ass=500, pr=.00, pf=.00,
                ext_input=200 * pA,
                g_ee=0.1 * nS, g_ie=0.1 * nS, g_ei=0.4 * nS, g_ii=0.4 * nS
                )
    nn_f.generate_ps_assemblies('gen_ordered')
    nn_f.set_net_connectivity()
    nn_f.set_spike_monitor()
    nn_f.set_rate_monitor()
    nn_f.set_voltage_monitor()
    nn_f.set_current_monitor()
    nn_f.set_conductance_monitor()

    '''
    gr0 = nn.p_ass_index[0][0]
    t = 21*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 
    t = 21.5*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 
    t = 22.*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 
    t = 22.5*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 
    t = 23.*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 
    t = 23.5*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 
    '''

    nn_f.mon_spike_e.record = False
    nn_f.mon_spike_i.record = False
    nn_f.balance(5 * second, 5.)
    nn_f.balance(5 * second, 1.)
    nn_f.balance(5 * second, .1)
    nn_f.balance(4 * second, .01)
    nn_f.mon_spike_e.record = True
    nn_f.mon_spike_i.record = True
    nn_f.balance(1 * second, .01)

    nstim = 20
    currs = [10 * pA, 20 * pA, 40 * pA, 80 * pA, 150 * pA]
    dur_stim, dur_relx = 100, 400
    dur = dur_stim + dur_relx

    for curr in currs:
        for i in range(nstim):
            nn_f.stim_curr(curr=curr, dur_stim=dur_stim, dur_relx=dur_relx)

    plotter.plot_pop_raster(nn_f)

    nsubs = len(currs)
    mfrl = []
    wbin = .1
    dur_stim, dur_pre = 120, 20
    base_fr = 5.
    plt.figure()
    for i, curr in enumerate(currs):
        tl = 20000 + i * nstim * dur + np.arange(nstim) * dur
        plt.subplot(nsubs, 1, 1 + i)
        mfr = plotter.plot_mean_curr_act(nn_f, tl, dur_stim=dur_stim,
                                         dur_pre=dur_pre, wbin=wbin)
        mfrl.append(calc_spikes.gaus_smooth(mfr, w=wbin, sigma=.2))

        # comment peak_time = np.argmax(mfrl[-1]) * wbin - dur_pre
        peak_value = np.max(mfrl[-1])

        peak80_time = (mfr > base_fr + (.8 * (peak_value - base_fr))).argmax() * wbin - dur_pre
        peak20_time = (mfr > base_fr + (.2 * (peak_value - base_fr))).argmax() * wbin - dur_pre

        time_const = peak80_time - peak20_time
        print('time const is ', time_const)

    plt.show()
    return nn_f


# noinspection PyPep8Naming
def test_boost_pf():
    Ne = 20000
    gfc = 1
    pr = 0.1
    pf = 0.00
    pf_boost = 0.04

    Ni = Ne // 4
    cp = .01

    # the default conductances used for Ne=20000
    ge0 = 0.1 * nS
    gi0 = 0.4 * nS

    gee = ge0 * (20000. / Ne) ** .5
    gii = gi0 * (20000. / Ne) ** .5

    pf = pf * (Ne / 20000.) ** .5
    pr = pr * (Ne / 20000.) ** .5

    neur_net = Nets(Ne=Ne, Ni=Ni, cp_ee=cp, cp_ie=cp, cp_ei=cp, cp_ii=cp,
                    n_ass=10, s_ass=500, pr=pr, pf=pf, ext_input=200 * pA,
                    g_ee=gee, g_ie=gee, g_ei=gii, g_ii=gii, g_ff_coef=gfc,
                    modified_contin=True)

    # nn.generate_ps_assemblies('gen_no_overlap')
    neur_net.generate_ps_assemblies('gen_ordered')
    neur_net.set_net_connectivity()
    neur_net.set_spike_monitor()
    neur_net.set_rate_monitor()
    neur_net.set_voltage_monitor()
    neur_net.set_current_monitor()

    gr_f = neur_net.p_ass_index[0][0]

    t = 19.5 * second
    neur_net.set_noisy_input(gr_f, t, sigma=0 * ms)
    t = 20 * second
    neur_net.set_noisy_input(gr_f, t, sigma=0 * ms)
    t = 22 * second
    neur_net.set_noisy_input(gr_f, t, sigma=0 * ms)
    t = 24 * second
    neur_net.set_noisy_input(gr_f, t, sigma=0 * ms)
    t = 26 * second
    neur_net.set_noisy_input(gr_f, t, sigma=0 * ms)
    t = 28 * second
    neur_net.set_noisy_input(gr_f, t, sigma=0 * ms)
    for i in range(9):
        t = (29 + i) * second
        neur_net.set_noisy_input(gr_f, t, sigma=0 * ms)

    neur_net.mon_spike_e.record = False
    neur_net.mon_spike_i.record = False

    # nn.boost_pff(0.04)
    neur_net.balance(5 * second, 5.)
    neur_net.balance(5 * second, 1.)
    neur_net.balance(5 * second, .1)
    neur_net.balance(4 * second, .01)
    neur_net.mon_spike_e.record = True
    neur_net.mon_spike_i.record = True
    neur_net.balance(1 * second, .01)
    neur_net.boost_pff(pf_boost)
    neur_net.balance(2 * second, 5.)
    neur_net.balance(2 * second, 1.)
    neur_net.balance(2 * second, .1)
    neur_net.balance(2 * second, .01)
    neur_net.run_sim(4 * second)

    return neur_net


# noinspection PyPep8Naming
def test_boost_pf_cont():
    Ne = 20000
    gfc = 1
    pr = 0.08
    pf = 0.00
    pf_boost = 0.04

    Ni = Ne // 4
    cp = .01

    # the default conductances used for Ne=20000
    ge0 = 0.1 * nS
    gi0 = 0.4 * nS

    gee = ge0 * (20000. / Ne) ** .5
    gii = gi0 * (20000. / Ne) ** .5

    pf = pf * (Ne / 20000.) ** .5
    pr = pr * (Ne / 20000.) ** .5

    nn_f = Nets(Ne=Ne, Ni=Ni, cp_ee=cp, cp_ie=cp, cp_ei=cp, cp_ii=cp,
                n_ass=10, s_ass=500, pr=pr, pf=pf, ext_input=200 * pA,
                g_ee=gee, g_ie=gee, g_ei=gii, g_ii=gii, g_ff_coef=gfc,
                continuous_ass=True, modified_contin=True)

    # nn.generate_ps_assemblies('gen_no_overlap')
    nn_f.generate_ps_assemblies('gen_ordered')
    nn_f.set_net_connectivity()
    nn_f.set_spike_monitor()
    nn_f.set_rate_monitor()
    nn_f.set_voltage_monitor()
    nn_f.set_current_monitor()

    gr_f = nn_f.p_ass_index[0][0]
    '''
    '''
    t = 19. * second
    nn_f.set_noisy_input(gr_f, t, sigma=0 * ms)
    t = 19.5 * second

    nn_f.set_noisy_input(gr_f, t, sigma=0 * ms)
    for i in range(9):
        t = (28 + i) * second
        nn_f.set_noisy_input(gr_f, t, sigma=0 * ms)

    nn_f.mon_spike_e.record = False
    nn_f.mon_spike_i.record = False

    nn_f.balance(5 * second, 5.)
    nn_f.balance(5 * second, 1.)
    nn_f.balance(5 * second, .1)
    nn_f.balance(4 * second, .01)
    nn_f.mon_spike_e.record = True
    nn_f.mon_spike_i.record = True
    nn_f.balance(1 * second, .01)
    nn_f.boost_pff(pf_boost)
    nn_f.balance(2 * second, 5.)
    nn_f.balance(2 * second, 1.)
    nn_f.balance(2 * second, .1)
    nn_f.balance(2 * second, .01)
    nn_f.run_sim(9 * second)

    frac = .1
    fname = 'contASS_pr' + str(pr) + 'pfboost' + str(pf_boost) + \
            'frac' + str(frac) + '.npz'
    spikes4save = calc_spikes.get_spike_times_ps(nn_f,
                                                 frac=frac, pick_first=False)
    np.savez_compressed(fname, spikes4save)

    return nn_f


def test_slopes():
    nn_f = Nets(Ne=20000, Ni=5000, cp_ee=.01, cp_ie=.01, cp_ei=0.01, cp_ii=.01,
                n_ass=1, s_ass=500, pr=.0, pf=.0,
                ext_input=200 * pA,
                g_ee=0.1 * nS, g_ie=0.1 * nS, g_ei=0.4 * nS, g_ii=0.4 * nS
                )
    nn_f.generate_ps_assemblies('gen_ordered')
    nn_f.set_net_connectivity()
    nn_f.set_spike_monitor()
    nn_f.set_rate_monitor()
    nn_f.set_voltage_monitor()
    nn_f.set_current_monitor()
    nn_f.set_conductance_monitor()

    '''
    gr0 = nn.p_ass_index[0][0]
    t = 21*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 
    t = 21.5*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 
    t = 22.*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 
    t = 22.5*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 
    t = 23.*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 
    t = 23.5*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 
    '''

    nn_f.mon_spike_e.record = False
    nn_f.mon_spike_i.record = False
    nn_f.balance(5 * second, 5.)
    nn_f.balance(5 * second, 1.)
    nn_f.balance(5 * second, .1)
    nn_f.balance(4 * second, .01)
    nn_f.mon_spike_e.record = True
    nn_f.mon_spike_i.record = True
    nn_f.balance(1 * second, .01)

    for nrn_f in nn_f.p_ass_index[0][0]:
        nn_f.Pe[nrn_f].I += 5 * pA
    nn_f.run_sim(.5 * second)
    for nrn_f in nn_f.p_ass_index[0][0]:
        nn_f.Pe[nrn_f].I -= 5 * pA
    nn_f.run_sim(.5 * second)

    for nrn_f in nn_f.p_assinh_index[0][0]:
        nn_f.Pi[nrn_f].I += 5 * pA
    nn_f.run_sim(.5 * second)
    for nrn_f in nn_f.p_assinh_index[0][0]:
        nn_f.Pi[nrn_f].I -= 5 * pA
    nn_f.run_sim(.5 * second)

    fe = calc_spikes.make_fr_from_spikes(nn_f, 0, 5, True)[0]
    fi = calc_spikes.make_fr_from_spikes(nn_f, 0, 5, False)[0]
    plt.subplot(211)
    plt.plot(fe)
    plt.subplot(212)
    plt.plot(fi)

    # plt.show()
    return nn_f


# noinspection PyPep8Naming
def test_contin(Ne=20000):
    gfc = 1
    Ni = Ne // 4
    cp = .01

    # the default conductances used for Ne=20000
    ge0 = 0.1 * nS
    gi0 = 0.4 * nS

    gee = ge0
    gii = gi0

    n_ass = 10
    s_ass = 500
    pr = .06
    pf = .06

    continuous_ass = True
    nn_f = Nets(Ne=Ne, Ni=Ni, cp_ee=cp, cp_ie=cp, cp_ei=cp, cp_ii=cp,
                n_ass=n_ass, s_ass=s_ass, pr=pr, pf=pf, ext_input=200 * pA,
                g_ee=gee, g_ie=gee, g_ei=gii, g_ii=gii, g_ff_coef=gfc,
                continuous_ass=continuous_ass)

    # nn.generate_ps_assemblies('gen_no_overlap')
    nn_f.generate_ps_assemblies('gen_ordered')
    nn_f.set_net_connectivity()
    nn_f.set_spike_monitor()
    nn_f.set_rate_monitor()
    nn_f.set_voltage_monitor()
    nn_f.set_current_monitor()

    gr_f = nn_f.p_ass_index[0][0]

    t = 20 * second
    nn_f.set_noisy_input(gr_f, t, sigma=0 * ms)
    t = 21 * second
    nn_f.set_noisy_input(gr_f, t, sigma=0 * ms)
    t = 22 * second
    nn_f.set_noisy_input(gr_f, t, sigma=0 * ms)
    t = 23 * second
    nn_f.set_noisy_input(gr_f, t, sigma=0 * ms)
    t = 24 * second
    nn_f.set_noisy_input(gr_f, t, sigma=0 * ms)

    nn_f.mon_spike_e.record = False
    nn_f.mon_spike_i.record = False

    nn_f.balance(5 * second, 5.)
    nn_f.balance(5 * second, 1.)
    nn_f.balance(5 * second, .1)
    nn_f.balance(5 * second, .01)
    nn_f.mon_spike_e.record = True
    nn_f.mon_spike_i.record = True
    nn_f.run_sim(5 * second)

    return nn_f


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    nn = Nets(Ne=20000, Ni=5000, cp_ee=.01, cp_ie=.01, cp_ei=0.01, cp_ii=.01,
              n_ass=10, s_ass=500, pr=.15, pf=.03, symmetric_sequence=True, p_rev=.03,
              g_ee=0.1 * nS, g_ie=0.1 * nS, g_ei=0.4 * nS, g_ii=0.4 * nS)

    nn.generate_ps_assemblies('gen_no_overlap')
    nn.set_net_connectivity()

    nn.set_spike_monitor()
    nn.set_rate_monitor()

    for gr_num in range(nn.n_ass):
        gr = nn.p_ass_index[0][gr_num]
        t_inp = (20.55 + gr_num * .1) * second
        nn.set_noisy_input(gr, t_inp, sigma=0 * ms)

    nn.balance(5 * second, 5.)
    nn.balance(5 * second, 1.)
    nn.balance(5 * second, .1)
    nn.balance(5 * second, .01)
    # nn.run_sim(4*second)
    nn.Pe.I -= .0 * pA

    for nrn in nn.p_ass_index[0][0]:
        nn.Pe[nrn].I += 3 * pA
    nn.run_sim(.5 * second)
    for nrn in nn.p_ass_index[0][0]:
        nn.Pe[nrn].I -= 3 * pA

    nn.Pe.I -= 9 * pA
    nn.run_sim(1. * second)
    nn.Pe.I += 9 * pA

    for nrn in nn.p_ass_index[0][9]:
        nn.Pe[nrn].I += 3 * pA
    nn.run_sim(.5 * second)
    for nrn in nn.p_ass_index[0][9]:
        nn.Pe[nrn].I -= 3 * pA


    plotter.plot_ps_raster(nn, chain_n=0, frac=.1)
    plt.xlim([20000, 22000])
