def get_spike_times_ps(nn, n_ps=1, frac=1.):
    """
    gets the spike times of the neurons participating in PS n_ps
    ordered according to the phase sequence arrangement
    frac is the fraction of neurons from each group to be returned
    """
    sp = []
    n = 0
    for gr in nn.p_ass_index[n_ps]:
        for nrn in gr[0:frac * len(gr)]:
            for t in nn.mon_spike_e[nrn]:
                sp.append((n, t))
            n += 1

    return sp
