# Memory replay in balanced recurrent networks

A port of the code from Chenkov et al. to current versions of Brian2 pipeline,
as a stsarting point for current investigations

Currently targeting:
  - brian2=2.2.2.1
  - numpy=1.18.1
  - python=3.6.10



### Old README
This code reproduces some results from the paper:

Memory replay in balanced recurrent networks.
N. Chenkov, H. Sprekeler, R. Kempter
[PLoS Comput. Biol., 13(1):e1005359, 2017. DOI: 10.1371/journal.pcbi.1005359](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005359)

The code managed to cold-rot, and old packages need to be imported.
One way to run the program is to create a virtual environment.
The following lines create a virtual environment in python2, activate it,
and install old versions of scipy, numpy and brian through pip:

    virtualenv2 assembly_vm
    source assembly_vm/bin/activate
    pip install scipy==0.17
    pip install numpy==1.10.4
    pip install brian=1.4.1


The main file is assemblyseq.py.
'run assemblyseq' in a python shell should illustrate
activity propagation through an assembly sequence.

