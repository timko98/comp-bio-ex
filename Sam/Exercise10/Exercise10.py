""" Computational Biology I: Quantitative Data Analysis
                Fruehjahrsemester 2019
                       Exercise 10
"""

import numpy as np
from scipy.special import gammaln as g

def readFasta(fname):
    """
    Call:
       seqs = readFasta(fname)
    Input argument:
       fname: string (fasta file name)
    Output arguments:
       seqs: dict of strings (DNA sequences)
    Example:
       seqs = readFasta('sequences1.fa')
       =>                                               
       seqs['hg19_v1_chr1_-_179545067_179545086 gc'] : 'GCTGGGTTCCTAACTTAC...GTCATCGAATTAGGGTCACT'
       seqs['hg19_v1_chr10_+_26727651_26727685 gc'] : 'TCCCCTGAGGGCGGGCTG...AGTCTCCCGGAACCTGGAGG'
       ...
       seqs['hg19_v1_chr16_+_81812925_81812936 gc'] : 'TTTATTTATTTTTGAGAC...CGCGGACGCTCGGAGCCACAC'
    """
    with open(fname) as f:
        a = f.read().strip().split("\n")
        seqs = dict(zip([i[1:] for i in a[0::2]], a[1::2]))

    return(seqs)

def countWord(seq,word):
    """
    Call:
       count = countWord(seq,word)
    Input argument:
       seq: string
       word: string
    Output arguments:
       count: integer
    Example:
       seqs = readFasta('sequences1.fa')
       countWord(seqs['hg19_v1_chr1_-_179545067_179545086 gc'],'AT')
       =>
       18
    """

    count = start = 0
    while True:
        start = seq.find(word, start) + 1  # find() returns -1 if not found
        if start > 0:
            count += 1
        else:
            break

    return(count)

def countMatrix(seq):
    """
    Call:
       counts = countMatrix(seq)
    Input argument:
       seq: string
    Output arguments:
       counts: 2-by-2 numpy float array
    Example:
       seqs = readFasta('sequences1.fa')
       countMatrix(seqs['hg19_v1_chr1_-_179545067_179545086 gc'])
       =>
       array([[ 409.,  236.],
              [ 237.,  137.]])
    """

    counts = np.zeros((2,2))
    dn = [["GG", "GC", "CG", "CC"], ["AG", "AC", "TG", "TC"], ["GA", "GT", "CA", "CT"], ["AA", "AT", "TA", "TT"]]
    for i, n in enumerate(dn):
        for c in n:
            counts[i//2, i%2] += countWord(seq, c)

    return(counts)

def independentLML(counts,lam=1.0):
    """
    Call:
       lml = independentLML(counts)
    Input argument:
       counts: 2-by-2 numpy float array
    Output arguments:
       lml: float
    Example:
       seqs = readFasta('sequences1.fa')
       counts = countMatrix(seqs['hg19_v1_chr1_-_179545067_179545086 gc'])
       independentLML(counts)
       =>
       -1342.7630822
    """
    n = np.sum(counts)
    c1, c2, c3, c4 = counts.flatten()
    logLM = g(2*lam) - g(2*n + 2*lam) + g(2*c1 + c2 + c3 + lam) + g(2*c4 + c2 + c3 + lam) -  2*g(lam)
    return(logLM)

def dependentLML(counts,lam=1.0):
    """
    Call:
       lml = dependentLML(counts)
    Input argument:
       counts: 2-by-2 numpy float array
    Output arguments:
       lml: float
    Example:
       seqs = readFasta('sequences1.fa')
       counts = countMatrix(seqs['hg19_v1_chr1_-_179545067_179545086 gc'])
       dependentLML(counts)
       =>
       -1347.9066488
    """
    n = np.sum(counts)
    logLM = g(4*lam) - g(n + 4*lam) + np.sum([g(c + lam) for c in counts.flatten()])

    return(logLM)

def dependentPosterior(counts):
    """
    Call:
       post = dependentPosterior(counts)
    Input argument:
       counts: 2-by-2 numpy float array
    Output arguments:
       post: float
    Example:
       seqs = readFasta('sequences1.fa')
       counts = countMatrix(seqs['hg19_v1_chr1_-_179545067_179545086 gc'])
       dependentPosterior(counts)
       =>
       0.00580296
    """
    bf = np.exp(dependentLML(counts) - independentLML(counts))
    post = np.exp(np.log(bf)) / (1 + np.exp(np.log(bf)))

    return(post)