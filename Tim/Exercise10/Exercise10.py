""" Computational Biology I: Quantitative Data Analysis
                Fruehjahrsemester 2019
                       Exercise 10
"""

import numpy as np
from scipy.special import gammaln


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
    seqs = dict()
    with open(fname) as f:
        while True:
            line_1 = f.readline()
            line_2 = f.readline()
            if not line_1 or not line_2:
                break
            else:
                seqs[line_1[1:-1]] = line_2
    return (seqs)


def countWord(seq, word):
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
    count = seq.count(word)
    return (count)


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
    counts = np.zeros(shape=(2, 2))
    for i in range(len(seq)):
        if seq[i] == 'G' or seq[i] == 'C':
            if seq[i + 1] == 'G' or seq[i + 1] == 'C':
                counts[0, 0] += 1
            if seq[i + 1] == 'T' or seq[i + 1] == 'A':
                counts[1, 0] += 1
        if seq[i] == 'A' or seq[i] == 'T':
            if seq[i + 1] == 'G' or seq[i + 1] == 'C':
                counts[0, 1] += 1
            if seq[i + 1] == 'T' or seq[i + 1] == 'A':
                counts[1, 1] += 1
    return (counts)


def independentLML(counts, lam=1.0):
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
    gamma = lambda x: np.math.factorial(x - 1)
    counts = counts.reshape(-1, 1)
    n = np.sum(counts)
    logLM = gammaln(2 * lam) \
            - gammaln(2 * n + 2 * lam) \
            + gammaln(2 * counts[0] + counts[1] + counts[2] + lam) \
            - gammaln(lam) \
            + gammaln(2 * counts[3] + counts[1] + counts[2] + lam) \
            - gammaln(lam)
    return (logLM[0])


def dependentLML(counts, lam=1.0):
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
    counts = counts.reshape(-1, 1)
    n = np.sum(counts)
    logLM = gammaln(4 * lam) - gammaln(n + 4 * lam) \
            + gammaln(counts[0] + lam) - gammaln(lam) \
            + gammaln(counts[1] + lam) - gammaln(lam) \
            + gammaln(counts[2] + lam) - gammaln(lam) \
            + gammaln(counts[3] + lam) - gammaln(lam)
    return (logLM[0])


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
    return (post)
