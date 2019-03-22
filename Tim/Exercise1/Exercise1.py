""" Computational Biology I: Quantitative Data Analysis
                Fruehjahrsemester 2018
                       Solution 1
"""

def sumList(l):
    """
    Call: 
     s = sumList(l)
    Input argument:
     l: list with numbers
    Output argument:
     s: sum
    Example: 
     sumList([1,2,3,4,5])
     =>
     15
    """
    s = sum(l)
    return(s)

def counts2frequencies(counts):
    """
    Call: 
     freq = counts2frequencies(counts)
    Input argument:
     counts: list with numbers
    Output argument:
     freq: list with frequencies
    Example: 
     counts2frequencies([8,2,3,10,5])
     =>
     [0.28571429, 0.07142857, 0.10714286, 0.35714286, 0.17857143]
    """
    freq = []
    for i in range(len(counts)):
        freq.append(counts[i]/sum(counts))
    return(freq)

def factorial(N):
    """
    Call: 
     f = factorial(N)
    Input argument(s):
     N: integer number
    Output argument(s):
     f = integer number
    Example: 
     factorial(12)
     =>
     479001600
    """
    if N == 0:
        return 1
    else:
        return(N * factorial(N-1))

def number_of_k_words(N,k):
    """
    Call:
       n = number_of_k_words(N,k)
    Input argument:
       N: integer number
       k: integer number
    Output argument(s):
       n: integer number
    Example:
       number_of_k_words(13,5)
       =>
       371293
    """
    n = N**k
    return(n)

def number_of_words(N):
    """
    Call:
       n = number_of_k_words(N)
    Input argument:
       N: integer number
    Output argument(s):
       n: integer number
    Example: 
       number_of_words(13)
       =>
       328114698808274
    """
    n = 0
    for i in range(N+1):
        n += number_of_k_words(N,i)
    return(n)

def number_of_k_words_no_repetition(N,k):
    """
    Call: 
       n = number_of_k_words_no_repetition(N,k)
    Input argument:
       N: integer number
       k: integer number
    Output argument(s):
       n: integer number
    Example: 
       number_of_k_words_no_repetition(13,5)
       =>
       154440
    """
    x = N
    n = 1
    while x > N-k:
        n *= x
        x -= 1
    return(n)

def number_of_permutations(k):
    """
    Call: 
       n = number_of_permutations(k)
    Input argument:
       k: integer number
    Output argument(s):
       n: integer number
    Example: 
       number_of_permutations(5)
       =>
       120
    """
    n = factorial(k)
    return(n)

def number_of_k_draws_no_repetition(N,k):
    """
    Call: 
       n = number_of_k_draws_no_repetition(N,k)
    Input argument:
       N: integer number
       k: integer number
    Output argument(s):
       n: integer number
    Example: 
       number_of_k_draws_no_repetition(13,5)
       =>
       1287
    """
    n = int((number_of_k_words_no_repetition(N,k)/number_of_permutations(k)))
    return(n)
