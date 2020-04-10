import math


def findFairTick(K, N, A):
    r = K*1.0/N
    p = int(round(math.log10(r)/2, 4))    # float safe # log100() = log10()/2
    #if p < 1:
    #    return 1
    nr = r / (100 ** p)

    best_i = 1
    best_v = math.inf
    for i in A:
        v = nr - i
        if 0 <= v < best_v:
            best_i = i
            best_v = v

    return best_i * (100 ** p)


def significantFormat(value):
    units = ['', 'K', 'M', 'B', 'T', 'Q']
    k = 1000.0
    if value <= 0:
        return '0'

    magnitude = int(math.floor(math.log(value, k)))
    return '%.3g%s' % (value / k**magnitude, units[magnitude])


def float2log10int(value):
    return int(round(10 ** value, 4))
