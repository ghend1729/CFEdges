import math
import scipy
import numpy
import numpy.linalg
import itertools

def norm(p):
    if p == (0,):
        return 1
    else:
        x = set(p)
        answer = 1
        for i in x:
            multiplicity = len([1 for j in p if j == i])
            answer = answer*( i**multiplicity )*math.factorial(multiplicity)
        return math.sqrt(answer)

def f(p, q, n, alpha):
    x = [( ( ( -n )**i )/( ( alpha**(2*i) )*math.factorial(i) ))*( math.factorial(q)/math.factorial(q-i) )*( math.factorial(p)/math.factorial(p-i) ) for i in range(min(p,q) + 1) ]
    return ((-1)**p)*(alpha**(p+q))*sum(x)

def splits(n):
    if n == 1:
        return set()
    else:
        orderedPairs = []
        for i in range(1, n):
           orderedPairs.append((i, n - i))
        return orderedPairs

def pairsOfElements(n):
    unorderedPairs = []
    for i in range(n):
        unorderedPairs = unorderedPairs + [(i,j) for j in range(i)]
    return unorderedPairs

def partitionNorm2(p):
    a = 1
    for i in p:
        a = a*i
    b = 1
    for i in set(p):
        b = b*math.factorial(p.count(i))
    return a*b

def combineBosons(b1, b2, p):
    x = list(p)
    x.remove(b1)
    x.remove(b2)
    x.append(b1+b2)
    x.sort(reverse=True)
    return tuple(x)

def splitBoson(b_init, b1_f, b2_f, p):
    x = list(p)
    x.remove(b_init)
    x.append(b1_f)
    x.append(b2_f)
    x.sort(reverse=True)
    return tuple(x)

def generatePartitions(L):
    """
    Takes in integer L and returns a list of its partitions.
    """
    #source: https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning
    answer = set()
    answer.add((L, ))
    for x in range(1, L):
        for y in generatePartitions(L - x):
            answer.add(tuple(sorted((x, ) + y, reverse = True)))
    return answer


def T111(p1, p2):
    basisReference = [p for p in generatePartitions(sum(p1))]
    basisReference_inv = {basisReference[i] : i for i in range(len(basisReference))}

    outVec = [0 for i in basisReference]
    #combine bosons
    for c in pairsOfElements(len(p1)):
        b1 = p1[c[0]]
        b2 = p1[c[1]]
        newParition = combineBosons(b1, b2, p1)
        newBasisIndex = basisReference_inv[newParition]
        outVec[newBasisIndex] = outVec[newBasisIndex] + 6*b1*b2
    
    #split bosons
    for x in p1:
        for c in splits(x):
            newParition = splitBoson(x, c[0], c[1], p1)
            newBasisIndex = basisReference_inv[newParition]
            outVec[newBasisIndex] = outVec[newBasisIndex] + 3*x
    
    return (outVec[basisReference_inv[p2]])*norm(p2)/norm(p1)

def T111New(p1, p2):
    if (p1 == p2) or (not (abs(len(p1) - len(p2)) == 1)):
        return 0
    else:
        occupiedMomenta = set(p1) | set(p2)
        occupancy1 = {n : sum([1 for x in p1 if x == n]) for n in occupiedMomenta}
        occupancy2 = {n : sum([1 for x in p2 if x == n]) for n in occupiedMomenta}
        diff12 = {n : occupancy1[n] - occupancy2[n] for n in occupiedMomenta}
        negDiff = {n : diff12[n] for n in occupiedMomenta if diff12[n] < 0}
        posDiff = {n : diff12[n] for n in occupiedMomenta if diff12[n] > 0}
        sumNeg = abs(sum([negDiff[n] for n in negDiff]))
        sumPos = abs(sum([posDiff[n] for n in posDiff]))
        if sumPos == 1 and sumNeg == 2:
            nsToTake = [n for n in negDiff]
            nsToAdd = [n for n in posDiff]
            if len(nsToTake) == 2:
                a = nsToAdd[0]
                b = nsToTake[0]
                c = nsToTake[1]
                return math.sqrt(a*b*c * (a + 1)*b*c)
            else:
                a = nsToAdd[0]
                b = nsToTake[0]
                return math.sqrt(a * b * b * (a + 1) *b*(b - 1))
        elif sumNeg == 1 and sumPos == 2:
            nsToTake = [n for n in negDiff]
            nsToAdd = [n for n in posDiff]
            if len(nsToAdd) == 2:
                a = nsToTake[0]
                b = nsToAdd[0]
                c = nsToAdd[1]
                return math.sqrt(a*b*c *a*(b+1)*(c+1))
            else:
                a = nsToTake[0]
                b = nsToAdd[0]
                return math.sqrt(a * b * b * a *(b+1)*(b + 2))
        else:
            return 0

def vertex(bra, ket, alpha):
    occupiedMomenta = set(bra[2] + ket[2])
    answer = 1/(norm(bra[2])*norm(ket[2]))
    for n in occupiedMomenta:
        p = len([1 for j in ket[2] if j == n])
        q = len([1 for j in bra[2] if j == n])
        answer = answer*f(p, q, n, alpha)
    return answer

def difByOne(p1, p2):
    diffMomentum = 0
    if (not abs(len(p1) - len(p2)) == 1) and (not p1 == (0,)) and (not p2 == (0,)):
        return False, 0
    else:
        occupiedMomenta = set(p1 + p2)
        occupiedMomenta = tuple([n for n in occupiedMomenta if not n == 0])
        onlyDiffByOneOrLess = []
        count = 0
        for n in occupiedMomenta:
            p = len([1 for j in p2 if j == n])
            q = len([1 for j in p1 if j == n])
            if abs(p-q) == 0:
                onlyDiffByOneOrLess.append(True)
            elif abs(p-q) == 1:
                onlyDiffByOneOrLess.append(True)
                diffMomentum = n
                count += 1
            else:
                onlyDiffByOneOrLess.append(False)
        return all(onlyDiffByOneOrLess) and (count == 1), diffMomentum

def field1(p1, p2, n):
    x = norm(p1)/norm(p2)
    if (len(p2) < len(p1)) or (p2 == (0,)):
        return x
    else:
        return field1(p2, p1, n)
        #return n*len([1 for j in p2 if j == n])*x

def field1Deriv(p1, p2, n):
    x = (n - 1) * norm(p1)/norm(p2)
    if (len(p2) < len(p1)) or (p2 == (0,)):
        return x
    else:
        return field1(p2, p1, -n)
        #return n*len([1 for j in p2 if j == n])*x

def field2(p1, p2):
    momentumDifference = sum(p2) - sum(p1)
    if momentumDifference < 0:
        return field2(p2, p1)
    else:
        occupiedMomenta = tuple(set(p1 + p2))
        occupiedMomenta = tuple([n for n in occupiedMomenta if not n == 0])
        mul1 = [len([1 for i in p1 if i == n]) for n in occupiedMomenta]
        mul2 = [len([1 for i in p2 if i == n]) for n in occupiedMomenta]
        mulDiff = [mul1[i] - mul2[i] for i in range(len(occupiedMomenta))]
        if len(p1) == len(p2) and sum([abs(m) for m in mulDiff]) == 2:
            momentumToChange1Index = mulDiff.index(1)
            momentumToChange2Index = mulDiff.index(-1)
            return 2*math.sqrt(mul1[momentumToChange1Index]*mul2[momentumToChange2Index]*occupiedMomenta[momentumToChange1Index]*occupiedMomenta[momentumToChange2Index])
        elif len(p2) - len(p1) == 2 and sum([abs(m) for m in mulDiff]) == 2:
            if -2 in mulDiff:
                momentumToChange1Index = mulDiff.index(-2)
                n = occupiedMomenta[momentumToChange1Index]
                m = mul2[momentumToChange1Index]
                return n*math.sqrt(m*(m-1))
            else:
                momentumToChangeIndexs = [i for i in range(len(mulDiff)) if mulDiff[i] == -1]
                n1 = occupiedMomenta[momentumToChangeIndexs[0]]
                n2 = occupiedMomenta[momentumToChangeIndexs[1]]
                m1 = mul2[momentumToChangeIndexs[0]]
                m2 = mul2[momentumToChangeIndexs[1]]
                return 2*math.sqrt(n1*n2*m1*m2)
        else:
            return 0
            

def matrixElement(bra, ket, h_1_0, h_0_11, h_2_11, h_22_0, N):
    diffBetween11 = difByOne(bra[1], ket[1])
    diffBetween22 = difByOne(bra[2], ket[2])
    a_02 = (1/math.sqrt(2))*(N - 2*ket[0])
    a_01 = math.sqrt(5/2)*N
    if bra == ket:
        E_0 = h_1_0*a_01 + h_22_0*( a_01**2 ) + h_0_11*( a_02**2 ) - h_2_11*a_01*( a_02**2 )
        return (h_0_11 - h_2_11*a_01)*sum(ket[2]) + h_0_11*sum([ n*(n**2 - 1) for n in ket[1]]) + E_0
    elif (bra[0] == ket[0]) and diffBetween11[0] and diffBetween22[0]:
        return 2*h_2_11*a_02*field1Deriv(bra[1], ket[1], diffBetween11[1])*field1(bra[2], ket[2], diffBetween22[0])
    elif (bra[0] == ket[0]) and diffBetween11[0]:
        return h_2_11*field1Deriv(bra[1], ket[1], diffBetween11[1])*field2(bra[2], ket[2])
    else:
        return 0

def eigenvalues(basis,h_1_0, h_0_11, h_2_11, h_22_0, N):
    Hamiltonian = [[matrixElement(bra, ket, h_1_0, h_0_11, h_2_11, h_22_0, N) for ket in basis] for bra in basis]
    return numpy.linalg.eigvals(Hamiltonian)
    


            

        
