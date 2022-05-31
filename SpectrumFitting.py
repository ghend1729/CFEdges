import math
from numpy.lib.scimath import sqrt
import scipy
import numpy
import itertools
import random
import scipy.optimize
import matplotlib
import matplotlib.pyplot as pyplot
import pickle
import Hamiltonian

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

def allowedN2s(L, N_total, m01, m02):
    b = m02 - m01 - N_total
    c = N_total*(N_total + 2*m01 + 1)/2 - L
    NUp = int( 0.5*(-b + math.sqrt( b**2 - 4*c ) ) ) + 1
    NDown = math.ceil( 0.5*( -b - math.sqrt( b**2 - 4*c ) ) )
    return range(NDown, NUp)

def LMin(N, N2, m01, m02):
    #used to give the minimum angular momentum for fixed N and N2
    return N2**2 + N2*(m02 - m01 - N) + N*(N + 2*m01 + 1)/2

def minimumL(N, m01, m02):
    #used to find the minimum angular momentum for a given N
    if not m01 == m02:
        N1 = math.floor(N/2)
        N2 = math.ceil(N/2)
    else:
        N1 = math.ceil(N/2)
        N2 = math.floor(N/2)
    return N1*(N1 + 2*m01 + 1)/2 + N2*(N2 + 2*m02 + 1)/2

def RValue(E_RSES, E_Model, EErrors):
    global ErrorTol
    global angularMomentumTestRange
    R = 0
    LDown = 0
    E_0M = min(E_Model[0])
    E_0R = min(E_RSES[0])

    for i in angularMomentumTestRange:
        norm = sum([errorW(EErrors[i][j], ErrorTol) for j in range(len(E_Model[i - angularMomentumTestRange[0]]))])
        Ws = [errorW(EErrors[i][j], ErrorTol) for j in range(len(E_Model[i - angularMomentumTestRange[0]]))]
        x = sum([ (( abs(E_RSES[i][j] - E_0R - E_Model[i - angularMomentumTestRange[0]][j] + E_0M ) )**2)*Ws[j] for j in range(len(E_Model[i - angularMomentumTestRange[0]])) ])
        R = R + x
    return R

def errorW(dE, ETol):
    if dE == 0:
        return 0
    else:
        return 1/(2*( dE**2 + 0*0.1**2 ))

def dataExtractorOldFormat():
    global NTOT
    global NA0
    global NAToTest
    global EntanglementData
    global EntanglementErrors
    global normalizations
    global NAs
    global LMins
    global LMinsH
    global LminsPlot
    global dTheta

    roughErrorForEntanglementLevel = 0.1
    #dTheta = "0045000"

    for NA in NAs:
        EntanglementData[NA] = []
        fileNames = ["SpectraData/finalData_NTOT" + str(NTOT) + "_NA" + str(NA) for NA in NAs]

    
    #ECuttoffs = [23, 10, 23, 10, 23]
    ECuttoffs = [23, 10, 23, 10, 23]
    #ECuttoffs = [10, 23, 10, 10, 23, 10, 23]
    """
    normFile = open("SpectraData/normalization_" + str(NTOT), 'r')
    for l in normFile.readlines():
        splitLine = l.split()
        normalizations[int(splitLine[1])] = [float(splitLine[2]), float(splitLine[3])]
    normFile.close()
    """
    
    for NA in NAs:
        normalizations[NA] = [0,0]
    
    EsActual = []
    EsErros = []
    ETemp = []
    f = open(fileNames[-1], 'r')
    for l in f.readlines():
        splitLine = l.split()
        ETemp.append([ int( splitLine[2] ), float( splitLine[4] ), math.sqrt(roughErrorForEntanglementLevel**2 + normalizations[NAs[-1]][1]**2) ])
    f.close()
    ETemp = [i for i in ETemp if i[1] < ECuttoffs[-1]]

    for i in range(len(ETemp)):
        ETemp[i][1] = ETemp[i][1] + normalizations[NAs[-1]][0]

    ERef = min( [ i[1] for i in ETemp if i[0] == 0 ])
    MaxL = max([i[0] for i in ETemp]) + 1
    for i in range(len(ETemp)):
        ETemp[i][1] = ETemp[i][1] - ERef
    for L in range(MaxL):
        ReducedETemp = [i for i in ETemp if i[0] == L]
        ReducedETemp.sort(key=lambda x: x[1])
        EsActual.append([i[1] for i in ReducedETemp])
        EsErros.append([i[2] for i in ReducedETemp])
    EntanglementData[NAs[-1]] = EsActual
    EntanglementErrors[NAs[-1]] = EsErros    

    for j in range(len(fileNames)-1):
        EsActual = []
        EsErros = []
        ETemp = []
        f = open(fileNames[j], 'r')
        for l in f.readlines():
            splitLine = l.split()
            ETemp.append([ int( splitLine[2] ), float( splitLine[4] ), math.sqrt(roughErrorForEntanglementLevel**2 + normalizations[NAs[j]][1]**2) ])
        f.close()
        ETemp = [i for i in ETemp if i[1] < ECuttoffs[j]]
        MaxL = max([i[0] for i in ETemp]) + 1
        for i in range(len(ETemp)):
            ETemp[i][1] = ETemp[i][1] - ERef + normalizations[NAs[j]][0]
        for L in range(MaxL):
            ReducedETemp = [i for i in ETemp if i[0] == L]
            ReducedETemp.sort(key=lambda x: x[1])
            EsActual.append([i[1] for i in ReducedETemp])
            EsErros.append([i[2] for i in ReducedETemp])
        EntanglementData[NAs[j]] = EsActual
        EntanglementErrors[NAs[j]] = EsErros


def dataExtractionN58Test():
    global NTOT
    global NA0
    global NAToTest
    global EntanglementData
    global EntanglementErrors
    global normalizations
    global NAs
    global LMins
    global LMinsH
    global LminsPlot

    for NA in NAs:
        EntanglementData[NA] = []
        fileNames = ["N58RawDataForPaper/NA" + str(NA) + "/finalData_NA" + str(NA) for NA in NAs]
    
    ECuttoffs = [23, 10, 23, 10, 23]
    #ECuttoffs = [23, 10, 23]
    
    normFile = open("N58RawDataForPaper/normalization/normalization_" + str(NTOT), 'r')
    for l in normFile.readlines():
        splitLine = l.split()
        normalizations[int(splitLine[1])] = [float(splitLine[2]), float(splitLine[3])]
    normFile.close()

    EsActual = []
    EsErros = []
    ETemp = []
    f = open(fileNames[-1], 'r')
    for l in f.readlines():
        splitLine = l.split()
        ETemp.append([ int( splitLine[2] ), float( splitLine[4] ), float( splitLine[5] ) ])
    f.close()
    ETemp = [i for i in ETemp if i[1] < ECuttoffs[-1]]

    for i in range(len(ETemp)):
        ETemp[i][1] = ETemp[i][1] + normalizations[NAs[-1]][0]

    ERef = min( [ i[1] for i in ETemp if i[0] == 0 ])
    MaxL = max([i[0] for i in ETemp]) + 1
    for i in range(len(ETemp)):
        ETemp[i][1] = ETemp[i][1] - ERef
    for L in range(MaxL):
        ReducedETemp = [i for i in ETemp if i[0] == L]
        ReducedETemp.sort(key=lambda x: x[1])
        EsActual.append([i[1] for i in ReducedETemp])
        EsErros.append([i[2] for i in ReducedETemp])
    EntanglementData[NAs[-1]] = EsActual
    EntanglementErrors[NAs[-1]] = EsErros

    for j in range(len(fileNames)-1):
        EsActual = []
        EsErros = []
        ETemp = []
        f = open(fileNames[j], 'r')
        for l in f.readlines():
            splitLine = l.split()
            ETemp.append([ int( splitLine[2] ), float( splitLine[4] ), float( splitLine[5] ) ])
        f.close()
        ETemp = [i for i in ETemp if i[1] < ECuttoffs[j]]
        MaxL = max([i[0] for i in ETemp]) + 1
        for i in range(len(ETemp)):
            ETemp[i][1] = ETemp[i][1] - ERef + normalizations[NAs[j]][0]
        for L in range(MaxL):
            ReducedETemp = [i for i in ETemp if i[0] == L]
            ReducedETemp.sort(key=lambda x: x[1])
            EsActual.append([i[1] for i in ReducedETemp])
            EsErros.append([i[2] for i in ReducedETemp])
        EntanglementData[NAs[j]] = EsActual
        EntanglementErrors[NAs[j]] = EsErros



#hierarchy model

def allowedN2sH(N, L):
    NUp = int( 0.5*N + math.sqrt(L) ) + 1
    NDown = math.ceil( 0.5*N - math.sqrt(L) )
    return range(NDown, NUp)


def energiesLevelLLocalH(L, N, N2, h_1_0, h_0_11, h_2_11, h_22_0):
    energies = []
    basis = []
    for l in range(L + 1):
        partitions2 = generatePartitions(l)
        partitions1 = generatePartitions(L-l)
        for p1 in partitions1:
            for p2 in partitions2:
                basis.append((N2, p1, p2))
    eigenValuesOutput = Hamiltonian.eigenvalues(basis, h_1_0, h_0_11, h_2_11, h_22_0, N)
    energies = [E for E in eigenValuesOutput]
    energies.sort()
    return energies

def findEnergiesLocalH(LRange, a, b, c, d, e, w, x, y, z, g, h, f, f_i, N):
    Es = []
    for L in LRange:
        Es.append( energiesLevelLLocalH(L, a, b, c, d, e, w, x, y, z, g, h, f, f_i, N) )
    return Es

def fToOptomiseLocalH(Params):
    global fixedParams
    global EntanglementData
    global MaxL
    global angularMomentumTestRange
    global NA0
    global m01
    global m02
    global NAToTest
    global RSigma
    global LMinsH

    fullParamSet = []
    variedParamCount = 0
    for i in range(13):
        if i in fixedParams:
            fullParamSet.append(fixedParams[i])
        else:
            fullParamSet.append(Params[variedParamCount])
            variedParamCount += 1

    Rtot = 0
    for i in range(len(NAToTest)):
        LRange = [LMinsH[i] + j for j in angularMomentumTestRange]
        EModel = findEnergiesLocalH(LRange, fullParamSet[0], fullParamSet[1], fullParamSet[2], fullParamSet[3], fullParamSet[4], fullParamSet[5], fullParamSet[6], fullParamSet[7], fullParamSet[8], fullParamSet[9], fullParamSet[10], fullParamSet[11], fullParamSet[12], NAToTest[i] - NA0)
        Rtot = Rtot + RValue(EntanglementData[NAToTest[i]], EModel, EntanglementErrors[NAToTest[i]])*math.exp(-((NAToTest[i] - NA0)**2)/( 2*(RSigma**2) ))
    return Rtot


#Testing

def fittingProcedure():
    global numberOfRuns
    global fixedParams
    RsLocal = []
    ParamsLocal = []
    ErrorsLocal = []
    for k in range(numberOfRuns):
        print("Performing optimisation " + str(k + 1) + " of " + str(numberOfRuns))

        startingValuesLocal = [ random.uniform(searchDown, searchUp) for i in range(13 - len(fixedParams)) ]

        fittedValuesLocal = scipy.optimize.minimize(fToOptomiseLocalH, tuple(startingValuesLocal))
        h = fittedValuesLocal.hess_inv
        for i in range(len(h)):
            print(math.sqrt(h[i][i])/fittedValuesLocal.x[i])
        RsLocal.append(fittedValuesLocal.fun)

        ParamsLocal.append(fittedValuesLocal.x)

        ErrorsLocal.append([math.sqrt(h[i][i]) for i in range(len(h))])

    RMinLocal = min(RsLocal)
    minIndexLocal = RsLocal.index(RMinLocal)
    print("Local Model R_fit:")
    print(RMinLocal)
    print(" ")

    x = ParamsLocal[minIndexLocal]
    dEs = ErrorsLocal[minIndexLocal]
    finalParamSet = []
    variedParamCount = 0
    for i in range(13):
        if i in fixedParams:
            finalParamSet.append(fixedParams[i])
        else:
            finalParamSet.append(x[variedParamCount])
            variedParamCount += 1

    print("Local model fitted parameters:")
    print(finalParamSet)
    print(dEs)
    print("")
    return finalParamSet, dEs


RSigma = 2
ErrorTol = 0.1

m01 = -1/2
m02 = -1/2
angularMomentumTestRange = range(4)
MaxL = 5
numberOfRuns = 20
fixedParams = {}

searchUp = 7
searchDown = -7



#NA0s = [19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 47, 51]
#NA0s = [29, 31, 33, 35, 37, 39, 41, 43, 47, 51]
#NA0s = [43, 47, 51]
NA0s = [39, 51]

NTOT = 0
NA0 = 0
NAToTest = []
EntanglementData = {}
EntanglementErrors = {}
normalizations = {}
NAs = []
LMins = []
LMinsH = []
LminsPlot = []
fittedParams1 = []
fittedParams2 = []



for NA0Loop in NA0s:
    global dTheta
    #fixedParams = {3: 0, 4: 0, 5:0, 6:0}
    fixedParams = {0:0, 3: 0, 5: 0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0}
    NTOT = 2*NA0Loop
    NA0 = NA0Loop
    print("")
    print("Fitting N = " + str(NTOT))
    print("")

    NAToTest = [NA0 - 4, NA0 - 3, NA0 - 2, NA0 - 1, NA0]
    NAs = [NA0 - 4, NA0 - 3, NA0 - 2, NA0 - 1, NA0]
    LMins = [minimumL(NA - NA0, m01, m02) for NA in NAToTest]
    LMinsH = [( (N - NA0) % 2 )/4 for N in NAToTest]
    LminsPlot = [( (N - NA0) % 2 )/4 for N in NAs]




    EntanglementData = {}
    EntanglementErrors = {}
    normalizations = {}

    dataExtractorOldFormat()
    #dataExtractionN58Test()   


    print("Fitting three parameter model")
    print("")
    fittedParams1, ParamErrors1 = fittingProcedure()
    print("Three parameter model fitted")
    print("")
    
    #fixedParams.pop(3)
    #fixedParams.pop(4)
    #fixedParams.pop(5)
    #fixedParams.pop(6)
    #fixedParams.pop(7)
    #fixedParams.pop(8)
    fixedParams.pop(9)
    fixedParams.pop(10)
    #fixedParams.pop(0)
    #fixedParams.pop(3)
    
    print("Fitting five parameter model")
    print("")
    fittedParams2, ParamErrors2 = fittingProcedure()
    print("Five parameter model fitted")
    print("")
    
    print("Saving fitted parameters")
    paramFile1 = open("ScalingTestForPaper/NA0_" + str(NA0) + "_3Param", 'wb')
    pickle.dump([fittedParams1, ParamErrors1], paramFile1)
    paramFile1.close()
    
    paramFile2 = open("ScalingTestForPaper/NA0_" + str(NA0) + "_5Param", 'wb')
    pickle.dump([fittedParams2, ParamErrors2], paramFile2)
    paramFile2.close()
    print("Saving complete")
    
    
    





#Plotting

E0CollectedM1 = []
E0CollectedM2 = []
E0CollectedR = []
DeltaNs = []
E0CollectedErrors = []
a0sOdd = [-2, -1, 0, 1, 2]
chargeEnergiesOdd = []

EsModel1ToSave = {}
EsModel2ToSave = {}

for j in range(len(NAs)):
    EsActual = EntanglementData[NAs[j]]
    EsErros = EntanglementErrors[NAs[j]]
    N = NAs[j]
    LRangeH = [LminsPlot[j] + k for k in range(MaxL)]

    params1 = fittedParams1
    params2 = fittedParams2

    EModel1 = findEnergiesLocalH( LRangeH , params1[0], params1[1], params1[2], params1[3], params1[4], params1[5], params1[6], params1[7], params1[8], params1[9], params1[10], params1[11], params1[12],  N - NA0)
    EModel2 = findEnergiesLocalH( LRangeH , params2[0], params2[1], params2[2], params2[3], params2[4], params2[5], params2[6], params2[7], params2[8], params2[9], params2[10], params2[11], params2[12],  N - NA0)
    EsModel1ToSave[NAs[j]] = [E for E in EModel1]
    EsModel2ToSave[NAs[j]] = [E for E in EModel2]

    if N % 2 == 1:
        q = [ min(EsActual[4]) - min(EsActual[0]) , min(EsActual[1]) - min(EsActual[0]) , 0 , max(EsActual[1]) - min(EsActual[0]) , max(EsActual[4]) - min(EsActual[0]) ]
        chargeEnergiesOdd.append(q)

    E0CollectedM1 += EModel1[0]
    E0CollectedM2 += EModel2[0]
    E0CollectedR += EsActual[0]
    E0CollectedErrors += EsErros[0]
    DeltaNs += [N - NA0 for i in range(len( EsActual[0] ))]

    EModCollected1 = []
    EModCollected2 = []
    ENumerical = []
    EsErrosCollected = []
    LsMMax1 = []
    LsMMin1 = []
    LsMMax2 = []
    LsMMin2 = []
    LsRMax = []
    LsRMin = []
    Ls = []
    for i in range(len(EModel1)):
        ENumerical = ENumerical + EsActual[i]
        x1 = [y.real for y in EModel1[i]]
        x2 = [y.real for y in EModel2[i]]
        EModCollected1 = EModCollected1 + x1
        EModCollected2 = EModCollected2 + x2
        EsErrosCollected = EsErrosCollected + [math.sqrt(dE**2 + normalizations[NAs[j]][1]**2) for dE in EsErros[i]]
        Ls = Ls + [i for k in EsActual[i]]
        s = 0.8
        LsRMax = LsRMax + [i + s*1/6 for k in EsActual[i]]
        LsRMin = LsRMin + [i - s*1/6 for k in EsActual[i]]
        LsMMin1 = LsMMin1 + [i - s*1/2 for k in EsActual[i]]
        LsMMax1 = LsMMax1 + [i - s*1/6 for k in EsActual[i]]
        LsMMin2 = LsMMin2 + [i + s*1/6 for k in EsActual[i]]
        LsMMax2 = LsMMax2 + [i + s*1/2 for k in EsActual[i]]


    pyplot.title(r"$N_A = " + str(N) + r"$, $N_B = " + str(2*NA0-N) + r"$, $\nu = 2/3$ Bosons", fontsize=28)
    pyplot.hlines(ENumerical, LsRMin, LsRMax, label='Numerical')
    pyplot.errorbar([L for L in Ls], ENumerical, yerr=EsErrosCollected, fmt='none', ecolor='r', capsize=10)
    pyplot.hlines(EModCollected1, LsMMin1, LsMMax1, colors='c', label=r'$H_{ES}( \epsilon \neq 0 )$')
    pyplot.hlines(EModCollected2, LsMMin2, LsMMax2, colors='r', label=r'$H_{ES}( \epsilon = 0)$')
    pyplot.xlabel(r'$\Delta L_z^A$', fontsize=25)
    pyplot.ylabel(r"$\Delta \xi$", fontsize=25)
    pyplot.legend(fontsize=25, loc='upper left')
    pyplot.subplots_adjust(bottom=0.12)
    pyplot.xticks(fontsize=18)
    pyplot.yticks(fontsize=18)
    pyplot.show()

s = 0.8
DeltaNsMinR = [x - s*1/6 for x in DeltaNs]
DeltaNsMaxR = [x + s*1/6 for x in DeltaNs]
DeltaNsMin1 = [x - s*1/2 for x in DeltaNs]
DeltaNsMax1 = [x - s*1/6 for x in DeltaNs]
DeltaNsMin2 = [x + s*1/6 for x in DeltaNs]
DeltaNsMax2 = [x + s*1/2 for x in DeltaNs]

pyplot.title(r"$N = " + str(NTOT) + r'$, $\Delta L = 0$ Sector For Various $\Delta N_A$', fontsize=28)
pyplot.hlines(E0CollectedM1, DeltaNsMin1, DeltaNsMax1, colors='c', label=r'$H_{ES}( \epsilon \neq 0 )$')
pyplot.hlines(E0CollectedM2, DeltaNsMin2, DeltaNsMax2, colors='r', label=r'$H_{ES}( \epsilon = 0 )$')
pyplot.hlines(E0CollectedR, DeltaNsMinR, DeltaNsMaxR, label='Numerical')
pyplot.errorbar(DeltaNs, E0CollectedR, yerr=E0CollectedErrors, fmt='none', ecolor='r', capsize=10)
pyplot.xlabel(r'$\Delta N_A$', fontsize=25)
pyplot.ylabel(r"$\Delta \xi$", fontsize=25)
pyplot.legend(fontsize=17)
pyplot.subplots_adjust(bottom=0.12)
pyplot.xticks(fontsize=18)
pyplot.yticks(fontsize=18)
pyplot.show()
i = 0
for q in chargeEnergiesOdd:
    pyplot.plot(a0sOdd, q, label = r"$N_A = " + str(NAs[i]) + "$")
    i += 2

pyplot.legend(fontsize=15)
pyplot.title(r"$\Delta \xi_0$ of neutral $U(1)$ charge", fontsize=15)
pyplot.xlabel(r"$\Delta n $", fontsize=15)
pyplot.ylabel(r"$\Delta \xi_0$", fontsize=15)

pyplot.show()

"""
paramFile3 = open("N58TestForPaper/NumericalDataProcessed", 'wb')
pickle.dump(EntanglementData, paramFile3)
paramFile3.close()
paramFile4 = open("N58TestForPaper/NumericalDataErrors", 'wb')
pickle.dump(EntanglementErrors, paramFile4)
paramFile4.close()
paramFile5 = open("N58TestForPaper/NormalisationsInData", 'wb')
pickle.dump(normalizations, paramFile5)
paramFile5.close()
paramFile6 = open("N58TestForPaper/Model1Data3Param", 'wb')
pickle.dump(EsModel1ToSave, paramFile6)
paramFile6.close()
paramFile7 = open("N58TestForPaper/Model2Data5Param", 'wb')
pickle.dump(EsModel2ToSave, paramFile7)
paramFile7.close()
"""

