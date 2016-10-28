#!/usr/bin/env python
import argparse, sys, os.path 
import subprocess
from joblib import Parallel, delayed
from elementMod import *
from numpy import matrix, dot, reshape, shape, zeros, linalg, diag, sqrt, transpose
__author__ = 'Ray Group'
 
parser = argparse.ArgumentParser(description='Basis Sets project')
parser.add_argument('-e','--element', help='Input element atomic number', type=int, required=True)
parser.add_argument('-b','--basis',help='Basis set', required=False, default="6-31G")
parser.add_argument('-t','--theory',help='Level of theory', required=False, default="UHF")
parser.add_argument('-d','--delta',help='The value of delta', required=False, type=float, default=0.001)
parser.add_argument('-c','--charge',help='The charge', required=False, type=int, default=0)
parser.add_argument('-s','--initial',help='Initial scale values', required=False, type=float, nargs='+')
parser.add_argument('-l','--limit',help='Cutoff limit', required=False, type=float, default=-1.0e-6)
parser.add_argument('-p','--parWith',help='Parallel processing within gaussian input', required=False, type=int, default=1)
parser.add_argument('-j','--parFile',help='Parallel processing for multiple gaussian files', required=False, type=int, default=4)
parser.add_argument('-g','--gamma',help='Base gamma coefficient value', required=False, type=float, default=0.1)

args = parser.parse_args()


## show values ##
Z=args.element
EleName=GetElemNam(Z)
cpu=args.parWith
CurrCutOff=args.limit
DeltaVal=args.delta

print ("Test element is {}".format(EleName))
print ("Basis set is {}".format(args.basis))
print ("Level of theory is {}".format(args.theory))
print ("The value of Delta is {}".format(args.delta))
print ("The cutoff is {}".format(args.limit))

## files names  ##
fileName=str(Z)+'_'+GetElemSym(Z).strip()+'_'+args.basis.strip()
GuessFile='Guess_'+fileName+'.txt'
EnergyFileI='EnergyI_'+fileName
EnergyFileF='EnergyF_'+fileName

sto=GetSTO(Z,args.basis)
stoLen=len(sto)

if args.initial is not None:
    if stoLen == len(args.initial):
        print("The guess values ", args.initial)
        guessScale=args.initial
    else:
        print(bcolors.FAIL,"\nSTOP STOP: number of guess values should be ", stoLen,bcolors.ENDC)
        sys.exit()
elif os.path.isfile(GuessFile):
        guessScale=[]
        file=open(GuessFile,'r')
        for line in file:
            guessScale.append(float(line.rstrip('\n')))
        file.close()
        print("The guess values (From the File) are ", guessScale)
else:
        guessScale=[1.0]*stoLen
        print("The guess values (Default Values) are ", guessScale)

# Store the values in file
file=open(GuessFile,'w')
for val in guessScale:
    file.write(str(val)+'\n')
file.close()

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
DEnergy=9999999999.99
counter=0
Tau = 0.375
while True: #counter < 2: #DEnergy > CurrCutOff:
    counter+=1
    #Calculate the initial energy
    OEnergy=Get_Energy(EnergyFileI,cpu,Z,args.charge,args.theory,args.basis,guessScale)
    print("OEnergy", OEnergy)    
    ### Generate Scale values to find Gradiant
    def Gradient(DeltaVal, guessScale):
        result = []
        sorted_gradient = []
        for i in range(len(guessScale)):
            plus = guessScale[:]
            plus[i] = round(guessScale[i] + DeltaVal, 15)
            minus = guessScale[:]
            minus[i] = round(guessScale[i] - DeltaVal, 15)
            result.append([plus, minus])
            sorted_gradient.append(plus)
            sorted_gradient.append(minus)
        return(result, sorted_gradient)
    
    Nr_of_scales = len(guessScale)

    def CreateIndices(Nr_of_scales):
        Indices = []
        Trace = []
        for i in range(Nr_of_scales):
            for j in range(Nr_of_scales):
                if j < i:
                   continue
                elif j == i:
                   Trace.append([i, j])
                else:
                   Indices.append([i, j])
        return(Indices, Trace)
    
    Indices, Trace = CreateIndices(Nr_of_scales)
    
    def CreateE2Scales(Nr_of_scales, DeltaVal, guessScale):
        E2Scales = []
        for i in range(Nr_of_scales):
            iScales = zeros(Nr_of_scales).tolist()
            for j in range(Nr_of_scales):
                iScales[j] = guessScale[j]
            iScales[i] = iScales[i] + 2 * DeltaVal
            E2Scales.append(iScales)
        return(E2Scales)

    def CreateEEScales(Nr_of_scales, DeltaVal1, DeltaVal2, guessScale, Indices):
        EEScales = []
        for (i, j) in Indices:
            ijScales = zeros(Nr_of_scales).tolist()
            for k in range(Nr_of_scales):
                ijScales[k] = guessScale[k]
            ijScales[i] = ijScales[i] + DeltaVal1
            ijScales[j] = ijScales[j] + DeltaVal2
            EEScales.append(ijScales)
        return(EEScales)

    GradientA, sorted_gradient = Gradient(DeltaVal, guessScale)

    E2PScales = CreateE2Scales(Nr_of_scales, DeltaVal, guessScale)
    E2MScales = CreateE2Scales(Nr_of_scales, -DeltaVal, guessScale)
    EPPScales = CreateEEScales(Nr_of_scales, DeltaVal, DeltaVal, guessScale, Indices)
    ENPScales = CreateEEScales(Nr_of_scales, -DeltaVal, DeltaVal, guessScale, Indices)
    EPNScales = CreateEEScales(Nr_of_scales, DeltaVal, -DeltaVal, guessScale, Indices)
    ENNScales = CreateEEScales(Nr_of_scales, -DeltaVal, -DeltaVal, guessScale, Indices)
    
    """
    print(E2PScales)
    print(E2MScales)
    print(EPPScales)
    print(ENPScales)
    print(EPNScales)
    print(ENNScales)
    """

    sorted_hessian = []
    sorted_hessian.extend(E2PScales)
    sorted_hessian.extend(E2MScales)
    sorted_hessian.extend(EPPScales)
    sorted_hessian.extend(ENPScales)
    sorted_hessian.extend(EPNScales)
    sorted_hessian.extend(ENNScales)
    
    ### Generate INPUT FILE and Run the Job ###
    # gradiants
    
    def EnergyPar(title,cpu,Z,charge,theory,basis,sto_out,index,EleName):      
        title2=title+'_'+EleName.strip()+'_'+args.basis.strip()+'_scale_'+str(index+1)
        GEnergy=Get_Energy(title2,cpu,Z,charge,theory,basis,sto_out)
        return [index,GEnergy]
        

    ll=Parallel(n_jobs=args.parFile)(delayed(EnergyPar)('Grad',cpu,Z,args.charge,args.theory,args.basis,sto_out,index,EleName)
        for index,sto_out in enumerate(sorted_gradient))
    EnergyGrad={} 
    EnergyGrad={t[0]:round(t[1], 15) for t in ll}
    #print("Gradiant E", EnergyGrad) 
    
    # calculate Gradiant
    Grad=[]
    GradMatLen = len(EnergyGrad)
    for val in range(0, GradMatLen, 2):
        Grad.append(round((float(EnergyGrad[val])-float(EnergyGrad[val + 1]))/(2.0*DeltaVal), 15))
    
    if sqrt(sum(map(lambda x:x*x,Grad))) < 1.0e-8:
        print(bcolors.FAIL,"\nSTOP STOP:Sum of Gradiant`s squares is Zero ",bcolors.ENDC,"\n", Grad)
        sys.exit()
    #Hessian
    ll=Parallel(n_jobs=args.parFile)(delayed(EnergyPar)('Hess',cpu,Z,args.charge,args.theory,args.basis,sto_out,index,EleName) 
        for index,sto_out in enumerate(sorted_hessian))
    EnergyHess={}
    EnergyHess={t[0]:round(t[1], 15) for t in ll}
    #print("Hess E", EnergyHess) 
        
    
    # calculate Hessian
    # one dim hessian
    
    HessianEnergies = []

    for i in range(len(sorted_hessian)):
        HessianEnergies.append(EnergyHess[i])
 
    HessianE2P = HessianEnergies[ : Nr_of_scales]
    HessianE2N = HessianEnergies[Nr_of_scales : 2 * Nr_of_scales]
    HessianEPP = HessianEnergies[2 * Nr_of_scales : 2 * Nr_of_scales + len(EPPScales)]
    HessianENP = HessianEnergies[2 * Nr_of_scales + len(EPPScales) : 2 * Nr_of_scales + 2 * len(EPPScales)]
    HessianEPN = HessianEnergies[2 * Nr_of_scales + 2 * len(EPPScales) : 2 * Nr_of_scales + 3 * len(EPPScales)]
    HessianENN = HessianEnergies[2 * Nr_of_scales + 3 * len(EPPScales) : ]

    HessianTrace = []

    for i in range(Nr_of_scales):
        HessianTrace.append((HessianE2P[i] + HessianE2N[i] - 2*OEnergy) /((2.0*DeltaVal)**2))

    HessianUpT = []

    for i in range(len(HessianEPP)):
        HessianUpT.append((HessianEPP[i] - HessianENP[i] - HessianEPN[i] + HessianENN[i]) / ((2.0*DeltaVal)**2))

    Hessian = zeros((Nr_of_scales, Nr_of_scales)).tolist()
    
    for i in range(Nr_of_scales):
        for j in range(Nr_of_scales):
            if i == j:
                Hessian[i][i] = HessianTrace[i]
                continue
            elif i < j:
                Hessian[i][j] = HessianUpT[i * (Nr_of_scales - i - 1) + j - 1]
                continue
            elif i > j:
                Hessian[i][j] = HessianUpT[j * (Nr_of_scales - j - 1) + i - 1]
                continue
            else:
                print("Wrong value!")
    
    #print("gradient:-", Grad)
    #print("hessian:-", Hessian)
    #break

    npHessian = zeros((len(Hessian), len(Hessian)))
    npHessian = matrix(npHessian)

    for i in range(len(Hessian)):
        for j in range(len(Hessian)):
            npHessian[i, j] = Hessian[i][j]
    eW, eV = linalg.eig(npHessian)
    ew = min(eW)
    
    print("Hessian")
    print(Hessian)
    print("npHessian")
    print(npHessian)
    print(len(npHessian))

    print("First eW")
    print(eW)
    print("First ew")
    print(ew)

    loop = 1

    while loop > 0:
        if ew > 0.0:
            loop = -1
        else:
            Lambda = ew - Tau #0.1 #+ (ew / 10.0)
            LHessian = npHessian.copy()
            print("Hessian")
            print(npHessian)
            for i in range(len(npHessian)):
                LHessian[i, i] = npHessian[i, i] - Lambda
            print("LHessian")
            print(LHessian)
            npHessian = LHessian.copy()
            eW, eV = linalg.eig(npHessian)
            ew = min(eW)
            print("Lambda")
            print(Lambda)
            print("Smallest eigenvalue")
            print(ew)
            print("Eigenvalues")
            print(eW)
            print("")

    HessLen2DInv = npHessian.I
    npGrad = transpose(matrix(Grad))
    npCorr=dot(HessLen2DInv, npGrad)
    Corr=transpose(npCorr).tolist()[0]

    guessScale=[float(i) - float(j) for i, j in zip(guessScale, Corr)] 
    
    print("Hessian")
    print(Hessian)
    print("HessianInv")
    print(HessLen2DInv)
    print("Corr")
    print(Corr)
    print("Grad")
    print(Grad)
    print("New guess scale")
    print(guessScale)
    print("")


    # calculate the new energy
    NEnergy=Get_Energy(EnergyFileF,cpu,Z,args.charge,args.theory,args.basis,guessScale)
    DEnergy=NEnergy-OEnergy

    print("npGrad")
    print(npGrad)
    npGradT = transpose(npGrad)
    print("npGradT")
    print(npGradT)
    print("npCorr")
    print(npCorr)
    dGT_C = dot(npGradT, npCorr)
    print("First dot")
    print(dGT_C)
    dGT_C = dGT_C.tolist()[0][0]
    print("To list")
    print(dGT_C)
 
    dCT_H_C = dot(dot(transpose(npCorr), npHessian), npCorr)
    print("dCT_H_C")
    print(dCT_H_C)
    dCT_H_C = dCT_H_C.tolist()[0][0]
    print(dCT_H_C)

    Ro = DEnergy / (dGT_C + 0.5 * dCT_H_C)
    print("Ro")
    print(Ro)
    print(type(Ro))
    print("DEnergy")
    print(DEnergy)
    
    normCorr = linalg.norm(npCorr)
    

    if Ro > 0.75 and Tau < (normCorr * 5.0 / 4.0):
        print("Tau is doubled")
        Tau = 2.0 * Tau
    elif Ro < 0.25:
        print("Tau = 1/4 |dX|")
        Tau = (1.0 / 4.0) * normCorr
    else:
        print("Tau is not changed")


    if DEnergy <= 0.0:
        ColoRR=bcolors.OKGREEN
    else:
        ColoRR=bcolors.FAIL
        

    print(ColoRR, guessScale, DEnergy, NEnergy, OEnergy, bcolors.ENDC)
    #break
    # store the new scale values 
    file=open(GuessFile,'w')
    for val in guessScale:
        file.write(str(val)+'\n')
    file.close()



    """
    Cond1 = eW[0]
    Cond2 = eW[1]

    newguessScale = guessScale

    if Cond1 > 0:
        if Cond2 > 0:
            newguessScale[0] = guessScale[0] - Corr[0]
            newguessScale[1] = guessScale[1] - Corr[1]
        else:
            newguessScale[1] = guessScale[1] - (Corr[1] * args.gamma)
    else:
        if Cond2 > 0:
            newguessScale[0] = guessScale[0] - (Corr[0] * args.gamma)
        else:
            newguessScale[0] = guessScale[0] - (Corr[0] * args.gamma)
            newguessScale[1] = guessScale[1] - (Corr[1] * args.gamma)
    print(guessScale)
    guessScale = newguessScale
    print(guessScale)
    """
