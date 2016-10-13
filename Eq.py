#!/usr/bin/env python
import argparse, sys, os.path 
import subprocess
from joblib import Parallel, delayed
from elementMod import *
from numpy import matrix, dot, reshape, shape, zeros
__author__ = 'Ray Group'
 
parser = argparse.ArgumentParser(description='Basis Sets project')
parser.add_argument('-e','--element', help='Input element atomic number', type=int, required=True)
parser.add_argument('-b','--basis',help='Basis set', required=False, default="6-31G")
parser.add_argument('-t','--theory',help='Level of theory', required=False, default="UHF")
parser.add_argument('-d','--delta',help='The value of delta', required=False, type=float, default=0.001)
parser.add_argument('-c','--charge',help='The charge', required=False, type=int, default=0)
parser.add_argument('-s','--initial',help='Initial value of scale', required=False, type=float, default=1.0)
parser.add_argument('-l','--limit',help='Cutoff limit', required=False, type=float, default=1.0e-6)
parser.add_argument('-p','--parWith',help='Parallel processing within gaussian input', required=False, type=int, default=1)
parser.add_argument('-j','--parFile',help='Parallel processing for multiple gaussian files', required=False, type=int, default=4)
args = parser.parse_args()

## show values ##
Z=args.element
EleName=GetElemNam(Z)
print ("Test element is {}".format(EleName))
print ("Basis set is {}".format(args.basis))
print ("Level of theory is {}".format(args.theory))
print ("The value of Delta is {}".format(args.delta))
print ("The cutoff is {}".format(args.limit))

sto=GetSTO(Z,args.basis)

## files names  ##
fileName=str(Z)+'_'+GetElemSym(Z).strip()+'_'+args.basis.strip()
GuessFile='Guess_'+fileName+'.txt'
#GradFile='Grad_'+fileName+'.txt'
#EnergyAlphaFile='EnAl_'+fileName+'.txt'
EnergyFileI='EnergyI_'+fileName
EnergyFileF='EnergyF_'+fileName

cpu=args.parWith

def EquValue2(scalevalues):
        X=scalevalues[0]
        Y=scalevalues[1]
        equ=X**2+Y**2
        return equ
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
DeltaVal=args.delta
DEnergy=999.99
CurrCutOff=args.limit
while abs(DEnergy) > abs(CurrCutOff):
    #print(str(DEnergy)+ '>'+ str(CurrCutOff))
    ### Read Guess scale values from the file ####
    if os.path.isfile(GuessFile):
        guessScale=[]
        file=open(GuessFile,'r')
        for line in file:
            guessScale.append(float(line.rstrip('\n')))
        file.close()
    else:
        guessScale=[str(args.initial)]*len(sto)
        file=open(GuessFile,'w')
        for index,sto_out in enumerate(sto):
            file.write(str(args.initial)+'\n')
        file.close()
     
    # Calculate the initial energy
    OEnergy=EquValue2(guessScale)#Get_Energy(EnergyFileI,cpu,Z,args.charge,args.theory,args.basis,guessScale)
    
    ### Generate Scale values to find Gradiant
    def Gradient(DeltaVal, guessScale):
        result = []
        for i in range(len(guessScale)):
            plus = guessScale[:]
            plus[i] = round(guessScale[i] + DeltaVal, 15)
            minus = guessScale[:]
            minus[i] = round(guessScale[i] - DeltaVal, 15)
            result.append([plus, minus])
        return(result)
    
    def Hessian(DeltaVal, GradientA):
        result = []
        sorted_gradient = []
        sorted_hessian = []
        for val in range(len(GradientA)):
            arr = GradientA[val]
            sorted_gradient.append(arr[0])
            sorted_gradient.append(arr[1])
            for i in range(len(arr[0])):
                arr_plus = arr[0][:]
                arr_plus[i] = round(arr[0][i] + DeltaVal, 15)
                arr_minus = arr[1][:]
                arr_minus[i] = round(arr[1][i] - DeltaVal, 15)
                result.append([arr_plus, arr_minus])
                sorted_hessian.append(arr_plus)
                sorted_hessian.append(arr_minus)
        return(result, sorted_gradient, sorted_hessian)
    
    GradientA = Gradient(DeltaVal, guessScale)
    HessianA, sorted_gradient, sorted_hessian = Hessian(DeltaVal, GradientA)
    
    ###  
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ### Generate INPUT FILE and Run the Job ###
    # gradiants
    
    """def EnergyPar(title,cpu,Z,charge,theory,basis,sto_out,index,EleName):      
        title2=title+'_'+EleName.strip()+'_'+args.basis.strip()+'_scale_'+str(index+1)
        GEnergy=Get_Energy(title2,cpu,Z,charge,theory,basis,sto_out)
        return [index,GEnergy]"""
        
    def EquValue(scalevalues,index):
        equ=EquValue2(scalevalues)
        return [index,equ]

    ll=Parallel(n_jobs=args.parFile)(delayed(EquValue)(sto_out,index)
        for index,sto_out in enumerate(sorted_gradient))
    
    """ll=Parallel(n_jobs=args.parFile)(delayed(EnergyPar)('Grad',cpu,Z,args.charge,args.theory,args.basis,sto_out,index,EleName)
        for index,sto_out in enumerate(sorted_gradient))"""
    EnergyGrad={} 
    EnergyGrad={t[0]:t[1] for t in ll}
    
    #Hessian
    ll=Parallel(n_jobs=args.parFile)(delayed(EquValue)(sto_out,index)
        for index,sto_out in enumerate(sorted_hessian))
        
    """ll=Parallel(n_jobs=args.parFile)(delayed(EnergyPar)('Hess',cpu,Z,args.charge,args.theory,args.basis,sto_out,index,EleName) 
        for index,sto_out in enumerate(sorted_hessian)) """
    EnergyHess={}
    EnergyHess={t[0]:t[1] for t in ll}
        
    # calculate Gradiant
    Grad=[]
    GradMatLen = len(EnergyGrad)
    for val in range(0, GradMatLen, 2):
        Grad.append(round((float(EnergyGrad[val])-float(EnergyGrad[val + 1]))/(2.0*DeltaVal), 15))
    
    # calculate Hessian
    # one dim hessian
    
    HessMatLen = len(EnergyHess)
    HESS = []
    for val in range(0, HessMatLen, 2):
        HESS.append(round((float(EnergyHess[val]) + float(EnergyHess[val + 1]) - 2*OEnergy)/((2.0*DeltaVal)**2), 15))

    HeSSiAn = zeros(len(HESS))
    for i in range(len(HESS)):
        HeSSiAn[i] = HESS[i]

    HeSSiAn = HeSSiAn.reshape(int(len(HESS)/2), 2)
    HessLen2DInv = matrix(HeSSiAn).I
    HessLen2DInv=HessLen2DInv.tolist()
    
    Corr=dot(matrix(Grad),matrix(HessLen2DInv))
    Corr=Corr.tolist()[0]
    
    guessScale=[float(i) - float(j)*0.01 for i, j in zip(guessScale, Corr)] 
    # calculate the new energy
    NEnergy=EquValue2(guessScale)#Get_Energy(EnergyFileF,cpu,Z,args.charge,args.theory,args.basis,guessScale)
    DEnergy=NEnergy-OEnergy
    print(guessScale, NEnergy, OEnergy, DEnergy)
    
    # store the new scale values 
    file=open(GuessFile,'w')
    for val in guessScale:
        file.write(str(val)+'\n')
    file.close()
