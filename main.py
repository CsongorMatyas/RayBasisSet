#!/usr/bin/env python
import argparse, sys, os.path 
import re
import subprocess
from joblib import Parallel, delayed
from elementMod import *
from numpy import matrix, dot
__author__ = 'Ray Group'
 
parser = argparse.ArgumentParser(description='Basis Sets project')
parser.add_argument('-e','--element', help='Input element atomic number', type=int, required=True)
parser.add_argument('-b','--basis',help='Basis set', required=False, default="6-31G")
parser.add_argument('-t','--theory',help='Level of theory', required=False, default="UHF")
parser.add_argument('-d','--delta',help='The value of delta', required=False, type=float, default=0.001)
parser.add_argument('-c','--charge',help='The charge', required=False, type=int, default=0)
parser.add_argument('-s','--initial',help='Initial value of scale', required=False, type=float, default=1.0)
parser.add_argument('-l','--limit',help='Cutoff limit', required=False, type=float, default=1.0e-5)
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
    OEnergy=Get_Energy(EnergyFileI,cpu,Z,args.charge,args.theory,args.basis,guessScale)
    
    ### Generate Scale values to find Gradiant
    AlphaValue=[]
    for index,sto_out in enumerate(guessScale):
        tempScale=guessScale[:]
        tempScale[index]=((tempScale[index])+DeltaVal)
        AlphaValue.append(tempScale)
    for index,sto_out in enumerate(guessScale):
        tempScale=guessScale[:]
        tempScale[index]=((tempScale[index])-DeltaVal)
        AlphaValue.append(tempScale)
    ###
    print(guessScale)
    print(AlphaValue)
    print("")
    ### Generate Scale values to find Hessian in 1 Dim
    def Hessian_Diff(DeltaVal1,DeltaVal2,guessScale):
        Value=[]
        for index1 in range(len(guessScale)):
            tempScale1=guessScale[:]
            tempScale1[index1]=((tempScale1[index1])+DeltaVal1)
            for index2 in range(index1,len(guessScale)):
                tempScale2=tempScale1[:]
                tempScale2[index2]=((tempScale2[index2])+DeltaVal2)
                Value.append(tempScale2)
        return Value
       
    FFmatrix=Hessian_Diff(DeltaVal,DeltaVal,guessScale)
    FRmatrix=Hessian_Diff(DeltaVal,-DeltaVal,guessScale)
    RFmatrix=Hessian_Diff(-DeltaVal,DeltaVal,guessScale)
    RRmatrix=Hessian_Diff(-DeltaVal,-DeltaVal,guessScale)
    FFFRRFRR=FFmatrix+FRmatrix+RFmatrix+RRmatrix
    print(DeltaVal)
    print(guessScale)
   
    print(FFmatrix)
    print(FRmatrix)
    print(RFmatrix)
    print(RRmatrix)
    print(FFFRRFRR)
    break    
    ###  
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ### Generate INPUT FILE and Run the Job ###
    # gradiants
    
    def EnergyPar(title,cpu,Z,charge,theory,basis,sto_out,index,EleName):      
        title2=title+'_'+EleName.strip()+'_'+args.basis.strip()+'_scale_'+str(index+1)
        GEnergy=Get_Energy(title2,cpu,Z,charge,theory,basis,sto_out)
        return [index,GEnergy]

    ll=Parallel(n_jobs=args.parFile)(delayed(EnergyPar)('Grad',cpu,Z,args.charge,args.theory,args.basis,sto_out,index,EleName) 
        for index,sto_out in enumerate(AlphaValue))
    EnergyGrad={} 
    EnergyGrad={t[0]:t[1] for t in ll}
    #Hessian
    ll=Parallel(n_jobs=args.parFile)(delayed(EnergyPar)('Hess',cpu,Z,args.charge,args.theory,args.basis,sto_out,index,EleName) 
        for index,sto_out in enumerate(FFFRRFRR)) 
    EnergyHess={}
    EnergyHess={t[0]:t[1] for t in ll}
        
    # calculate Gradiant
    Grad=[]
    half_len_grid=int(len(EnergyGrad)/2)
    for val in range(half_len_grid):
        Grad.append((float(EnergyGrad[val])-float(EnergyGrad[val+half_len_grid]))/(2.0*DeltaVal))
    # calculate Hessian
    # one dim hessian
    Hess=[]
    quart_len_hess=int(len(EnergyHess)/4)
    for val in range(quart_len_hess):
        Hess.append((float(EnergyHess[val])+float(EnergyHess[val+3*quart_len_hess])
            -float(EnergyHess[val+quart_len_hess])-float(EnergyHess[val+2*quart_len_hess]))/(2.0*DeltaVal)**2)
    
    HessLen1D=len(Hess)
    HessLen2D=len(guessScale)
    HessMatrix = [[0 for x in range(HessLen2D)] for y in range(HessLen2D)]
    
    M=HessLen1D-1
    K=M
    N=HessLen2D-1
    #print(M,N,K)
    for J in range(N+1):
        JX=N-J
        for I in range(JX+1):
            IX=JX-I
            HessMatrix[IX][JX]=Hess[K]
            HessMatrix[JX][IX]=Hess[K]
            K=K-1
    
    
    HessLen2DInv = matrix(HessMatrix).I
    HessLen2DInv=HessLen2DInv.tolist()
    #print(HessMatrix,HessLen2DInv)
    #print('RRRRRRRRRRR')
    Corr=dot(matrix(Grad),matrix(HessLen2DInv))
    Corr=Corr.tolist()[0]
    #print(Corr)
    
    #sys.exit(0)
    # calculate the new scale values
    #print(Grad)
    #print(guessScale)
    guessScale=[float(i) - j*0.01 for i, j in zip(guessScale, Corr)] 
    #print(guessScale)
    # calculate the new energy
    NEnergy=Get_Energy(EnergyFileF,cpu,Z,args.charge,args.theory,args.basis,guessScale)
    DEnergy=NEnergy-OEnergy
    print(guessScale, NEnergy, OEnergy, DEnergy)
    
    # store the new scale values 
    file=open(GuessFile,'w')
    for val in guessScale:
        file.write(str(val)+'\n')
    file.close()
    '''if DEnergy < 0.0:
        file=open(GuessFile,'w')
        for val in guessScale:
            file.write(str(val)+'\n')
        file.close()
    else:
        DeltaVal=DeltaVal*0.1
        print('Change dalta value to '+ str(DeltaVal))
        #sys.exit(0)
    #subprocess.call(['qsub',title+'.sh'])'''
    