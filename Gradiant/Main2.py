import argparse, sys, elementMod, os.path 
import re
import subprocess
from joblib import Parallel, delayed
__author__ = 'Ray Group'
 
parser = argparse.ArgumentParser(description='Basis Sets project')
parser.add_argument('-e','--element', help='Input element atomic number', type=int, required=True)
parser.add_argument('-b','--basis',help='Basis set', required=False, default="6-31G")
parser.add_argument('-t','--theory',help='Level of theory', required=False, default="UHF")
parser.add_argument('-d','--delta',help='The value of delta', required=False, type=float, default=0.4)
parser.add_argument('-c','--charge',help='The charge', required=False, type=int, default=0)
parser.add_argument('-s','--initial',help='Initial value of scale', required=False, type=float, default=1.0)
parser.add_argument('-l','--limit',help='Cutoff limit', required=False, type=float, default=1.0e-5)
parser.add_argument('-p','--parWith',help='Parallel processing within gaussian input', required=False, type=int, default=1)
parser.add_argument('-j','--parFile',help='Parallel processing for multiple gaussian files', required=False, type=int, default=4)
args = parser.parse_args()

## show values ##
Z=args.element
EleName=elementMod.GetElemNam(Z)
print ("Test element is {}".format(EleName))
print ("Basis set is {}".format(args.basis))
print ("Level of theory is {}".format(args.theory))
print ("The value of Delta is {}".format(args.delta))
print ("The cutoff is {}".format(args.limit))

sto=elementMod.GetSTO(Z,args.basis)

## files names  ##
fileName=str(Z)+'_'+elementMod.GetElemSym(Z).strip()+'_'+args.basis.strip()
GuessFile='Guess_'+fileName+'.txt'
GradFile='Grad_'+fileName+'.txt'
EnergyAlphaFile='EnAl_'+fileName+'.txt'
EnergyFileI='EnergyI_'+fileName
EnergyFileF='EnergyF_'+fileName

cpu=args.parWith

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
DeltaVal=args.delta
DEnergy=999.99
CurrCutOff=-abs(args.limit)
while DEnergy < CurrCutOff or DEnergy >=0.0:
    print(str(DEnergy)+ '>'+ str(CurrCutOff))
    ### Read Guess scale values from the file ####
    if os.path.isfile(GuessFile):
        guessScale=[]
        file=open(GuessFile,'r')
        for line in file:
            guessScale.append(line.rstrip('\n'))
        file.close()
    else:
        guessScale=[str(args.initial)]*len(sto)
        file=open(GuessFile,'w')
        for index,sto_out in enumerate(sto):
            file.write(str(args.initial)+'\n')
        file.close()
     
    # calculate the energy
    OEnergy=elementMod.Get_Energy(EnergyFileI,cpu,Z,args.charge,args.theory,args.basis,guessScale)

    ### Generate Scale values to find Gradiant
    AlphaValue=[]
    for index,sto_out in enumerate(guessScale):
        tempScale=guessScale[:]
        tempScale[index]=str(float(tempScale[index])+DeltaVal)
        AlphaValue.append(tempScale)
    for index,sto_out in enumerate(guessScale):
        tempScale=guessScale[:]
        tempScale[index]=str(float(tempScale[index])-DeltaVal)
        AlphaValue.append(tempScale)
    ### 
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ### Generate INPUT FILE and Run the Job ###
    title=EleName.strip()+'_'+args.basis.strip()
    EnergyGrad={}   
    def EnergyGr(title,cpu,Z,charge,theory,basis,sto_out,index,EleName):      
        title2=title+'_scale_'+str(index+1)
        GEnergy=elementMod.Get_Energy(title2,cpu,Z,charge,theory,basis,sto_out)
        #print(GEnergy,index)
        return [index,GEnergy]

    ll=Parallel(n_jobs=args.parFile)(delayed(EnergyGr)(title,cpu,Z,args.charge,args.theory,args.basis,sto_out,index,EleName) 
        for index,sto_out in enumerate(AlphaValue)) 
    EnergyGrad={t[0]:t[1] for t in ll}
    #print(EnergyGrad)
    
    # calculate Gradiant
    Grad=[]
    half_len_grid=int(len(EnergyGrad)/2)
    for val in range(half_len_grid):
        Grad.append((float(EnergyGrad[val])-float(EnergyGrad[val+half_len_grid]))/2.0*DeltaVal)
    
    # calculate the new scale values
    guessScale=[float(i) - j for i, j in zip(guessScale, Grad)] 
    
    # calculate the new energy
    NEnergy=elementMod.Get_Energy(EnergyFileF,cpu,Z,args.charge,args.theory,args.basis,guessScale)
    DEnergy=NEnergy-OEnergy
    print(guessScale, NEnergy, OEnergy, DEnergy)
    
    # store the new scale values 
    if DEnergy < 0.0:
        file=open(GuessFile,'w')
        for val in guessScale:
            file.write(str(val)+'\n')
        file.close()
    else:
        DeltaVal=DeltaVal*0.5
        print('Change dalta value to '+ str(DeltaVal))
        #sys.exit(0)
    #subprocess.call(['qsub',title+'.sh'])'''
    
