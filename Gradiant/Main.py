import argparse, sys, elementMod, os.path 
import re
import subprocess
__author__ = 'Ray Group'
 
parser = argparse.ArgumentParser(description='Basis Sets project')
parser.add_argument('-e','--element', help='Input element atomic number', type=int, required=True)
parser.add_argument('-b','--basis',help='Basis set', required=False, default="6-31G")
parser.add_argument('-t','--theory',help='Level of theory', required=False, default="UHF")
parser.add_argument('-d','--delta',help='The value of delta', required=False, type=float, default=0.01)
parser.add_argument('-c','--charge',help='The charge', required=False, type=int, default=0)
parser.add_argument('-s','--initial',help='Initial value of scale', required=False, type=float, default=1.0)
parser.add_argument('-l','--limit',help='Cutoff limit', required=False, type=float, default=1.0e-6)
args = parser.parse_args()

## show values ##
Z=args.element
EleName=elementMod.GetElemNam(Z)
print ("Test element is {}".format(EleName))
print ("Basis set is {}".format(args.basis))
print ("Level of theory is {}".format(args.theory))
print ("The value of Delta is {}".format(args.delta))


sto=elementMod.GetSTO(Z,args.basis)
## Guess file name  ##
GuessFile='Guess_'+str(Z)+elementMod.GetElemSym(Z).strip()+args.basis.strip()+'.txt'
GradFile='grad_'+str(Z)+elementMod.GetElemSym(Z).strip()+args.basis.strip()+'.txt'

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
OEnergy=elementMod.Get_Energy('energy',Z,args.charge,args.theory,args.basis,guessScale)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
### Generate Scale values to find Gradiant
AlphaValue=[]
for index,sto_out in enumerate(guessScale):
    tempScale=guessScale[:]
    tempScale[index]=str(float(tempScale[index])+args.delta)
    AlphaValue.append(tempScale)
for index,sto_out in enumerate(guessScale):
    tempScale=guessScale[:]
    tempScale[index]=str(float(tempScale[index])-args.delta)
    AlphaValue.append(tempScale)
### 
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
### Generate OUTPUT FILE ###
job=[]
title=EleName.strip()+'_'+args.basis.strip()
for index,sto_out in enumerate(AlphaValue):
    title2=title+'_scale_'+str(index+1)
    job.append(title2)
    file=open(title2+'.gfj','w')
    file.write(elementMod.returnInput(Z,args.charge,args.theory,args.basis,sto_out)+'\n\n')
    file.close()

# run the bash file
EnergyGrad=[]
for ijob in job:
    subprocess.call('g09 < '+ijob+'.gfj > '+ijob+'.out\n',shell=True)
    Genergy=subprocess.check_output('grep "SCF Done:" '+ijob+'.out | tail -1|awk \'{ print $5 }\'', shell=True)
    EnergyGrad.append(Genergy.decode('ascii').rstrip('\n'))

# calculate Gradiant
Grad=[]
half_len_grid=int(len(EnergyGrad)/2)
for val in range(half_len_grid):
    Grad.append((float(EnergyGrad[val])-float(EnergyGrad[val+half_len_grid]))/2.0*args.delta)

# calculate the new scale values
guessScale=[float(i) - j for i, j in zip(guessScale, Grad)] 

# calculate the new energy
NEnergy=elementMod.Get_Energy('energy',Z,args.charge,args.theory,args.basis,guessScale)
print(guessScale, NEnergy, OEnergy, NEnergy-OEnergy)

# store the new scale values 
file=open(GuessFile,'w')
for val in guessScale:
	file.write(str(val)+'\n')
file.close()

#subprocess.call(['qsub',title+'.sh'])
