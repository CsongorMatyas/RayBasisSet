#!/usr/bin/env python
import numpy as np
from scipy.optimize import minimize
import argparse, sys, os.path, subprocess
from joblib import Parallel, delayed

__author__ = "Raymond Poirier's Group - Ahmed Alravashdeh, Ibrahim Awad, Csongor Matyas"
 
parser = argparse.ArgumentParser(description='Basis Sets project')
parser.add_argument('-e','--element', help='Input element atomic number', type=int, required=True)
parser.add_argument('-b','--basis',help='Basis set', required=False, default="6-31G")####################################################BasisSet
parser.add_argument('-t','--theory',help='Level of theory', required=False, default="UHF")###############################################Method?
parser.add_argument('-d','--delta',help='The value of delta', required=False, type=float, default=0.001)
parser.add_argument('-c','--charge',help='The charge', required=False, type=int, default=0)
parser.add_argument('-s','--initial',help='Initial scale values', required=False, type=float, nargs='+')
parser.add_argument('-l','--limit',help='Cutoff limit', required=False, type=float, default=1.0e-6)
parser.add_argument('-p','--parWith',help='Parallel processing within gaussian input', required=False, type=int, default=1)##############Need better name
parser.add_argument('-j','--parFile',help='Parallel processing for multiple gaussian files', required=False, type=int, default=4) #######Need better name

args = parser.parse_args()



class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

##################################################################################################################################################
#Get functions
##################################################################################################################################################

def GetElementSymbol(Z):
    if Z < 1:
        print('Error: the atomic number is less than one (Z<1)\nProgram Exit ):')
        sys.exit(0)
    elif Z> 92:
        print('Error: the atomic number is greater than 92 (Z>92)\nProgram Exit ):')
        sys.exit(0)
    element=['H ',                                                                       'He', 
    'Li','Be',                                                  'B ','C ','N ','O ','F ','Ne', 
    'Na','Mg',                                                  'Al','Si','P ','S ','Cl','Ar', 
    'K ','Ca','Sc','Ti','V ','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr', 
    'Rb','Sr','Y ','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I ','Xe', 
    'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf', 
    'Ta','W ','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U ']
    return element[Z - 1]


def GetElementName(Z):
    if Z < 1:
        print('Error: the atomic number is less than one (Z<1)\nProgram Exit ):')
        sys.exit(0)
    elif Z > 92:
        print('Error: the atomic number is greater than 92 (Z>92)\nProgram Exit ):')
        sys.exit(0)
    element=['HYDROGEN    ','HELIUM      ','LITHIUM     ','BERYLLIUM   ', 
    'BORON       ','CARBON      ','NITROGEN    ','OXYGEN      ', 
    'FLUORINE    ','NEON        ','SODIUM      ','MAGNESIUM   ', 
    'ALUMINUM    ','SILICON     ','PHOSPHORUS  ','SULFUR      ', 
    'CHLORINE    ','ARGON       ','POTASSIUM   ','CALCIUM     ', 
    'SCANDIUM    ','TITANIUM    ','VANADIUM    ','CHROMIUM    ', 
    'MANGANESE   ','IRON        ','COBALT      ','NICKEL      ', 
    'COPPER      ','ZINC        ','GALLIUM     ','GERMANIUM   ', 
    'ARSENIC     ','SELENIUM    ','BROMINE     ','KRYPTON     ', 
    'RUBIDIUM    ','STRONTIUM   ','YTTRIUM     ','ZIRCONIUM   ', 
    'NIOBIUM     ','MOLYBDENUM  ','TECHNETIUM  ','RUTHENIUM   ', 
    'RHODIUM     ','PALLADIUM   ','SILVER      ','CADMIUM     ', 
    'INDIUM      ','TIN         ','ANTIMONY    ','TELLURIUM   ', 
    'IODINE      ','XENON       ','Cesium      ','Barium      ', 
    'Lanthanum   ','Cerium      ','Praseodymium', 
    'Neodymium   ','Promethium  ','Samarium    ','Europium    ', 
    'Gadolinium  ','Terbium     ','Dysprosium  ','Holmium     ', 
    'Erbium      ','Thulium     ','Ytterbium   ', 
    'Lutetium    ','Hafnium     ','Tantalum    ','Tungsten    ', 
    'Rhenium     ','Osmium      ','Iridium     ','Platinum    ', 
    'Gold        ','Mercury     ','Thallium    ','Lead        ', 
    'Bismuth     ','Polonium    ','Astatine    ','Radon       ', 
    'Francium    ','Radium      ','Actinium    ','Thorium     ', 
    'Protactinium','Uranium     ']
    return element[Z - 1]

def GetElementGroupPeriod(Z):
    if Z == 0:
        print('Error: the atomic number is less than one (Z<1)\nProgram Exit ):')
        sys.exit(0)
    ElementOrder = [2, 8, 8, 18, 18, 36, 36]
    Period = 1
    Group  = 0
    for i in range(len(ElementOrder)):
        if Group < Z - ElementOrder[i]:
            Group  += ElementOrder[i]
            Period += 1
    Group = Z - Group
    return(Group, Period)

   
def GetElementMultiplicity(Z, Charge):
    Group, Period = GetElementGroupPeriod(Z)
    g = Group - Charge #Everything after this should be checked, or we should find a formula that will calculate multiplicity instead
    if g in [1,3,7,17]:
        return 2
    elif g in [2,8,18]:
        return 1
    elif g in [4,6,16]:
        return 3
    elif g in [5,15]:
        return 4 #############What if g is 9, 10, 11, 12, 13, 14 or >18?

def GetElementCoreValence(Z):
    ElementsOrbitals = [['1S'],['2S','2P'],['3S','3P'],['4S','3D','4P'],['5S','4D','5P'],['6S','4F','5D','6P'],['7S','5F','6D','7P']]
    Group, Period = GetElementGroupPeriod(Z)
    
    AtomicCore=[]
    for sto in range(0, Period - 1):
        for psto in ElementsOrbitals[sto]:
            AtomicCore.append(psto)
    
    AtomicValance=[]
    for vsto in ElementsOrbitals[Period - 1]:
        AtomicValance.append(vsto)
                                                #These also have to be checked
    return(AtomicCore, AtomicValance)

def GetBasisSetCoreValence(BasisSet):
    Core = BasisSet[:BasisSet.find('-')]
    Valence = BasisSet[BasisSet.find('-') + 1 : BasisSet.find('G')]  #RE is needed here :(
    return Core, Valence

def GetSTO(Z, BasisSet):
    STO=[]
    Core, Valence = GetBasisSetCoreValence(BasisSet)
    AtomicCore, AtomicValence = GetElementCoreValence(Z)
    
    for corevalue in Core:
        for atomiccore in AtomicCore:
            STO.append('STO ' + atomiccore + ' ' + str(corevalue))
            
    for valencevalue in Valence:
        for atomicvalence in AtomicValence:
            STO.append('STO ' + atomicvalence + ' ' + str(valencevalue))
            
    return STO

##################################################################################################################################################
##### Functions related to generating the Input file
##################################################################################################################################################

def GenerateFirstLine(Method):
    FirstLine = '# ' + Method + '/gen gfprint\n'
    return FirstLine

def GenerateTitle(Z, Scaling_factors):
    all_factors = ''
    Element = GetElementName(Z).strip()

    for i in range(len(Scaling_factors)):
        all_factors = all_factors + str(Scaling_factors[i]) + '_'

    Title = "\n" + Element + "_" + all_factors + "\n\n"
    return Title

def GenerateChargeMultiplicity(Z, Charge):
    ChargeMultiplicity = "{} {}\n".format(Charge, GetElementMultiplicity(Z, Charge))
    return ChargeMultiplicity

def GenerateZMatrix(Z):
    ZMatrix = GetElementSymbol(Z).strip() + "\n\n"
    return ZMatrix

def GenerateCartesianCoordinates(Z):
    CartesianCoordinates = GetElementSymbol(Z).strip() + ' 0\n'
    return CartesianCoordinates
    
def GenerateInput(cpu, Z, Charge, Method, BasisSet, Scaling_factors):
    inputtext = '%NPROCS=' + str(cpu) + '\n' + GenerateFirstLine(Method)
    inputtext += GenerateTitle(Z, Scaling_factors)
    inputtext += GenerateChargeMultiplicity(Z, Charge)
    inputtext += GenerateZMatrix(Z)
    inputtext += GenerateCartesianCoordinates(Z)

    sto = GetSTO(Z, BasisSet)
    for index, sto_out in enumerate(sto):
        inputtext += sto_out + ' ' + str(Scaling_factors[index])+'\n'
    inputtext += '****\n\n'
    return inputtext
    
def Get_Energy(FileName, cpu, Z, Charge, Method, BasisSet, guessScale):
    file=open(FileName+'.gjf','w')
    file.write(GenerateInput(cpu, Z, Charge, Method, BasisSet, guessScale) + '\n\n')
    file.close()
    
    #subprocess.call('GAUSS_SCRDIR="/nqs/$USER"\n', shell=True)
    subprocess.call('g09 < '+ FileName + '.gjf > ' + FileName + '.out\n', shell=True)
    Energy = subprocess.check_output('grep "SCF Done:" ' + FileName + '.out | tail -1|awk \'{ print $5 }\'', shell=True)
    Energy = Energy.decode('ascii').rstrip('\n')
    if Energy != "":
         EnergyNUM=float(Energy)

    else:
         print(bcolors.FAIL,"\n STOP STOP: Gaussian is stupid, Sorry for that :(", bcolors.ENDC)
         print(bcolors.FAIL,"File Name: ", FileName, bcolors.ENDC, "\n\n GOOD LUCK NEXT TIME!!!")
         sys.exit(0)
    return EnergyNUM

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

def CreateE2Scales(Nr_of_scales, DeltaVal, guessScale):
    E2Scales = []
    for i in range(Nr_of_scales):
        iScales = np.zeros(Nr_of_scales).tolist()
        for j in range(Nr_of_scales):
            iScales[j] = guessScale[j]
        iScales[i] = iScales[i] + 2 * DeltaVal
        E2Scales.append(iScales)
    return(E2Scales)

def CreateEEScales(Nr_of_scales, DeltaVal1, DeltaVal2, guessScale, Indices):
    EEScales = []
    for (i, j) in Indices:
        ijScales = np.zeros(Nr_of_scales).tolist()
        for k in range(Nr_of_scales):
            ijScales[k] = guessScale[k]
        ijScales[i] = ijScales[i] + DeltaVal1
        ijScales[j] = ijScales[j] + DeltaVal2
        EEScales.append(ijScales)
    return(EEScales)

def EnergyPar(title,cpu,Z,charge,theory,basis,sto_out,index,ElementName):
    title2=title+'_'+ElementName.strip()+'_'+args.basis.strip()+'_scale_'+str(index+1)
    GEnergy=Get_Energy(title2,cpu,Z,charge,theory,basis,sto_out)
    return [index,GEnergy]

def Initiate():
    Z = args.element
    ElementName = GetElementName(Z)
    cpu = args.parWith
    CurrCutOff = args.limit
    DeltaVal = args.delta

print ("Test element is {}".format(ElementName))
print ("Basis set is {}".format(args.basis))
print ("Level of theory is {}".format(args.theory))
print ("The value of Delta is {}".format(args.delta))
print ("The cutoff is {}".format(args.limit))

## files names  ##
fileName = str(Z) + '_' + GetElementSymbol(Z).strip() + '_' + args.basis.strip()
GuessFile = 'Guess_' + fileName + '.txt'
EnergyFileI = 'EnergyI_' + fileName
EnergyFileF = 'EnergyF_' + fileName

sto=GetSTO(Z, args.basis)
stoLen=len(sto)

if args.initial is not None:
    if stoLen == len(args.initial):
        print("The guess values ", args.initial)
        guessScale = args.initial
    else:
        print(bcolors.FAIL,"\nSTOP STOP: number of guess values should be ", stoLen,bcolors.ENDC)
        sys.exit()
elif os.path.isfile(GuessFile):
        guessScale=[]
        File = open(GuessFile, 'r')
        for line in File:
            guessScale.append(float(line.rstrip('\n')))
        File.close()
        print("The guess values (From the File) are ", guessScale)
else:
        guessScale = [1.0] * stoLen
        print("The guess values (Default Values) are ", guessScale)

# Store the values in file
File = open(GuessFile,'w')
for val in guessScale:
    File.write(str(val) + '\n')
File.close()

Nr_of_scales = len(guessScale)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
DEnergy=9999999999.99
while abs(DEnergy) > abs(CurrCutOff):
    # Calculate the initial energy
    OEnergy=Get_Energy(EnergyFileI,cpu,Z,args.charge,args.theory,args.basis,guessScale)
    
    Indices, Trace = CreateIndices(Nr_of_scales)
    GradientA, sorted_gradient = Gradient(DeltaVal, guessScale)

    E2PScales = CreateE2Scales(Nr_of_scales, DeltaVal, guessScale)
    E2MScales = CreateE2Scales(Nr_of_scales, -DeltaVal, guessScale)
    EPPScales = CreateEEScales(Nr_of_scales, DeltaVal, DeltaVal, guessScale, Indices)
    ENPScales = CreateEEScales(Nr_of_scales, -DeltaVal, DeltaVal, guessScale, Indices)
    EPNScales = CreateEEScales(Nr_of_scales, DeltaVal, -DeltaVal, guessScale, Indices)
    ENNScales = CreateEEScales(Nr_of_scales, -DeltaVal, -DeltaVal, guessScale, Indices)

    sorted_hessian = []
    sorted_hessian.extend(E2PScales)
    sorted_hessian.extend(E2MScales)
    sorted_hessian.extend(EPPScales)
    sorted_hessian.extend(ENPScales)
    sorted_hessian.extend(EPNScales)
    sorted_hessian.extend(ENNScales)
    
    ### Generate INPUT FILE and Run the Job ###
    ll=Parallel(n_jobs=args.parFile)(delayed(EnergyPar)('Grad',cpu,Z,args.charge,args.theory,args.basis,sto_out,index,ElementName)
        for index,sto_out in enumerate(sorted_gradient))
    EnergyGrad={} 
    EnergyGrad={t[0]:round(t[1], 15) for t in ll}

    # calculate Gradiant
    Grad=[]
    GradMatLen = len(EnergyGrad)
    for val in range(0, GradMatLen, 2):
        Grad.append(round((float(EnergyGrad[val]) - float(EnergyGrad[val + 1])) / (2.0 * DeltaVal), 15))
    
    if any(val==0.0 for val in Grad):
        print(bcolors.FAIL,"\nSTOP STOP: Gradiant contains Zero values",bcolors.ENDC,"\n", Grad)
        sys.exit()
    #Hessian
    ll=Parallel(n_jobs=args.parFile)(delayed(EnergyPar)('Hess',cpu,Z,args.charge,args.theory,args.basis,sto_out,index,ElementName) 
        for index,sto_out in enumerate(sorted_hessian))
    EnergyHess={}
    EnergyHess={t[0]:round(t[1], 15) for t in ll}
        
    
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

    Hessian = np.zeros((Nr_of_scales, Nr_of_scales)).tolist()
    
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
    
    print(Grad)
    print(Hessian)
    break
    HessLen2DInv = np.matrix(Hessian).I
    HessLen2DInv=HessLen2DInv.tolist()
    
    Corr=np.dot(np.matrix(Grad),np.matrix(HessLen2DInv))
    Corr=Corr.tolist()[0]
    
    guessScale=[float(i) - float(j) for i, j in zip(guessScale, Corr)] 
    # calculate the new energy
    NEnergy=Get_Energy(EnergyFileF,cpu,Z,args.charge,args.theory,args.basis,guessScale)
    DEnergy=NEnergy-OEnergy

    #print(Hessian)
    #print("Hessian")
    #print(HessLen2DInv)
    #print("HessianInv")
    #print(Corr)
    #print("Corr")
    #print(Grad)
    #print("Grad")
    #print("")
    #break
    if DEnergy <= 0.0:
        ColoRR=bcolors.OKGREEN
    else:
        ColoRR=bcolors.FAIL

    print(ColoRR, guessScale, Grad, DEnergy, NEnergy, OEnergy, bcolors.ENDC)
    #break
    # store the new scale values 
    file=open(GuessFile,'w')
    for val in guessScale:
        file.write(str(val)+'\n')
    file.close()

