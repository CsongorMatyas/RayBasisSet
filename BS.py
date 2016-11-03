#!/usr/bin/env python
import numpy as np
import argparse, sys, os, subprocess, joblib
#from scipy.optimize import minimize

__author__ = "Raymond Poirier's Group - Ahmed Alravashdeh, Ibrahim Awad, Csongor Matyas"

def Arguments():
    parser = argparse.ArgumentParser(description='Basis Sets project')
    parser.add_argument('-e','--Element',     required=True, type=int,  help='Input element atomic number')
    parser.add_argument('-c','--Charge',      required=False,type=int,  help='The charge',                       default=0)
    parser.add_argument('-m','--Method',      required=False,type=str,  help='Level of theory',                  default="UHF")
    parser.add_argument('-b','--BasisSet',    required=False,type=str,  help='Basis set',                        default="6-31G")
    parser.add_argument('-P','--GaussianProc',required=False,type=int,  help='Number of processors for Gaussian',default=2)
    parser.add_argument('-p','--ParallelProc',required=False,type=int,  help='Total number of processors used',  default=4)
    parser.add_argument('-s','--Scales',      required=False,type=float,help='Initial scale values',             nargs='+')
    parser.add_argument('-D','--Delta',       required=False,type=float,help='The value of Delta',               default=0.001)
    parser.add_argument('-l','--Limit',       required=False,type=float,help='Error limit',                      default=1.0e-6)
    parser.add_argument('-G','--Gamma',       required=False,type=float,help='Base gamma coefficient value',     default=0.1)

    arguments = parser.parse_args()
    return(arguments)

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
    Element=['H ',                                                                       'He', 
    'Li','Be',                                                  'B ','C ','N ','O ','F ','Ne', 
    'Na','Mg',                                                  'Al','Si','P ','S ','Cl','Ar', 
    'K ','Ca','Sc','Ti','V ','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr', 
    'Rb','Sr','Y ','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I ','Xe', 
    'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf', 
    'Ta','W ','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U ']
    return Element[Z - 1]


def GetElementName(Z):
    if Z < 1:
        print('Error: the atomic number is less than one (Z<1)\nProgram Exit ):')
        sys.exit(0)
    elif Z > 92:
        print('Error: the atomic number is greater than 92 (Z>92)\nProgram Exit ):')
        sys.exit(0)
    Element=['HYDROGEN    ','HELIUM      ','LITHIUM     ','BERYLLIUM   ', 
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
    return Element[Z - 1]

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
    
def GenerateInput(CPU, Z, Charge, Method, BasisSet, Scaling_factors):
    inputtext = '%NPROCS=' + str(CPU) + '\n' + GenerateFirstLine(Method)
    inputtext += GenerateTitle(Z, Scaling_factors)
    inputtext += GenerateChargeMultiplicity(Z, Charge)
    inputtext += GenerateZMatrix(Z)
    inputtext += GenerateCartesianCoordinates(Z)

    sto = GetSTO(Z, BasisSet)
    for index, sto_out in enumerate(sto):
        inputtext += sto_out + ' ' + str(Scaling_factors[index])+'\n'
    inputtext += '****\n\n'
    return inputtext

############################################################################################################################
#Functions related to energy and gradient / hessian
############################################################################################################################

def Get_Energy(FileName, CPU, Z, Charge, Method, BasisSet, guessScale):
    file=open(FileName+'.gjf','w')
    file.write(GenerateInput(CPU, Z, Charge, Method, BasisSet, guessScale) + '\n\n')
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

def GetGradientScales(Delta, guessScale):
    Gradient_scales = []
    Sorted_Gradient_scales = []
    for i in range(len(guessScale)):
        plus = guessScale[:]
        plus[i] = round(guessScale[i] + Delta, 15)
        minus = guessScale[:]
        minus[i] = round(guessScale[i] - Delta, 15)
        Gradient_scales.append([plus, minus])
        Sorted_Gradient_scales.append(plus)
        Sorted_Gradient_scales.append(minus)
    return(Gradient_scales, Sorted_Gradient_scales)

def CreateIndices(Nr_of_scales):
    Indices = []
    Diagonal = []
    for i in range(Nr_of_scales):
        for j in range(Nr_of_scales):
            if j < i:
               continue
            elif j == i:
               Diagonal.append([i, j])
            else:
               Indices.append([i, j])
    return(Indices, Diagonal)

def CreateE2Scales(Nr_of_scales, Delta, guessScale):
    E2Scales = []
    for i in range(Nr_of_scales):
        iScales = np.zeros(Nr_of_scales).tolist()
        for j in range(Nr_of_scales):
            iScales[j] = guessScale[j]
        iScales[i] = iScales[i] + 2 * Delta
        E2Scales.append(iScales)
    return(E2Scales)

def CreateEEScales(Nr_of_scales, Delta1, Delta2, guessScale, Indices):
    EEScales = []
    for (i, j) in Indices:
        ijScales = np.zeros(Nr_of_scales).tolist()
        for k in range(Nr_of_scales):
            ijScales[k] = guessScale[k]
        ijScales[i] = ijScales[i] + Delta1
        ijScales[j] = ijScales[j] + Delta2
        EEScales.append(ijScales)
    return(EEScales)

def GetHessianScales(Nr_of_scales, Delta, guessScale, Indices):
    E2PScales = CreateE2Scales(Nr_of_scales, Delta, guessScale)
    E2MScales = CreateE2Scales(Nr_of_scales, -Delta, guessScale)
    EPPScales = CreateEEScales(Nr_of_scales, Delta, Delta, guessScale, Indices)
    ENPScales = CreateEEScales(Nr_of_scales, -Delta, Delta, guessScale, Indices)
    EPNScales = CreateEEScales(Nr_of_scales, Delta, -Delta, guessScale, Indices)
    ENNScales = CreateEEScales(Nr_of_scales, -Delta, -Delta, guessScale, Indices)

    Sorted_Hessian_scales = []
    Sorted_Hessian_scales.extend(E2PScales)
    Sorted_Hessian_scales.extend(E2MScales)
    Sorted_Hessian_scales.extend(EPPScales)
    Sorted_Hessian_scales.extend(ENPScales)
    Sorted_Hessian_scales.extend(EPNScales)
    Sorted_Hessian_scales.extend(ENNScales)
    return(E2PScales, E2MScales, EPPScales, ENPScales, EPNScales, ENNScales, Sorted_Hessian_scales)

def EnergyParallel(Title, CPU, Z, Charge, Method, BasisSet, sto_out, index, ElementName):
    Title = Title+'_'+ElementName.strip()+'_'+arguments.BasisSet.strip()+'_scale_'+str(index+1)
    Energy = Get_Energy(Title, CPU, Z, Charge, Method, BasisSet, sto_out)
    return(index, Energy)

##################################################################################################################################
#Functions to be used by Main
##################################################################################################################################

def Initiate(arguments):
    Z = arguments.Element
    ElementName = GetElementName(Z)
    CPU = arguments.GaussianProc
    Limit = arguments.Limit
    Delta = arguments.Delta

    print ("Test element is {}".format(ElementName))
    print ("Basis set is {}".format(arguments.BasisSet))
    print ("Level of theory is {}".format(arguments.Method))
    print ("The value of Delta is {}".format(arguments.Delta))
    print ("The cutoff is {}".format(arguments.Limit))

    ## files names  ##
    fileName = str(Z) + '_' + GetElementSymbol(Z).strip() + '_' + arguments.BasisSet.strip()
    GuessFile = 'Guess_' + fileName + '.txt'
    EnergyFileI = 'EnergyI_' + fileName
    EnergyFileF = 'EnergyF_' + fileName

    sto = GetSTO(Z, arguments.BasisSet)
    stoLen = len(sto)

    if arguments.Scales is not None:
        if stoLen == len(arguments.Scales):
            print("The guess values ", arguments.Scales)
            guessScale = arguments.Scales
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
    Nr_of_scales = len(guessScale)

    return(Z, ElementName, CPU, Limit, Delta, GuessFile, EnergyFileI, EnergyFileF, sto, guessScale, Nr_of_scales)

def WriteGuessScales(GuessFile, guessScale):
    File = open(GuessFile,'w')
    for val in guessScale:
        File.write(str(val) + '\n')
    File.close()

def GetGradientEnergies(arguments, CPU, Z, ElementName, Sorted_Gradient_scales):
    ll=joblib.Parallel(n_jobs=arguments.ParallelProc)(joblib.delayed(EnergyParallel)('Grad',CPU,Z,arguments.Charge,arguments.Method,arguments.BasisSet,sto_out,index,ElementName)
        for index,sto_out in enumerate(Sorted_Gradient_scales))
    GradientEnergyDictionary={} 
    GradientEnergyDictionary={t[0]:round(t[1], 15) for t in ll}

    Gradient=[]
    for val in range(0, len(GradientEnergyDictionary), 2):
        Gradient.append(round((float(GradientEnergyDictionary[val]) - float(GradientEnergyDictionary[val + 1])) / (2.0 * Delta), 15))
        
    if any(val==0.0 for val in Gradient):
        print(bcolors.FAIL,"\nSTOP STOP: Gradiant contains Zero values", bcolors.ENDC, "\n", Gradient)
        sys.exit(0)
    return(GradientEnergyDictionary, Gradient)

def GetHessianEnergies(arguments, CPU, Z, ElementName, Sorted_Hessian_scales):
    ll=joblib.Parallel(n_jobs=arguments.ParallelProc)(joblib.delayed(EnergyParallel)('Hess',CPU,Z,arguments.Charge,arguments.Method,arguments.BasisSet,sto_out,index,ElementName) 
        for index,sto_out in enumerate(Sorted_Hessian_scales))
    HessianEnergyDictionary={}
    HessianEnergyDictionary={t[0]:round(t[1], 15) for t in ll}
    
    HessianEnergies = []

    for i in range(len(Sorted_Hessian_scales)):
        HessianEnergies.append(HessianEnergyDictionary[i])
 
    HessianE2P = HessianEnergies[ : Nr_of_scales]
    HessianE2N = HessianEnergies[Nr_of_scales : 2 * Nr_of_scales]
    HessianEPP = HessianEnergies[2 * Nr_of_scales : 2 * Nr_of_scales + len(EPPScales)]
    HessianENP = HessianEnergies[2 * Nr_of_scales + len(EPPScales) : 2 * Nr_of_scales + 2 * len(EPPScales)]
    HessianEPN = HessianEnergies[2 * Nr_of_scales + 2 * len(EPPScales) : 2 * Nr_of_scales + 3 * len(EPPScales)]
    HessianENN = HessianEnergies[2 * Nr_of_scales + 3 * len(EPPScales) : ]

    HessianDiagonal = []

    for i in range(Nr_of_scales):
        HessianDiagonal.append((HessianE2P[i] + HessianE2N[i] - 2*E0) /((2.0*Delta)**2))

    HessianUpT = []

    for i in range(len(HessianEPP)):
        HessianUpT.append((HessianEPP[i] - HessianENP[i] - HessianEPN[i] + HessianENN[i]) / ((2.0*Delta)**2))

    Hessian = np.zeros((Nr_of_scales, Nr_of_scales)).tolist()
    
    for i in range(Nr_of_scales):
        for j in range(Nr_of_scales):
            if i == j:
                Hessian[i][i] = HessianDiagonal[i]
                continue
            elif i < j:
                Hessian[i][j] = HessianUpT[i * (Nr_of_scales - i - 1) + j - 1]
                continue
            elif i > j:
                Hessian[i][j] = HessianUpT[j * (Nr_of_scales - j - 1) + i - 1]
                continue
            else:
                print("Wrong value!")
    return(HessianEnergyDictionary, HessianEnergies, Hessian)

def Main():
    arguments = Arguments()
    Z, ElementName, CPU, Limit, Delta, GuessFile, EnergyFileI, EnergyFileF, sto, guessScale, Nr_of_scales = Initiate(arguments)
    WriteGuessScales(GuessFile, guessScale)

    DEnergy=9999999999.99
    E0 = 0.0
    while abs(DEnergy) > abs(Limit):
        #Calculating the initial energy
        if E0 = 0.0:
            E0 = Get_Energy(EnergyFileI,CPU,Z,arguments.Charge,arguments.Method,arguments.BasisSet,guessScale)
        else:
            continue
        
        #Generating scales for the gradient and hessian
        Indices, Diagonal = CreateIndices(Nr_of_scales)
        Gradient_scales, Sorted_Gradient_scales = GetGradientScales(Delta, guessScale)
        E2PScales, E2MScales, EPPScales, ENPScales, EPNScales, ENNScales, Sorted_Hessian_scales = GetHessianScales(Nr_of_scales, Delta, guessScale, Indices)

        #Generating Gaussian input file and running them
        #Gradient
        GradientEnergyDictionary, Gradient = GetGradientEnergies(arguments, CPU, Z, ElementName, Sorted_Gradient_scales)
        
        #Hessian
        HessianEnergyDictionary, HessianEnergies, Hessian = GetHessianEnergies(arguments, CPU, Z, ElementName, Sorted_Hessian_scales)

        print(Gradient)
        print(Hessian)
        break
        HessLen2DInv = np.matrix(Hessian).I
        HessLen2DInv=HessLen2DInv.tolist()
        
        Corr=np.dot(np.matrix(Gradient),np.matrix(HessLen2DInv))
        Corr=Corr.tolist()[0]
        
        guessScale=[float(i) - float(j) for i, j in zip(guessScale, Corr)]

        #Calculating the new energy
        NEnergy=Get_Energy(EnergyFileF,CPU,Z,arguments.Charge,arguments.Method,arguments.BasisSet,guessScale)

        DEnergy=NEnergy-E0

        #print("Hessian")
        #print(Hessian)
        #print("HessianInv")
        #print(HessLen2DInv)
        #print("Corr")
        #print(Corr)
        #print("Gradient")
        #print(Gradient)
        #print("")
        #break

        if DEnergy <= 0.0:
            ColoRR=bcolors.OKGREEN
        else:
            ColoRR=bcolors.FAIL

        print(ColoRR, guessScale, Gradient, DEnergy, NEnergy, E0, bcolors.ENDC)

        #Saving the new E0
        E0 = NEnergy

        #Storing the new scale values
        WriteGuessScales(GuessFile, guessScale) 

if __name__ == "__main__":
    Main()

sys.exit(0)