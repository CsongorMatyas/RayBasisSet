#!/usr/bin/env python
import numpy as np
import argparse, sys, os, subprocess, joblib, math
from scipy.optimize import minimize #, differential_evolution

__author__ = "Raymond Poirier's Group - Ahmad Alrawashdeh, Ibrahim Awad, Csongor Matyas"

def Arguments():
    parser = argparse.ArgumentParser(description='Basis Sets optimizing project - Using Newton method')
    parser.add_argument('-o','--OutputFile', nargs='?', type=argparse.FileType('w'), default=sys.stdout, help='Name of the output file (string)')
    parser.add_argument('-e','--Element',       required=True, type=int,  help='Input element atomic number')
    parser.add_argument('-c','--Charge',        required=False,type=int,  help='The charge',                       default=0)
    parser.add_argument('-m','--Method',        required=False,type=str,  help='Level of theory',                  default="UHF")
    parser.add_argument('-b','--BasisSet',      required=False,type=str,  help='Basis set',                        default="6-31G")
    parser.add_argument('-P','--GaussianProc',  required=False,type=int,  help='Number of processors for Gaussian',default=1)
    parser.add_argument('-p','--ParallelProc',  required=False,type=int,  help='Total number of processors used',  default=1)
    parser.add_argument('-s','--Scales',        required=False,type=float,help='Initial scale values',             nargs='+')
    parser.add_argument('-D','--Delta',         required=False,type=float,help='The value of Delta',               default=0.001)
    parser.add_argument('-l','--Limit',         required=False,type=float,help='Error limit',                      default=1.0e-6)
    #parser.add_argument('-G','--Gamma',         required=False,type=float,help='Base gamma coefficient value',     default=0.1)

    parser.add_argument('-X','--NumberOfScales',required=False,type=float,help='DO NOT GIVE! Will be calculated! - Number of scales used')
    parser.add_argument('-Y','--GuessFile',     required=False,type=str  ,help='DO NOT GIVE! Will be calculated! - File that stores recent scales')
    parser.add_argument('-Z','--E0',            required=False,type=float,help='DO NOT GIVE! Will be calculated! - Energy assigned to scale values')
    parser.add_argument('-W','--ElementName',   required=False,type=str  ,help='DO NOT GIVE! Will be calculated! - Name of the element')

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

def GetElementSymbol(arguments):
    if arguments.Element < 1:
        print('Error: the atomic number is less than one (Z<1)\nExit Program')
        sys.exit(0)
    elif float(arguments.Element) > 92:
        print('Error: the atomic number is greater than 92 (Z>92)\nExit Program')
        sys.exit(0)
    Element=['H ',                                                                       'He', 
    'Li','Be',                                                  'B ','C ','N ','O ','F ','Ne', 
    'Na','Mg',                                                  'Al','Si','P ','S ','Cl','Ar', 
    'K ','Ca','Sc','Ti','V ','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr', 
    'Rb','Sr','Y ','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I ','Xe', 
    'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf', 
    'Ta','W ','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U ']
    return Element[arguments.Element - 1]


def GetElementName(arguments):
    if arguments.Element < 1:
        print('Error: the atomic number is less than one (Z<1)\nExit Program')
        sys.exit(0)
    elif arguments.Element > 92:
        print('Error: the atomic number is greater than 92 (Z>92)\nExit Program')
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
    return Element[arguments.Element - 1]

def GetElementGroupPeriod(arguments):
    if arguments.Element == 0:
        print('Error: the atomic number is less than one (Z<1)\nProgram Exit ):')
        sys.exit(0)
    ElementOrder = [2, 8, 8, 18, 18, 36, 36]
    Period = 1
    Group  = 0
    for i in range(len(ElementOrder)):
        if Group < arguments.Element - ElementOrder[i]:
            Group  += ElementOrder[i]
            Period += 1
    Group = arguments.Element - Group
    return(Group, Period)

   
def GetElementMultiplicity(arguments):
    Group, Period = GetElementGroupPeriod(arguments)
    g = Group - arguments.Charge #Everything after this should be checked, or we should find a formula that will calculate multiplicity instead
    if g in [2,4,10,12,18,20,36,38,54,56,86]:
        return 1
    elif g in [1,3,5,9,11,13,17,19,31,35,37,49,53,55,81,85]:
        return 2
    elif g in [6,8,14,16,32,34,50,52,82,84]:
        return 3
    elif g in [7,15,33,51,83]:
        return 4

def GetElementCoreValence(arguments):
    ElementsOrbitals = [['1S'],['2S','2P'],['3S','3P'],['4S','3D','4P'],['5S','4D','5P'],['6S','4F','5D','6P'],['7S','5F','6D','7P']]
    Group, Period = GetElementGroupPeriod(arguments)
    
    AtomicCore=[]
    for sto in range(0, Period - 1):
        for psto in ElementsOrbitals[sto]:
            AtomicCore.append(psto)
    
    AtomicValance=[]
    for vsto in ElementsOrbitals[Period - 1]:
        AtomicValance.append(vsto)
                                                #These also have to be checked
    return(AtomicCore, AtomicValance)

def GetBasisSetCoreValence(arguments):
    Core = arguments.BasisSet[:arguments.BasisSet.find('-')]
    Valence = arguments.BasisSet[arguments.BasisSet.find('-') + 1 : arguments.BasisSet.find('G')]  #RE is needed here :(
    return Core, Valence

def GetSTO(arguments):
    STO=[]
    Core, Valence = GetBasisSetCoreValence(arguments)
    AtomicCore, AtomicValence = GetElementCoreValence(arguments)
    
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

def GenerateFirstLine(arguments):
    FirstLine = '# ' + arguments.Method + '/gen gfprint\n'
    return FirstLine

def GenerateTitle(arguments, Scale_values):
    all_factors = ''

    for i in range(len(Scale_values)):
        all_factors = all_factors + str(Scale_values[i]) + '_'

    Title = "\n" + arguments.ElementName.strip() + "_" + all_factors + "\n\n"
    return Title

def GenerateChargeMultiplicity(arguments):
    ChargeMultiplicity = "{} {}\n".format(arguments.Charge, GetElementMultiplicity(arguments))
    return ChargeMultiplicity

def GenerateZMatrix(arguments):
    ZMatrix = GetElementSymbol(arguments).strip() + "\n\n"
    return ZMatrix

def GenerateCartesianCoordinates(arguments):
    CartesianCoordinates = GetElementSymbol(arguments).strip() + ' 0\n'
    return CartesianCoordinates
    
def GenerateInput(arguments, Scale_values):
    inputtext = '%NPROCS=' + str(arguments.GaussianProc) + '\n' + GenerateFirstLine(arguments)
    inputtext += GenerateTitle(arguments, Scale_values)
    inputtext += GenerateChargeMultiplicity(arguments)
    inputtext += GenerateZMatrix(arguments)
    inputtext += GenerateCartesianCoordinates(arguments)

    sto = GetSTO(arguments)
    for index, sto_out in enumerate(sto):
        inputtext += sto_out + ' ' + str(Scale_values[index])+'\n'
    inputtext += '****\n\n'
    return inputtext

############################################################################################################################
#Functions related to energy and gradient / hessian
############################################################################################################################

def Get_Energy(FileName, arguments, Scale_values):
    file=open(FileName+'.gjf','w')
    file.write(GenerateInput(arguments, Scale_values) + '\n\n')
    file.close()
    
    #subprocess.call('GAUSS_SCRDIR="/nqs/$USER"\n', shell=True)
    subprocess.call('g09 < '+ FileName + '.gjf > ' + FileName + '.out\n', shell=True)
    Energy = subprocess.check_output('grep "SCF Done:" ' + FileName + '.out | tail -1|awk \'{ print $5 }\'', shell=True)
    Energy = Energy.decode('ascii').rstrip('\n')
    if Energy != "":
        EnergyNUM=float(Energy)
        return EnergyNUM

    else:
        file=open(FileName+'.gjf','w')
        file.write(GenerateInput(arguments, Scale_values) + '\n\n')
        file.close()

        subprocess.call('g09 < '+ FileName + '.gjf > ' + FileName + '.out\n', shell=True)
        Energy = subprocess.check_output('grep "SCF Done:" ' + FileName + '.out | tail -1|awk \'{ print $5 }\'', shell=True)
        Energy = Energy.decode('ascii').rstrip('\n')
        if Energy != "":
            EnergyNUM=float(Energy)
            return EnergyNUM
        else:
            print(bcolors.FAIL,"\n STOP STOP: Gaussian job did not terminate normally", bcolors.ENDC)
            print(bcolors.FAIL,"File Name: ", FileName, bcolors.ENDC, "\n\n GOOD LUCK NEXT TIME!!!")
            sys.exit(0)
            return EnergyNUM
    

def GetGradientScales(arguments, Scales):
    Gradient_scales = []
    Sorted_Gradient_scales = []
    for i in range(arguments.NumberOfScales):
        plus = Scales[:].tolist()
        plus[i] = round(Scales[i] + arguments.Delta, 15)
        minus = Scales[:].tolist()
        minus[i] = round(Scales[i] - arguments.Delta, 15)
        Gradient_scales.append([plus, minus])
        Sorted_Gradient_scales.append(plus)
        Sorted_Gradient_scales.append(minus)
    return(Gradient_scales, Sorted_Gradient_scales)

def CreateIndices(arguments):
    Indices = []
    Diagonal = []
    for i in range(arguments.NumberOfScales):
        for j in range(arguments.NumberOfScales):
            if j < i:
               continue
            elif j == i:
               Diagonal.append([i, j])
            else:
               Indices.append([i, j])
    return(Indices, Diagonal)

def CreateE2Scales(arguments, Delta, Scales):
    E2Scales = []
    for i in range(arguments.NumberOfScales):
        iScales = np.zeros(arguments.NumberOfScales).tolist()
        for j in range(arguments.NumberOfScales):
            iScales[j] = Scales[j]
        iScales[i] = iScales[i] + 2 * Delta
        E2Scales.append(iScales)
    return(E2Scales)

def CreateEEScales(arguments, Delta1, Delta2, Indices, Scales):
    EEScales = []
    for (i, j) in Indices:
        ijScales = np.zeros(arguments.NumberOfScales).tolist()
        for k in range(arguments.NumberOfScales):
            ijScales[k] = Scales[k]
        ijScales[i] = ijScales[i] + Delta1
        ijScales[j] = ijScales[j] + Delta2
        EEScales.append(ijScales)
    return(EEScales)

def GetHessianScales(arguments, Indices, Scales):
    E2PScales = CreateE2Scales(arguments, arguments.Delta, Scales)
    E2MScales = CreateE2Scales(arguments, -arguments.Delta, Scales)
    EPPScales = CreateEEScales(arguments, arguments.Delta, arguments.Delta, Indices, Scales)
    ENPScales = CreateEEScales(arguments, -arguments.Delta, arguments.Delta, Indices, Scales)
    EPNScales = CreateEEScales(arguments, arguments.Delta, -arguments.Delta, Indices, Scales)
    ENNScales = CreateEEScales(arguments, -arguments.Delta, -arguments.Delta, Indices, Scales)

    Sorted_Hessian_scales = []
    Sorted_Hessian_scales.extend(E2PScales)
    Sorted_Hessian_scales.extend(E2MScales)
    Sorted_Hessian_scales.extend(EPPScales)
    Sorted_Hessian_scales.extend(ENPScales)
    Sorted_Hessian_scales.extend(EPNScales)
    Sorted_Hessian_scales.extend(ENNScales)
    return(E2PScales, E2MScales, EPPScales, ENPScales, EPNScales, ENNScales, Sorted_Hessian_scales)

def EnergyParallel(Title, arguments, sto_out, index):
    Title = Title+'_'+arguments.ElementName.strip()+'_'+arguments.BasisSet.strip()+'_scale_'+str(index+1)
    Energy = Get_Energy(Title, arguments, sto_out)
    return(index, Energy)

##################################################################################################################################
#Functions to be used by Main
##################################################################################################################################

def Initiate(arguments):
    arguments.ElementName = GetElementName(arguments)
    
    print ("Test element is {}".format(arguments.ElementName))
    print ("Basis set is {}".format(arguments.BasisSet))
    print ("Level of theory is {}".format(arguments.Method))
    print ("The value of Delta is {}".format(arguments.Delta))
    print ("The cutoff is {}".format(arguments.Limit))

    ## files names  ##
    fileName = str(arguments.Element) + '_' + GetElementSymbol(arguments).strip() + '_' + arguments.BasisSet.strip()
    GuessFile = 'Guess_' + fileName + '.txt'
    EnergyFileI = 'EnergyI_' + fileName
    EnergyFileF = 'EnergyF_' + fileName

    sto = GetSTO(arguments)
    stoLen = len(sto)

    if arguments.Scales is not None:
        if stoLen == len(arguments.Scales):
            print("The guess values ", arguments.Scales)
            Scales = arguments.Scales
        else:
            print(bcolors.FAIL,"\nSTOP STOP: number of guess values should be ", stoLen,bcolors.ENDC)
            sys.exit(0)
    elif os.path.isfile(GuessFile):
            Scales=[]
            File = open(GuessFile, 'r')
            for line in File:
                Scales.append(float(line.rstrip('\n')))
            File.close()
            print("The guess values (From the File) are ", Scales)
    else:
            Scales = [1.0] * stoLen
            print("The guess values (Default Values) are ", Scales)

    arguments.GuessFile = GuessFile
    arguments.Scales = Scales
    arguments.NumberOfScales = len(arguments.Scales)

    return(EnergyFileI, EnergyFileF, sto)

def WriteScales(arguments):
    File = open(arguments.GuessFile,'w')
    for val in arguments.Scales:
        File.write(str(val) + '\n')
    File.close()

def GetGradient(Scales):
    global arguments
    Gradient_scales, Sorted_Gradient_scales = GetGradientScales(arguments, Scales)

    #Parallel    
    """
    ll=joblib.Parallel(n_jobs=arguments.ParallelProc)(joblib.delayed(EnergyParallel)('Grad',arguments,sto_out,index)
        for index,sto_out in enumerate(Sorted_Gradient_scales))
    GradientEnergyDictionary={} 
    GradientEnergyDictionary={t[0]:round(t[1], 15) for t in ll}
    """

    #"""
    #Serial
    GradientEnergyDictionary={}
    p = []
    for index, scales in enumerate(Sorted_Gradient_scales):
        a, b =EnergyParallel('Grad',arguments,scales,index)
        p.append([a, b])
    GradientEnergyDictionary={t[0]:round(t[1], 15) for t in p}
    #"""


    GradientList=[]
    for val in range(0, len(GradientEnergyDictionary), 2):
        GradientList.append(round((float(GradientEnergyDictionary[val]) - float(GradientEnergyDictionary[val + 1])) / (2.0 * arguments.Delta), 15))
    
    Gradient = np.transpose(np.matrix(GradientList))
    #Gradient = np.transpose(Gradient)

    if any(val==0.0 for val in GradientList):
        print(bcolors.FAIL,"\nSTOP STOP: Gradient contains Zero values", bcolors.ENDC, "\n", GradientList)
        sys.exit(0)
    return(Gradient)

def GetHessian(Scales):
    global arguments
    Indices, Diagonal = CreateIndices(arguments)
    E2PScales, E2MScales, EPPScales, ENPScales, EPNScales, ENNScales, Sorted_Hessian_scales = GetHessianScales(arguments, Indices, Scales)

    #Parallel
    """
    ll=joblib.Parallel(n_jobs=arguments.ParallelProc)(joblib.delayed(EnergyParallel)('Hess',arguments,sto_out,index) 
        for index,sto_out in enumerate(Sorted_Hessian_scales))
    HessianEnergyDictionary={}
    HessianEnergyDictionary={t[0]:round(t[1], 15) for t in ll}
    """

    #Serial
    HessianEnergyDictionary={}
    p = []
    for index, scales in enumerate(Sorted_Hessian_scales):
        a, b = EnergyParallel('Hess',arguments,scales,index)
        p.append([a, b])
    HessianEnergyDictionary={t[0]:round(t[1], 15) for t in p}
    print(HessianEnergyDictionary)

    HessianEnergies = []

    for i in range(len(Sorted_Hessian_scales)):
        HessianEnergies.append(HessianEnergyDictionary[i])
 
    HessianE2P = HessianEnergies[ : arguments.NumberOfScales]
    HessianE2N = HessianEnergies[arguments.NumberOfScales : 2 * arguments.NumberOfScales]
    HessianEPP = HessianEnergies[2 * arguments.NumberOfScales : 2 * arguments.NumberOfScales + len(EPPScales)]
    HessianENP = HessianEnergies[2 * arguments.NumberOfScales + len(EPPScales) : 2 * arguments.NumberOfScales + 2 * len(EPPScales)]
    HessianEPN = HessianEnergies[2 * arguments.NumberOfScales + 2 * len(EPPScales) : 2 * arguments.NumberOfScales + 3 * len(EPPScales)]
    HessianENN = HessianEnergies[2 * arguments.NumberOfScales + 3 * len(EPPScales) : ]

    HessianDiagonal = []

    for i in range(arguments.NumberOfScales):
        HessianDiagonal.append((HessianE2P[i] + HessianE2N[i] - 2*arguments.E0) /((2.0*arguments.Delta)**2))

    HessianUpT = []

    for i in range(len(HessianEPP)):
        HessianUpT.append((HessianEPP[i] - HessianENP[i] - HessianEPN[i] + HessianENN[i]) / ((2.0*arguments.Delta)**2))

    HessianList = np.zeros((arguments.NumberOfScales, arguments.NumberOfScales)).tolist()
    
    for i in range(arguments.NumberOfScales):
        for j in range(arguments.NumberOfScales):
            if i == j:
                HessianList[i][i] = HessianDiagonal[i]
                continue
            elif i < j:
                HessianList[i][j] = HessianUpT[i * (arguments.NumberOfScales - i - 1) + j - 1]
                continue
            elif i > j:
                HessianList[i][j] = HessianUpT[j * (arguments.NumberOfScales - j - 1) + i - 1]
                continue
            else:
                print("Wrong value!")
    Hessian = np.zeros((len(HessianList), len(HessianList)))
    Hessian = np.matrix(Hessian)

    for i in range(len(HessianList)):
        for j in range(len(HessianList)):
            Hessian[i, j] = HessianList[i][j]

    return Hessian

def Main():
    global arguments
    arguments = Arguments()
    EnergyFileI, EnergyFileF, sto = Initiate(arguments)

    #Save current scales in a file
    WriteScales(arguments)

    #Calculating the initial energy
    arguments.E0 = Get_Energy(EnergyFileI,arguments, arguments.Scales)
    
    #Gradient = GetGradient(Scales)
    #Hessian = GetHessian(Scales)

    def Function(Scales):
        global arguments
        Scales_text = ""
        for i in range(arguments.NumberOfScales):
            Scales_text += "_" + str(Scales[i])
        Energy = Get_Energy(arguments.ElementName.strip() + Scales_text, arguments, Scales)
        print(Scales.tolist(), Energy)
        return Energy

    x0 = np.array(arguments.Scales)

    #result = minimize(Function, x0, method='CG', options={'gtol': arguments.Limit, 'disp': True}) #6 iterations 117 function eval 26 gradient eval
    #51 sec E = -0.49587724265
    #result = minimize(Function, x0, method='Nelder-Mead', options={'disp': True}) #39 iterations 72 function eval
    #32 sec E = -0.495879191425
    result = minimize(Function, x0, jac=GetGradient, method='L-BFGS-B', options={'gtol': arguments.Limit, 'disp': True}) #13 iterations 21 function eval
    #45 sec E = -0.49587945111600001
    ###result = minimize(Function, x0, jac=GetGradient, method='Newton-CG', options={'xtol': arguments.Limit, 'disp': True})
    ##result = minimize(Function, x0, jac=GetGradient, method='TNC', options={'disp': True})
    ##result = minimize(Function, x0, method='COBYLA', options={'disp': True})
    #result = minimize(Function, x0, method='SLSQP', options={'ftol': arguments.Limit, 'disp': True}) #3 iterations 12 function eval 3 gradient eval -0.494978
    ###result = minimize(Function, x0, jac=GetGradient, hess=GetHessian, method='dogleg', options={'disp': True})
    ##result = minimize(Function, x0, jac=GetGradient, hess=GetHessian, method='trust-ncg', options={'disp': True})

    """
    x0_n = np.zeros((arguments.NumberOfScales, 2))
    for i in range(arguments.NumberOfScales):
        x0_n[i, 0] = 0.1
        x0_n[i, 1] = 20
    print(x0_n)


    result = differential_evolution(Function, x0_n)
    """


    print(result.x)
    sys.exit(0)

    #Calculating the new energy
    NEnergy=Get_Energy(EnergyFileF,arguments, arguments.Scales)
    DEnergy=NEnergy-arguments.E0

    if DEnergy <= 0.0:
        ColoRR=bcolors.OKGREEN
    else:
        ColoRR=bcolors.FAIL

    print(ColoRR, arguments.Scales, Gradient, DEnergy, NEnergy, arguments.E0, bcolors.ENDC)

    #Saving the new E0
    arguments.E0 = NEnergy

    #Storing the new scale values
    WriteScales(arguments)

    SumOfSquaredGradientValues = 0.0

    for i in range(arguments.NumberOfScales):
        SumOfSquaredGradientValues += (Gradient[i] * Gradient[i]) 

    Convergence_criteria = math.sqrt(SumOfSquaredGradientValues)

if __name__ == "__main__":
    Main()

sys.exit(0)
