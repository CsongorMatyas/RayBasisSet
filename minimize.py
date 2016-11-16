#!/usr/bin/env python
import numpy as np
import argparse, sys, os, subprocess, joblib, math
from scipy.optimize import minimize, differential_evolution

__author__ = "Raymond Poirier's Group - Ahmad Alrawashdeh, Ibrahim Awad, Csongor Matyas"

class a: #Class of the arguments, "a" for short, all arguments that will be passed in any function will be defined here as =None than changed
    OutputFile = None       #Name of the output file
    Element = None          #Z of the desired element
    Z = None                #Z of the desired element
    Charge = None           #Charge of the element
    OptMethod = None        #Method that will be used by Gaussian to get the energy
    MinMethod = None        #Method that will be used to minimize the given scale values
    BasisSet = None         #Basis set that will be used by Gaussian to get the energy (Not exactly, will be split up in STO's)
    GaussianProc = None     #Number of processors that will be used by one gaussian job
    ParallelProc = None     #Total number of processors
    Scales = None           #Scale values to be used as a guess
    Ranges = None           #Ranges of the scale values (length of the ranges list = 2 * lenght of the scales list)
    Delta = None            #The change that will be used to calculate the gradient and hessian
    Limit = None            #Precision limit, 1.0e-6 for example
    ElementName = None      #Name of the element
    Warnings = []           #This will contain any potential warnings related to our part of the code (not necessarily to the packages)
    Result = None           #This will contain the result(s) given by the packages
    NumberOfScales = None   #Number of scales used
    GuessFile = None        #File that stores recent scales
    E0 = None               #Energy assigned to scale values (Initial in every loop)
    x0 = None               #numpy array of scales that will be passed in to scipy functions
    x_r = None              #numpy array of the ranges of the scales that will be passed in to some scipy functions
    AlphaValues = None      #Alpha values to be changed
    AlphaValueRanges = None #Ranges for the Alpha values to be changed
    a0 = None               #numpy array of the alpha values
    a_r = None              #numpy array of the ranges of the alpha values

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def Arguments():
    parser = argparse.ArgumentParser(description='Basis Sets optimizing project - Using various minimizing methods')

    parser.add_argument('-o','--OutputFile', nargs='?', type=argparse.FileType('w'), default=sys.stdout, help='Name of the output file (string)')

    parser.add_argument('-e','--Element',       required=True, type=int,  help='Input element atomic number')
    parser.add_argument('-c','--Charge',        required=False,type=int,  help='The charge',                       default=0)

    parser.add_argument('-m','--OptMethod',     required=False,type=str,  help='Optimization method',              default='UHF',
                        choices=['UHF', 'ROHF', 'HF', 'B3LYP', 'MP2'])
    parser.add_argument('-M','--MinMethod',     required=False,type=str,  help='Minimization method',              default='en',
                        choices=['en', 'own', 'comb', 'scan', 'scan2D','GA', 'CG', 'NM', 'LBF', 'NCG', 'TNC', 'SLS', 'TR', 'all'])
    parser.add_argument('-b','--BasisSet',      required=False,type=str,  help='Basis set',                        default='6-31G',
                        choices=['6-31G', '6-311G', '6-31G(d,p)'])

    parser.add_argument('-P','--GaussianProc',  required=False,type=int,  help='Number of processors for Gaussian',default=1)
    parser.add_argument('-p','--ParallelProc',  required=False,type=int,  help='Total number of processors used',  default=1)
    parser.add_argument('-s','--Scales',        required=False,type=float,help='Initial scale values',             nargs='+')
    parser.add_argument('-r','--Ranges',        required=False,type=float,help='Range of each scale value',        nargs='+')
    parser.add_argument('-D','--Delta',         required=False,type=float,help='The value of Delta',               default=0.001)
    parser.add_argument('-l','--Limit',         required=False,type=float,help='Error limit',                      default=1.0e-6)
    parser.add_argument('-a','--AlphaValues',   required=False,type=float,help='Alpha values',                     nargs='+')
    parser.add_argument('-A','--AlphaValueRanges',required=False,type=float,help='Ranges for alpha values',        nargs='+')

    arguments = parser.parse_args()

    a.Element = arguments.Element
    a.Z = arguments.Element
    a.Charge = arguments.Charge
    a.OptMethod = arguments.OptMethod
    a.MinMethod = arguments.MinMethod
    a.BasisSet = arguments.BasisSet
    a.GaussianProc = arguments.GaussianProc
    a.ParallelProc = arguments.ParallelProc
    a.Scales = arguments.Scales
    a.Ranges = arguments.Ranges

    a.Delta = arguments.Delta
    a.Limit = arguments.Limit
    a.AlphaValues = arguments.AlphaValues
    a.AlphaValueRanges = arguments.AlphaValueRanges

    a.ElementName = GetElementName()

    return(arguments)

##################################################################################################################################################
#Get functions
##################################################################################################################################################

def GetElementSymbol():
    if a.Element < 1:
        print('Error: the atomic number is less than one (Z<1)\nExit Program')
        sys.exit()
    elif float(a.Element) > 92:
        print('Error: the atomic number is greater than 92 (Z>92)\nExit Program')
        sys.exit()
    Element=['H ',                                                                       'He', 
    'Li','Be',                                                  'B ','C ','N ','O ','F ','Ne', 
    'Na','Mg',                                                  'Al','Si','P ','S ','Cl','Ar', 
    'K ','Ca','Sc','Ti','V ','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr', 
    'Rb','Sr','Y ','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I ','Xe', 
    'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf', 
    'Ta','W ','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U ']
    return Element[a.Element - 1]


def GetElementName():
    if a.Element < 1:
        print('Error: the atomic number is less than one (Z<1)\nExit Program')
        sys.exit()
    elif a.Element > 92:
        print('Error: the atomic number is greater than 92 (Z>92)\nExit Program')
        sys.exit()
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
    return Element[a.Element - 1]

def GetElementGroupPeriod():          # Valid for representative elements, only.
    if a.Z == 0:
        print('Error: the atomic number is less than one (Z < 1)\nProgram Exit')
        sys.exit()
    ElementOrder = [2, 8, 8, 18, 18, 36, 36]
    Period = 1
    Group  = 0
    for i in range(len(ElementOrder)):
        if Group < a.Z - ElementOrder[i]:
            Group  += ElementOrder[i]
            Period += 1
    Group = a.Z - Group
    if (a.Z > 30 and a.Z < 37) or (a.Z > 48  and a.Z <  55): Group = Group - 10
    if (a.Z > 80 and a.Z < 87) or (a.Z > 112 and a.Z < 119): Group = Group - 24
    return(Group, Period)
   
def GetElementMultiplicity():
    N_el = a.Z - a.Charge
    if   N_el in [0,2,4,10,12,18,20,36,38,54,56,86]            :return 1
    elif N_el in [1,3,5,9,11,13,17,19,31,35,37,49,53,55,81,85] :return 2
    elif N_el in [6,8,14,16,32,34,50,52,82,84]                 :return 3
    elif N_el in [7,15,33,51,83]                               :return 4

def GetElementCoreValence():    #This has to be extended 
    if   a.Z in [0,1,2]                        : return ([],['1S'])
    elif a.Z in [3,4]                          : return (['1S'],['2S'])
    elif a.Z in [5,6,7,8,9,10]                 : return (['1S'],['2S','2P'])
    elif a.Z in [11,12]                        : return (['1S','2S','2P'],['3S'])
    elif a.Z in [13,14,15,16,17,18]            : return (['1S','2S','2P'],['3S','3P'])
    elif a.Z in [19,20]                        : return (['1S','2S','2P','3S','3P'],['4S'])
    elif a.Z in [21,22,23,24,25,26,27,28,29,30]: return ([],[])
    elif a.Z in [31,32,33,34,35,36]            : return (['1S','2S','2P','3S','3P','3D'],['4S','4P'])
    else                                       : return ([],[])
   
def GetBasisSetCoreValence():
    if   a.Z < 19                         : return ('6','31')
    elif a.Z in [19,20,31,32,33,34,35,36] : return ('6','6')
    else                                  : return ('','')

def GetSTO():
    STO=[]
    Core, Valence = GetBasisSetCoreValence()
    AtomicCore, AtomicValence = GetElementCoreValence()
    
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

def GenerateFirstLine():
    FirstLine = '# ' + a.OptMethod + '/gen gfinput\n'
    return FirstLine

def GenerateTitle():
    Title = "\n" + a.ElementName.strip() + "\n\n"
    return Title

def GenerateChargeMultiplicity():
    ChargeMultiplicity = "{} {}\n".format(a.Charge, GetElementMultiplicity())
    return ChargeMultiplicity

def GenerateZMatrix():
    ZMatrix = GetElementSymbol().strip() + "\n\n"
    return ZMatrix

def GenerateCartesianCoordinates():
    CartesianCoordinates = GetElementSymbol().strip() + ' 0\n'
    return CartesianCoordinates
    
def GenerateInput(Scale_values):
    inputtext = '%NPROCS=' + str(a.GaussianProc) + '\n' + GenerateFirstLine()
    inputtext += GenerateTitle()
    inputtext += GenerateChargeMultiplicity()
    inputtext += GenerateZMatrix()
    inputtext += GenerateCartesianCoordinates()

    sto = GetSTO()

    for index, sto_out in enumerate(sto):
        inputtext += sto_out + ' ' + str(Scale_values[index])+'\n'
    inputtext += '****\n\n'
    return inputtext

############################################################################################################################
#Functions related to energy and gradient / hessian
############################################################################################################################

def Get_Energy(FileName, Scale_values):
    #Save current scales in a file
    WriteScales(Scale_values)


    file=open(FileName+'.gjf','w')
    file.write(GenerateInput(Scale_values) + '\n\n')
    file.close()
    
    #subprocess.call('GAUSS_SCRDIR="/nqs/$USER"\n', shell=True)
    subprocess.call('g09 < '+ FileName + '.gjf > ' + FileName + '.out\n', shell=True)
    Energy = subprocess.check_output('grep "SCF Done:" ' + FileName + '.out | tail -1|awk \'{ print $5 }\'', shell=True)
    Energy = Energy.decode('ascii').rstrip('\n')
    if Energy != "":
        EnergyNUM=float(Energy)
        print('Scale Values: {}; Energy: {}'.format(Scale_values, EnergyNUM))
        return EnergyNUM

    else:
        file=open(FileName+'.gjf','w')
        file.write(GenerateInput(Scale_values) + '\n\n')
        file.close()

        subprocess.call('g09 < '+ FileName + '.gjf > ' + FileName + '.out\n', shell=True)
        Energy = subprocess.check_output('grep "SCF Done:" ' + FileName + '.out | tail -1|awk \'{ print $5 }\'', shell=True)
        Energy = Energy.decode('ascii').rstrip('\n')
        if Energy != "":
            EnergyNUM=float(Energy)
            print('Scale Values: {}; Energy: {}'.format(Scale_values, EnergyNUM))
            return EnergyNUM
        else:
            print('Scale Values: {}; Energy: ----------'.format(Scale_values))
            print(bcolors.FAIL,"\n STOP STOP: Gaussian job did not terminate normally", bcolors.ENDC)
            print(bcolors.FAIL,"File Name: ", FileName, bcolors.ENDC, "\n\n GOOD LUCK NEXT TIME!!!")
            sys.exit(0)
            return EnergyNUM
    

def GetGradientScales(Scales):
    Gradient_scales = []
    Sorted_Gradient_scales = []
    for i in range(a.NumberOfScales):
        plus = Scales[:].tolist()
        plus[i] = round(Scales[i] + a.Delta, 15)
        minus = Scales[:].tolist()
        minus[i] = round(Scales[i] - a.Delta, 15)
        Gradient_scales.append([plus, minus])
        Sorted_Gradient_scales.append(plus)
        Sorted_Gradient_scales.append(minus)
    return(Gradient_scales, Sorted_Gradient_scales)

def CreateIndices():
    Indices = []
    Diagonal = []
    for i in range(a.NumberOfScales):
        for j in range(a.NumberOfScales):
            if j < i:
               continue
            elif j == i:
               Diagonal.append([i, j])
            else:
               Indices.append([i, j])
    return(Indices, Diagonal)

def CreateE2Scales(Delta, Scales):
    E2Scales = []
    for i in range(a.NumberOfScales):
        iScales = np.zeros(a.NumberOfScales).tolist()
        for j in range(a.NumberOfScales):
            iScales[j] = Scales[j]
        iScales[i] = iScales[i] + 2 * Delta
        E2Scales.append(iScales)
    return(E2Scales)

def CreateEEScales(Delta1, Delta2, Indices, Scales):
    EEScales = []
    for (i, j) in Indices:
        ijScales = np.zeros(a.NumberOfScales).tolist()
        for k in range(a.NumberOfScales):
            ijScales[k] = Scales[k]
        ijScales[i] = ijScales[i] + Delta1
        ijScales[j] = ijScales[j] + Delta2
        EEScales.append(ijScales)
    return(EEScales)

def GetHessianScales(Indices, Scales):
    E2PScales = CreateE2Scales(a.Delta, Scales)
    E2MScales = CreateE2Scales(-a.Delta, Scales)
    EPPScales = CreateEEScales(a.Delta, a.Delta, Indices, Scales)
    ENPScales = CreateEEScales(-a.Delta, a.Delta, Indices, Scales)
    EPNScales = CreateEEScales(a.Delta, -a.Delta, Indices, Scales)
    ENNScales = CreateEEScales(-a.Delta, -a.Delta, Indices, Scales)

    Sorted_Hessian_scales = []
    Sorted_Hessian_scales.extend(E2PScales)
    Sorted_Hessian_scales.extend(E2MScales)
    Sorted_Hessian_scales.extend(EPPScales)
    Sorted_Hessian_scales.extend(ENPScales)
    Sorted_Hessian_scales.extend(EPNScales)
    Sorted_Hessian_scales.extend(ENNScales)
    return(E2PScales, E2MScales, EPPScales, ENPScales, EPNScales, ENNScales, Sorted_Hessian_scales)

def EnergyParallel(Title, sto_out, index):
    Title = Title+'_'+a.ElementName.strip()+'_'+a.BasisSet.strip()+'_scale_'+str(index+1)
    Energy = Get_Energy(Title, sto_out)
    return(index, Energy)

##################################################################################################################################
#Functions to be used by Main
##################################################################################################################################

def Initiate(arguments):    
    print ("Test element is {}".format(a.ElementName))
    print ("Basis set is {}".format(a.BasisSet))
    print ("Level of theory is {}".format(a.OptMethod))
    print ("The value of Delta is {}".format(a.Delta))
    print ("The cutoff is {}".format(a.Limit))

    ## files names  ##
    fileName = str(a.Element) + '_' + GetElementSymbol().strip()
    GuessFile = 'Guess_' + fileName + '.txt'
    EnergyFileI = 'EnergyI_' + fileName
    EnergyFileF = 'EnergyF_' + fileName

    sto = GetSTO()
    stoLen = len(sto)

    if arguments.Scales is not None:
        if stoLen == len(arguments.Scales):
            print("The guess values ", arguments.Scales)
            Scales = arguments.Scales
        else:
            print(bcolors.FAIL,"\nSTOP STOP: number of guess values should be ", stoLen,bcolors.ENDC)
            sys.exit()
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

    a.GuessFile = GuessFile
    a.Scales = Scales
    a.NumberOfScales = len(a.Scales)

    #Numpy array of the scales to be changed, this will be passed in to functions
    a.x0 = np.array(a.Scales)

    if a.Ranges != None:
        a.x_r = np.array(a.Ranges)
        a.x_r = np.reshape(a.x_r, (a.NumberOfScales, 2))   #Ranges for the values to be changed, array of min max pairs


    return(EnergyFileI, EnergyFileF, sto)

def WriteScales(Scales):
    File = open(a.GuessFile,'w')
    for val in Scales:
        File.write(str(val) + '\n')
    File.close()

def GetGradient(Scales):
    print(bcolors.OKBLUE, '\nCalculating Gradient for scales: ', bcolors.ENDC, '{}\n\n'.format(Scales.tolist()))
    Gradient_scales, Sorted_Gradient_scales = GetGradientScales(Scales)

    #Parallel    
    """
    ll=joblib.Parallel(n_jobs=a.ParallelProc)(joblib.delayed(EnergyParallel)('Grad',sto_out,index)
        for index,sto_out in enumerate(Sorted_Gradient_scales))
    GradientEnergyDictionary={} 
    GradientEnergyDictionary={t[0]:round(t[1], 15) for t in ll}
    #"""

    #"""
    #Serial
    GradientEnergyDictionary={}
    p = []
    for index, scales in enumerate(Sorted_Gradient_scales):
        A, B =EnergyParallel('Grad',scales,index)
        p.append([A, B])
    GradientEnergyDictionary={t[0]:round(t[1], 15) for t in p}
    #"""


    GradientList=[]
    for val in range(0, len(GradientEnergyDictionary), 2):
        GradientList.append(round((float(GradientEnergyDictionary[val]) - float(GradientEnergyDictionary[val + 1])) / (2.0 * a.Delta), 15))
    
    Gradient = np.transpose(np.matrix(GradientList))
    #Gradient = np.transpose(Gradient)

    """
    if any(val==0.0 for val in GradientList):
        print(bcolors.FAIL,"\nSTOP STOP: Gradient contains Zero values", bcolors.ENDC, "\n", GradientList)
        sys.exit(0)
    """
    return(Gradient)

def GetHessian(Scales):

    print(bcolors.OKBLUE, '\nCalculating Hessian for scales: ', bcolors.ENDC, '{}\n\n'.format(Scales.tolist()))
    Indices, Diagonal = CreateIndices()
    E2PScales, E2MScales, EPPScales, ENPScales, EPNScales, ENNScales, Sorted_Hessian_scales = GetHessianScales(Indices, Scales)

    #Parallel
    """
    ll=joblib.Parallel(n_jobs=a.ParallelProc)(joblib.delayed(EnergyParallel)('Hess',sto_out,index) 
        for index,sto_out in enumerate(Sorted_Hessian_scales))
    HessianEnergyDictionary={}
    HessianEnergyDictionary={t[0]:round(t[1], 15) for t in ll}
    #"""

    #Serial
    HessianEnergyDictionary={}
    p = []
    for index, scales in enumerate(Sorted_Hessian_scales):
        A, B = EnergyParallel('Hess',scales,index)
        p.append([A, B])
    HessianEnergyDictionary={t[0]:round(t[1], 15) for t in p}
    print(HessianEnergyDictionary)

    HessianEnergies = []

    for i in range(len(Sorted_Hessian_scales)):
        HessianEnergies.append(HessianEnergyDictionary[i])
 
    HessianE2P = HessianEnergies[ : a.NumberOfScales]
    HessianE2N = HessianEnergies[a.NumberOfScales : 2 * a.NumberOfScales]
    HessianEPP = HessianEnergies[2 * a.NumberOfScales : 2 * a.NumberOfScales + len(EPPScales)]
    HessianENP = HessianEnergies[2 * a.NumberOfScales + len(EPPScales) : 2 * a.NumberOfScales + 2 * len(EPPScales)]
    HessianEPN = HessianEnergies[2 * a.NumberOfScales + 2 * len(EPPScales) : 2 * a.NumberOfScales + 3 * len(EPPScales)]
    HessianENN = HessianEnergies[2 * a.NumberOfScales + 3 * len(EPPScales) : ]

    HessianDiagonal = []

    for i in range(a.NumberOfScales):
        HessianDiagonal.append((HessianE2P[i] + HessianE2N[i] - 2*a.E0) /((2.0*a.Delta)**2))

    HessianUpT = []

    for i in range(len(HessianEPP)):
        HessianUpT.append((HessianEPP[i] - HessianENP[i] - HessianEPN[i] + HessianENN[i]) / ((2.0*a.Delta)**2))

    HessianList = np.zeros((a.NumberOfScales, a.NumberOfScales)).tolist()
    
    for i in range(a.NumberOfScales):
        for j in range(a.NumberOfScales):
            if i == j:
                HessianList[i][i] = HessianDiagonal[i]
                continue
            elif i < j:
                HessianList[i][j] = HessianUpT[i * (a.NumberOfScales - i - 1) + j - 1]
                continue
            elif i > j:
                HessianList[i][j] = HessianUpT[j * (a.NumberOfScales - j - 1) + i - 1]
                continue
            else:
                print("Wrong value!")
    Hessian = np.zeros((len(HessianList), len(HessianList)))
    Hessian = np.matrix(Hessian)

    for i in range(len(HessianList)):
        for j in range(len(HessianList)):
            Hessian[i, j] = HessianList[i][j]

    return Hessian

def NormalTermination():
    print('\n')
    for w in a.Warnings:
        print(bcolors.WARNING,   '\n{}\n'.format(w), bcolors.ENDC)
    if a.Warnings == []:
        COLORR = bcolors.OKGREEN
    else:
        COLORR = bcolors.OKBLUE
    print(COLORR, '\n           ------------Normal termination------------           \n', bcolors.ENDC)
    sys.exit(0)

def ErrorTermination():
    print('\n')
    for w in a.Warnings:
        print(bcolors.WARNING,   '\n{}\n'.format(w), bcolors.ENDC)
    print(bcolors.FAIL, '\n           ------------Error  termination------------           \n', bcolors.ENDC)
    sys.exit(0)    

def Function(Scales):
    Scales_text = ""
    for i in range(len(Scales)):
        Scales_text += "_" + str(Scales[i])
    Energy = Get_Energy(a.ElementName.strip() + Scales_text, Scales)
    return Energy

def GetEnergyA(FileName, AV):
    file=open(FileName+'.gjf','w')
    file.write('# HF/gen gfinput\n\nTitle\n\n0 2\nH\n\nH 0\nS   3 1.00    0.0000000000\n     {} 0.2549381454D-01\n'.format(AV[0]))
    file.write('     {} 0.1903731086D+00\n     {} 0.8521614860D+00\nS   1 1.00    0.0000000000\n     {} 1.0\n****\n\n\n'.format(AV[1], AV[2], AV[3]))
    file.close()
    
    subprocess.call('g09 < '+ FileName + '.gjf > ' + FileName + '.out\n', shell=True)
    Energy = subprocess.check_output('grep "SCF Done:" ' + FileName + '.out | tail -1|awk \'{ print $5 }\'', shell=True)
    Energy = Energy.decode('ascii').rstrip('\n')
    if Energy != "":
        EnergyNUM=float(Energy)
        print('Alpha Values: {}; Energy: {}'.format(AV, EnergyNUM))
        return EnergyNUM

    else:
        file=open(FileName+'.gjf','w')
        file.write('# HF/gen gfinput\n\nTitle\n\n0 2\nH\n\nH 0\nS   3 1.00    0.0000000000\n     {} 0.2549381454D-01\n'.format(AV[0]))
        file.write('     {} 0.1903731086D+00\n     {} 0.8521614860D+00\nS   1 1.00    0.0000000000\n     {} 1.0\n****\n\n\n'.format(AV[1], AV[2], AV[3]))
        file.close()

        subprocess.call('g09 < '+ FileName + '.gjf > ' + FileName + '.out\n', shell=True)
        Energy = subprocess.check_output('grep "SCF Done:" ' + FileName + '.out | tail -1|awk \'{ print $5 }\'', shell=True)
        Energy = Energy.decode('ascii').rstrip('\n')
        if Energy != "":
            EnergyNUM=float(Energy)
            print('Alpha Values: {}; Energy: {}'.format(AV, EnergyNUM))
            return EnergyNUM
        else:
            print('Alpha Values: {}; Energy: ----------'.format(AV))
            print(bcolors.FAIL,"\n STOP STOP: Gaussian job did not terminate normally", bcolors.ENDC)
            print(bcolors.FAIL,"File Name: ", FileName, bcolors.ENDC, "\n\n GOOD LUCK NEXT TIME!!!")
            sys.exit(0)
            return EnergyNUM

def FunctionA(AV):
    Alpha_text = ""
    for i in range(len(AV)):
        Alpha_text += "_" + str(AV[i])
    Energy = GetEnergyA(a.ElementName.strip() + Alpha_text, AV)
    return Energy


def MinimizeAlphas():
    a.a0 = np.array(a.AlphaValues)
    if a.AlphaValueRanges != None:
        a.a_r = np.array(a.AlphaValueRanges)
        a.a_r = np.reshape(a.a_r, (len(a.AlphaValues), 2))   #Ranges for the values to be changed, array of min max pairs

    print(bcolors.OKBLUE, '\nStart of program: Minimize energy.\n', bcolors.ENDC)
    #a.Result = minimize(FunctionA, a.a0, method='Nelder-Mead', options={'disp': True})
    #a.Result = minimize(FunctionA, a.a0, method='CG', options={'gtol': a.Limit, 'disp': True})
    #a.Result = minimize(FunctionA, a.a0, jac=GetGradient, method='L-BFGS-B', options={'gtol': a.Limit, 'disp': True})
    #a.Result = minimize(FunctionA, a.a0, jac=GetGradient, bounds=a.a_r ,method='TNC', options={'disp': True})
    a.Result = differential_evolution(FunctionA, a.a_r)

    print('\nThe results are: {}\n'.format(a.Result.x))
    print(bcolors.OKBLUE, '\nEnd of program: Minimize energy.\n', bcolors.ENDC)

def Main(arguments):
    EnergyFileI, EnergyFileF, sto = Initiate(arguments)

    #Calculating the initial energy
    #a.E0 = Get_Energy(EnergyFileI, a.Scales)

    if   a.MinMethod == 'en':
        print('End of program: Calculate single energy with given scales.')
    
    elif a.MinMethod == 'own':
        pass
    
    elif a.MinMethod == 'comb':
        pass
    
    elif a.MinMethod == 'scan':
        pass
    
    elif a.MinMethod == 'scan2D':
        pass
    
    elif a.MinMethod == 'NM':
        print(bcolors.OKBLUE, '\nStart of program: Minimize energy using Nelder-Mead algorithm from scipy.optimize.minimize python package.\n', bcolors.ENDC)
        a.Result = minimize(Function, a.x0, method='Nelder-Mead', options={'disp': True})
        print('\nThe results are: {}\n'.format(a.Result.x))
        print(bcolors.OKBLUE, '\nEnd of program: Minimize energy using Nelder-Mead algorithm from scipy.optimize.minimize python package.\n', bcolors.ENDC)
    
    elif a.MinMethod == 'CG':
        print(bcolors.OKBLUE, '\nStart of program: Minimize energy using conjugate gradient algorithm from scipy.optimize.minimize python package.\n', bcolors.ENDC)
        a.Result = minimize(Function, a.x0, method='CG', options={'gtol': a.Limit, 'disp': True})
        print('\nThe results are: {}\n'.format(a.Result.x))
        print(bcolors.OKBLUE, '\nEnd of program: Minimize energy using conjugate gradient algorithm from scipy.optimize.minimize python package.\n', bcolors.ENDC)
    
    elif a.MinMethod == 'LBF':
        print(bcolors.OKBLUE, '\nStart of program: Minimize energy using L-BFGS-B algorithm from scipy.optimize.minimize python package.\n', bcolors.ENDC)
        a.Result = minimize(Function, a.x0, jac=GetGradient, method='L-BFGS-B', options={'gtol': a.Limit, 'disp': True})
        print('\nThe results are: {}\n'.format(a.Result.x))
        print(bcolors.OKBLUE, '\nEnd of program: Minimize energy using L-BFGS-B algorithm from scipy.optimize.minimize python package.\n', bcolors.ENDC)
    
    elif a.MinMethod == 'TNC':
        if len(a.Ranges) == 2 * len(a.Scales):
            print(bcolors.OKBLUE, '\nStart of program: Minimize energy using truncated Newton (TNC) algorithm from scipy.optimize.minimize python package.\n', bcolors.ENDC)
            a.Result = minimize(Function, a.x0, jac=GetGradient, bounds=a.x_r ,method='TNC', options={'disp': True})
            print('\nThe results are: {}\n'.format(a.Result.x))
            print(bcolors.OKBLUE, '\nEnd of program: Minimize energy using truncated Newton (TNC) algorithm from scipy.optimize.minimize python package.\n', bcolors.ENDC)
        else:
            a.Warnings.append('Ranges (min / max) for each scale value must be given for this method with the option "-r".\nlen(R) = 2 * len(S) condition not met!')
            ErrorTermination()

    elif a.MinMethod == 'NCG':
        ###result = minimize(Function, x0, jac=GetGradient, method='Newton-CG', options={'xtol': a.Limit, 'disp': True})
        pass
    
    elif a.MinMethod == 'SLS':
        #result = minimize(Function, x0, method='SLSQP', bounds=x_r, options={'ftol': a.Limit, 'disp': True})
        pass
    
    elif a.MinMethod == 'TR':
        ##result = minimize(Function, x0, jac=GetGradient, hess=GetHessian, method='trust-ncg', options={'disp': True})
        pass
    
    elif a.MinMethod == 'GA':
        if len(a.Ranges) == 2 * len(a.Scales):
            print(bcolors.OKBLUE, '\nStart of program: Minimize energy using differential_evolution algorithm from scipy.optimize python package.\n', bcolors.ENDC)
            a.Result = differential_evolution(Function, a.x_r)
            print('\nThe results are: {}\n'.format(a.Result.x))
            print(bcolors.OKBLUE, '\nEnd of program: Minimize energy using differential_evolution algorithm from scipy.optimize python package.\n', bcolors.ENDC)
        else:
            a.Warnings.append('Ranges (min / max) for each scale value must be given for this method with the option "-r".\nlen(R) = 2 * len(S) condition not met!')
            ErrorTermination()
    elif a.MinMethod == 'all':
        MinimizeAlphas()
    else:
        a.Warnings.append('This method is unknown')
        ErrorTermination()

    #End of Main() function

if __name__ == "__main__":
    arguments = Arguments()
    Main(arguments)
    NormalTermination()
