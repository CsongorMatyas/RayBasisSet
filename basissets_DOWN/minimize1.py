#!/usr/bin/env python
import numpy as np
import argparse, sys, os, subprocess, joblib, math
from scipy.optimize import minimize#, differential_evolution

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
    AInput = None           #String that contains input text for alpha minimization
    TextNum = 1

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
    Energy = GetEnergyA(Title, sto_out)
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
    Energy = GetEnergyA(a.ElementName.strip() + Scales_text, Scales)
    return Energy

#def GetInputA():
#    inputtext = ''
#    inputtext += '# {}/gen gfinput\n\nTitle\n\n'.format(str(a.Method))
#    inputtext += '{} {}\n'.format(str(a.Charge), str(mult=GetElementMultiplicity()))
#    inputtext += '{}\n\n{} 0\n'.format(symb=GetElementSymbol(), symb=GetElementSymbol())


def GetEnergyA(FileName, AV):
    a.Z=6
    symb = GetElementSymbol()
    mult = GetElementMultiplicity()
    inputtext = ''
    inputtext += '# {}/gen gfinput\n\nTitle\n\n'.format(str(a.OptMethod))
    inputtext += '{} {}\n'.format(str(a.Charge), str(mult))
    inputtext += '{}\n\n{} 0\n'.format(symb, symb)

    if a.Z == 1:
        DV = ['0.03349460', '0.23472695', '0.81375733', '1.0000000']
        inputtext += 'S   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[0], DV[0])
        inputtext += '      {} {}\n'.format(AV[1], DV[1])
        inputtext += '      {} {}\n'.format(AV[2], DV[2])
        inputtext += 'S   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[3], DV[3])
        inputtext += '****\n\n\n'

    elif a.Z == 2:
        DV = ['0.0237660', '0.1546790', '0.4696300', '1.0000000']
        inputtext += 'S   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[0], DV[0])
        inputtext += '      {} {}\n'.format(AV[1], DV[1])
        inputtext += '      {} {}\n'.format(AV[2], DV[2])
        inputtext += 'S   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[3], DV[3])
        inputtext += '****\n\n\n'

    elif a.Z == 6:

        DV = ['0.0021426', '0.0162089', '0.0773156', '0.2457860', '0.4701890', '0.3454708', '-0.0350917', '-0.1912328', '1.0839878',
              '0.0089415', '0.1410095', '0.9453637', '1.0000000', '1.0000000']
        inputtext += 'S   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[0], AV[14])
        inputtext += '      {} {}\n'.format(AV[1], AV[15])
        inputtext += '      {} {}\n'.format(AV[2], AV[16])
        inputtext += '      {} {}\n'.format(AV[3], AV[17])
        inputtext += '      {} {}\n'.format(AV[4], AV[18])
        inputtext += '      {} {}\n'.format(AV[5], AV[19])
        inputtext += 'S   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[6], AV[20])
        inputtext += '      {} {}\n'.format(AV[7], AV[21])
        inputtext += '      {} {}\n'.format(AV[8], AV[22])
        inputtext += 'P   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[9], AV[23])
        inputtext += '      {} {}\n'.format(AV[10], AV[24])
        inputtext += '      {} {}\n'.format(AV[11], AV[25])
        inputtext += 'S   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[12], 1.000)
        inputtext += 'P   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[13], 1.000)
        inputtext += '****\n\n\n'

    elif a.Z == 4:

        DV = ['0.0019448', '0.0148351', '0.0720906', '0.2371542', '0.4691987', '0.3565202', '-0.1126487', '-0.2295064', '1.1869167', 
              '0.0559802', '0.2615506', '0.7939723', '1.0000000', '1.0000000']
        inputtext += 'S   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[0], DV[0])
        inputtext += '      {} {}\n'.format(AV[1], DV[1])
        inputtext += '      {} {}\n'.format(AV[2], DV[2])
        inputtext += '      {} {}\n'.format(AV[3], DV[3])
        inputtext += '      {} {}\n'.format(AV[4], DV[4])
        inputtext += '      {} {}\n'.format(AV[5], DV[5])
        inputtext += 'S   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[6], DV[6])
        inputtext += '      {} {}\n'.format(AV[7], DV[7])
        inputtext += '      {} {}\n'.format(AV[8], DV[8])
        inputtext += 'P   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[9], DV[9])
        inputtext += '      {} {}\n'.format(AV[10], DV[10])
        inputtext += '      {} {}\n'.format(AV[11], DV[11])
        inputtext += 'S   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[12], DV[12])
        inputtext += 'P   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[13], DV[13])
        inputtext += '****\n\n\n'

    elif a.Z == 5:

        DV = ['0.0018663', '0.0142515', '0.0695516', '0.2325729', '0.4670787', '0.3634314', '-0.1303938', '-0.1307889', '1.1309444', '0.0745976',
              '0.3078467', '0.7434568', '1.0000000', '1.0000000']
        inputtext += 'S   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[0], DV[0])
        inputtext += '      {} {}\n'.format(AV[1], DV[1])
        inputtext += '      {} {}\n'.format(AV[2], DV[2])
        inputtext += '      {} {}\n'.format(AV[3], DV[3])
        inputtext += '      {} {}\n'.format(AV[4], DV[4])
        inputtext += '      {} {}\n'.format(AV[5], DV[5])
        inputtext += 'S   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[6], DV[6])
        inputtext += '      {} {}\n'.format(AV[7], DV[7])
        inputtext += '      {} {}\n'.format(AV[8], DV[8])
        inputtext += 'P   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[9], DV[9])
        inputtext += '      {} {}\n'.format(AV[10], DV[10])
        inputtext += '      {} {}\n'.format(AV[11], DV[11])
        inputtext += 'S   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[12], DV[12])
        inputtext += 'P   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[13], DV[13])
        inputtext += '****\n\n\n'

    elif a.Z == 3:
        VA = ['831.709134','152.492959','42.662035','14.65556','5.691183','2.343943','0.738242','39.98543','7.85956','1.44648','0.796927','0.084293','0.632324','0.996866']
        DV = ['0.009164','0.049361','0.168538','0.370563','0.416492','0.130334','1.076415','-0.126702','-0.386772','0.378103','0.609554','0.063583','1.000000','1.000000']
        inputtext += 'S   6 {}    0.0000000000\n'.format(AV[0])
        inputtext += '      {} {}\n'.format(VA[0], DV[0])
        inputtext += '      {} {}\n'.format(VA[1], DV[1])
        inputtext += '      {} {}\n'.format(VA[2], DV[2])
        inputtext += '      {} {}\n'.format(VA[3], DV[3])
        inputtext += '      {} {}\n'.format(VA[4], DV[4])
        inputtext += '      {} {}\n'.format(VA[5], DV[5])
        inputtext += 'S   3 {}    0.0000000000\n'.format(AV[1])
        inputtext += '      {} {}\n'.format(VA[6], DV[6])
        inputtext += '      {} {}\n'.format(VA[7], DV[7])
        inputtext += '      {} {}\n'.format(VA[8], DV[8])
        inputtext += 'P   3 {}    0.0000000000\n'.format(AV[2])
        inputtext += '      {} {}\n'.format(VA[9], DV[9])
        inputtext += '      {} {}\n'.format(VA[10], DV[10])
        inputtext += '      {} {}\n'.format(VA[11], DV[11])
        inputtext += 'S   1 {}    0.0000000000\n'.format(AV[3])
        inputtext += '      {} {}\n'.format(VA[12], DV[12])
        inputtext += 'P   1 {}    0.0000000000\n'.format(AV[4])
        inputtext += '      {} {}\n'.format(VA[13], DV[13])
        inputtext += '****\n\n\n'

    elif a.Z == 7:

        DV = ['0.0018348', '0.0139950', '0.0685870', '0.2322410', '0.4690700', '0.3604550', '-0.1149610', '-0.1691180', '1.1458520', '0.0675800',
              '0.3239070', '0.7408950', '1.0000000', '1.0000000']
        inputtext += 'S   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[0], DV[0])
        inputtext += '      {} {}\n'.format(AV[1], DV[1])
        inputtext += '      {} {}\n'.format(AV[2], DV[2])
        inputtext += '      {} {}\n'.format(AV[3], DV[3])
        inputtext += '      {} {}\n'.format(AV[4], DV[4])
        inputtext += '      {} {}\n'.format(AV[5], DV[5])
        inputtext += 'S   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[6], DV[6])
        inputtext += '      {} {}\n'.format(AV[7], DV[7])
        inputtext += '      {} {}\n'.format(AV[8], DV[8])
        inputtext += 'P   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[9], DV[9])
        inputtext += '      {} {}\n'.format(AV[10], DV[10])
        inputtext += '      {} {}\n'.format(AV[11], DV[11])
        inputtext += 'S   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[12], DV[12])
        inputtext += 'P   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[13], DV[13])
        inputtext += '****\n\n\n'

    elif a.Z == 8:

        DV = ['0.0018311', '0.0139501', '0.0684451', '0.2327143', '0.4701930', '0.3585209', '-0.1107775', '-0.1480263', '1.1307670', '0.0708743',
              '0.3397528', '0.7271586', '1.0000000', '1.0000000']
        inputtext += 'S   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[0], DV[0])
        inputtext += '      {} {}\n'.format(AV[1], DV[1])
        inputtext += '      {} {}\n'.format(AV[2], DV[2])
        inputtext += '      {} {}\n'.format(AV[3], DV[3])
        inputtext += '      {} {}\n'.format(AV[4], DV[4])
        inputtext += '      {} {}\n'.format(AV[5], DV[5])
        inputtext += 'S   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[6], DV[6])
        inputtext += '      {} {}\n'.format(AV[7], DV[7])
        inputtext += '      {} {}\n'.format(AV[8], DV[8])
        inputtext += 'P   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[9], DV[9])
        inputtext += '      {} {}\n'.format(AV[10], DV[10])
        inputtext += '      {} {}\n'.format(AV[11], DV[11])
        inputtext += 'S   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[12], DV[12])
        inputtext += 'P   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[13], DV[13])
        inputtext += '****\n\n\n'

    elif a.Z == 9:

        DV = ['0.0018196169', '0.0139160796', '0.0684053245', '0.233185760', '0.471267439', '0.356618546', '-0.108506975', '-0.146451658', '1.128688580',
              '0.0716287243', '0.3459121030', '0.7224699570', '1.0000000', '1.0000000']
        inputtext += 'S   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[0], DV[0])
        inputtext += '      {} {}\n'.format(AV[1], DV[1])
        inputtext += '      {} {}\n'.format(AV[2], DV[2])
        inputtext += '      {} {}\n'.format(AV[3], DV[3])
        inputtext += '      {} {}\n'.format(AV[4], DV[4])
        inputtext += '      {} {}\n'.format(AV[5], DV[5])
        inputtext += 'S   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[6], DV[6])
        inputtext += '      {} {}\n'.format(AV[7], DV[7])
        inputtext += '      {} {}\n'.format(AV[8], DV[8])
        inputtext += 'P   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[9], DV[9])
        inputtext += '      {} {}\n'.format(AV[10], DV[10])
        inputtext += '      {} {}\n'.format(AV[11], DV[11])
        inputtext += 'S   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[12], DV[12])
        inputtext += 'P   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[13], DV[13])
        inputtext += '****\n\n\n'

    elif a.Z == 10:

        DV = ['0.0018843481', '0.0143368994', '0.0701096233', '0.2373732660', '0.4730071260', '0.3484012410', '-0.107118287', '-0.146163821', '1.127773500',
              '0.0719095885', '0.3495133720', '0.7199405120', '1.0000000', '1.0000000']
        inputtext += 'S   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[0], DV[0])
        inputtext += '      {} {}\n'.format(AV[1], DV[1])
        inputtext += '      {} {}\n'.format(AV[2], DV[2])
        inputtext += '      {} {}\n'.format(AV[3], DV[3])
        inputtext += '      {} {}\n'.format(AV[4], DV[4])
        inputtext += '      {} {}\n'.format(AV[5], DV[5])
        inputtext += 'S   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[6], DV[6])
        inputtext += '      {} {}\n'.format(AV[7], DV[7])
        inputtext += '      {} {}\n'.format(AV[8], DV[8])
        inputtext += 'P   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[9], DV[9])
        inputtext += '      {} {}\n'.format(AV[10], DV[10])
        inputtext += '      {} {}\n'.format(AV[11], DV[11])
        inputtext += 'S   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[12], DV[12])
        inputtext += 'P   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[13], DV[13])
        inputtext += '****\n\n\n'

    elif a.Z == 11:

        DV = ['0.0019377', '0.0148070', '0.0727060', '0.2526290', '0.4932420', '0.3131690',
              '-0.0035421', '-0.0439590', '-0.1097521', '0.1873980', '0.6466990', '0.3060580',
              '0.0050017', '0.0355110', '0.1428250', '0.3386200', '0.4515790', '0.2732710',
              '-0.2485030', '-0.1317040', '1.2335200', '-0.0230230', '0.9503590', '0.0598580',
              '1.0000000', '1.0000000']
        inputtext += 'S   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[0], DV[0])
        inputtext += '      {} {}\n'.format(AV[1], DV[1])
        inputtext += '      {} {}\n'.format(AV[2], DV[2])
        inputtext += '      {} {}\n'.format(AV[3], DV[3])
        inputtext += '      {} {}\n'.format(AV[4], DV[4])
        inputtext += '      {} {}\n'.format(AV[5], DV[5])
        inputtext += 'S   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[6], DV[6])
        inputtext += '      {} {}\n'.format(AV[7], DV[7])
        inputtext += '      {} {}\n'.format(AV[8], DV[8])
        inputtext += '      {} {}\n'.format(AV[9], DV[9])
        inputtext += '      {} {}\n'.format(AV[10], DV[10])
        inputtext += '      {} {}\n'.format(AV[11], DV[11])
        inputtext += 'P   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[12], DV[12])
        inputtext += '      {} {}\n'.format(AV[13], DV[13])
        inputtext += '      {} {}\n'.format(AV[14], DV[14])
        inputtext += '      {} {}\n'.format(AV[15], DV[15])
        inputtext += '      {} {}\n'.format(AV[16], DV[16])
        inputtext += '      {} {}\n'.format(AV[17], DV[17])
        inputtext += 'S   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[18], DV[18])
        inputtext += '      {} {}\n'.format(AV[19], DV[19])
        inputtext += '      {} {}\n'.format(AV[20], DV[20])
        inputtext += 'P   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[21], DV[21])
        inputtext += '      {} {}\n'.format(AV[22], DV[22])
        inputtext += '      {} {}\n'.format(AV[23], DV[23])
        inputtext += 'S   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[24], DV[24])
        inputtext += 'P   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[25], DV[25])
        inputtext += '****\n\n\n'

    elif a.Z == 12:

        DV = ['0.0019778', '0.0151140', '0.0739110', '0.2491910', '0.4879280', '0.3196620',
              '-0.0032372', '-0.0410080', '-0.1126000', '0.1486330', '0.6164970', '0.3648290',
              '0.0049281', '0.0349890', '0.1407250', '0.3336420', '0.4449400', '0.2692540',
              '-0.2122900', '-0.1079850', '1.1758400', '-0.0224190', '0.1922700', '0.8461810',
              '1.0000000', '1.0000000']
        inputtext += 'S   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[0], DV[0])
        inputtext += '      {} {}\n'.format(AV[1], DV[1])
        inputtext += '      {} {}\n'.format(AV[2], DV[2])
        inputtext += '      {} {}\n'.format(AV[3], DV[3])
        inputtext += '      {} {}\n'.format(AV[4], DV[4])
        inputtext += '      {} {}\n'.format(AV[5], DV[5])
        inputtext += 'S   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[6], DV[6])
        inputtext += '      {} {}\n'.format(AV[7], DV[7])
        inputtext += '      {} {}\n'.format(AV[8], DV[8])
        inputtext += '      {} {}\n'.format(AV[9], DV[9])
        inputtext += '      {} {}\n'.format(AV[10], DV[10])
        inputtext += '      {} {}\n'.format(AV[11], DV[11])
        inputtext += 'P   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[12], DV[12])
        inputtext += '      {} {}\n'.format(AV[13], DV[13])
        inputtext += '      {} {}\n'.format(AV[14], DV[14])
        inputtext += '      {} {}\n'.format(AV[15], DV[15])
        inputtext += '      {} {}\n'.format(AV[16], DV[16])
        inputtext += '      {} {}\n'.format(AV[17], DV[17])
        inputtext += 'S   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[18], DV[18])
        inputtext += '      {} {}\n'.format(AV[19], DV[19])
        inputtext += '      {} {}\n'.format(AV[20], DV[20])
        inputtext += 'P   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[21], DV[21])
        inputtext += '      {} {}\n'.format(AV[22], DV[22])
        inputtext += '      {} {}\n'.format(AV[23], DV[23])
        inputtext += 'S   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[24], DV[24])
        inputtext += 'P   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[25], DV[25])
        inputtext += '****\n\n\n'

    elif a.Z == 13:

        DV = ['0.00194267', '0.0148599', '0.0728494', '0.2468300', '0.4872580', '0.3234960',
              '-0.00292619', '-0.0374080', '-0.1144870', '0.1156350', '0.6125950', '0.3937990',
              '0.00460285', '0.0331990', '0.1362820', '0.3304760', '0.4491460', '0.2657040',
              '-0.2276060', '0.00144583', '1.0927900', '-0.0175130', '0.2445330', '0.8049340',
              '1.0000000', '1.0000000']
        inputtext += 'S   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[0], DV[0])
        inputtext += '      {} {}\n'.format(AV[1], DV[1])
        inputtext += '      {} {}\n'.format(AV[2], DV[2])
        inputtext += '      {} {}\n'.format(AV[3], DV[3])
        inputtext += '      {} {}\n'.format(AV[4], DV[4])
        inputtext += '      {} {}\n'.format(AV[5], DV[5])
        inputtext += 'S   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[6], DV[6])
        inputtext += '      {} {}\n'.format(AV[7], DV[7])
        inputtext += '      {} {}\n'.format(AV[8], DV[8])
        inputtext += '      {} {}\n'.format(AV[9], DV[9])
        inputtext += '      {} {}\n'.format(AV[10], DV[10])
        inputtext += '      {} {}\n'.format(AV[11], DV[11])
        inputtext += 'P   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[12], DV[12])
        inputtext += '      {} {}\n'.format(AV[13], DV[13])
        inputtext += '      {} {}\n'.format(AV[14], DV[14])
        inputtext += '      {} {}\n'.format(AV[15], DV[15])
        inputtext += '      {} {}\n'.format(AV[16], DV[16])
        inputtext += '      {} {}\n'.format(AV[17], DV[17])
        inputtext += 'S   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[18], DV[18])
        inputtext += '      {} {}\n'.format(AV[19], DV[19])
        inputtext += '      {} {}\n'.format(AV[20], DV[20])
        inputtext += 'P   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[21], DV[21])
        inputtext += '      {} {}\n'.format(AV[22], DV[22])
        inputtext += '      {} {}\n'.format(AV[23], DV[23])
        inputtext += 'S   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[24], DV[24])
        inputtext += 'P   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[25], DV[25])
        inputtext += '****\n\n\n'

    elif a.Z == 14:

        DV = ['0.00195948', '0.01492880', '0.07284780', '0.24613000', '0.48591400', '0.32500200',
              '-0.00278094', '-0.03571460', '-0.11498500', '0.09356340', '0.60301700', '0.41895900',
              '0.00443826', '0.03266790', '0.13472100', '0.32867800', '0.44964000', '0.26137200',
              '-0.24463000', '0.00431572', '1.09818000', '-0.01779510', '0.25353900', '0.80066900',
              '1.0000000', '1.0000000']
        inputtext += 'S   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[0], DV[0])
        inputtext += '      {} {}\n'.format(AV[1], DV[1])
        inputtext += '      {} {}\n'.format(AV[2], DV[2])
        inputtext += '      {} {}\n'.format(AV[3], DV[3])
        inputtext += '      {} {}\n'.format(AV[4], DV[4])
        inputtext += '      {} {}\n'.format(AV[5], DV[5])
        inputtext += 'S   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[6], DV[6])
        inputtext += '      {} {}\n'.format(AV[7], DV[7])
        inputtext += '      {} {}\n'.format(AV[8], DV[8])
        inputtext += '      {} {}\n'.format(AV[9], DV[9])
        inputtext += '      {} {}\n'.format(AV[10], DV[10])
        inputtext += '      {} {}\n'.format(AV[11], DV[11])
        inputtext += 'P   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[12], DV[12])
        inputtext += '      {} {}\n'.format(AV[13], DV[13])
        inputtext += '      {} {}\n'.format(AV[14], DV[14])
        inputtext += '      {} {}\n'.format(AV[15], DV[15])
        inputtext += '      {} {}\n'.format(AV[16], DV[16])
        inputtext += '      {} {}\n'.format(AV[17], DV[17])
        inputtext += 'S   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[18], DV[18])
        inputtext += '      {} {}\n'.format(AV[19], DV[19])
        inputtext += '      {} {}\n'.format(AV[20], DV[20])
        inputtext += 'P   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[21], DV[21])
        inputtext += '      {} {}\n'.format(AV[22], DV[22])
        inputtext += '      {} {}\n'.format(AV[23], DV[23])
        inputtext += 'S   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[24], DV[24])
        inputtext += 'P   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[25], DV[25])
        inputtext += '****\n\n\n'

    elif a.Z == 15:

        DV = ['0.0018516', '0.0142062', '0.0699995', '0.2400790', '0.4847620', '0.3352000',
              '-0.00278217', '-0.0360499', '-0.1166310', '0.0968328', '0.6144180', '0.4037980',
              '0.00456462', '0.03369360', '0.13975500', '0.33936200', '0.45092100', '0.23858600',
              '-0.2529230', '0.0328517', '1.0812500', '-0.01776530', '0.27405800', '0.78542100',
              '1.0000000', '1.0000000']
        inputtext += 'S   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[0], DV[0])
        inputtext += '      {} {}\n'.format(AV[1], DV[1])
        inputtext += '      {} {}\n'.format(AV[2], DV[2])
        inputtext += '      {} {}\n'.format(AV[3], DV[3])
        inputtext += '      {} {}\n'.format(AV[4], DV[4])
        inputtext += '      {} {}\n'.format(AV[5], DV[5])
        inputtext += 'S   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[6], DV[6])
        inputtext += '      {} {}\n'.format(AV[7], DV[7])
        inputtext += '      {} {}\n'.format(AV[8], DV[8])
        inputtext += '      {} {}\n'.format(AV[9], DV[9])
        inputtext += '      {} {}\n'.format(AV[10], DV[10])
        inputtext += '      {} {}\n'.format(AV[11], DV[11])
        inputtext += 'P   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[12], DV[12])
        inputtext += '      {} {}\n'.format(AV[13], DV[13])
        inputtext += '      {} {}\n'.format(AV[14], DV[14])
        inputtext += '      {} {}\n'.format(AV[15], DV[15])
        inputtext += '      {} {}\n'.format(AV[16], DV[16])
        inputtext += '      {} {}\n'.format(AV[17], DV[17])
        inputtext += 'S   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[18], DV[18])
        inputtext += '      {} {}\n'.format(AV[19], DV[19])
        inputtext += '      {} {}\n'.format(AV[20], DV[20])
        inputtext += 'P   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[21], DV[21])
        inputtext += '      {} {}\n'.format(AV[22], DV[22])
        inputtext += '      {} {}\n'.format(AV[23], DV[23])
        inputtext += 'S   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[24], DV[24])
        inputtext += 'P   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[25], DV[25])
        inputtext += '****\n\n\n'

    elif a.Z == 16:

        DV = ['0.0018690', '0.0142300', '0.0696960', '0.2384870', '0.4833070', '0.3380740',
              '-0.0023767', '-0.0316930', '-0.1133170', '0.0560900', '0.5922550', '0.4550060',
              '0.0040610', '0.0306810', '0.1304520', '0.3272050', '0.4528510', '0.2560420',
              '-0.2503740', '0.0669570', '1.0545100', '-0.0145110', '0.3102630', '0.7544830',
              '1.0000000', '1.0000000']
        inputtext += 'S   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[0], DV[0])
        inputtext += '      {} {}\n'.format(AV[1], DV[1])
        inputtext += '      {} {}\n'.format(AV[2], DV[2])
        inputtext += '      {} {}\n'.format(AV[3], DV[3])
        inputtext += '      {} {}\n'.format(AV[4], DV[4])
        inputtext += '      {} {}\n'.format(AV[5], DV[5])
        inputtext += 'S   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[6], DV[6])
        inputtext += '      {} {}\n'.format(AV[7], DV[7])
        inputtext += '      {} {}\n'.format(AV[8], DV[8])
        inputtext += '      {} {}\n'.format(AV[9], DV[9])
        inputtext += '      {} {}\n'.format(AV[10], DV[10])
        inputtext += '      {} {}\n'.format(AV[11], DV[11])
        inputtext += 'P   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[12], DV[12])
        inputtext += '      {} {}\n'.format(AV[13], DV[13])
        inputtext += '      {} {}\n'.format(AV[14], DV[14])
        inputtext += '      {} {}\n'.format(AV[15], DV[15])
        inputtext += '      {} {}\n'.format(AV[16], DV[16])
        inputtext += '      {} {}\n'.format(AV[17], DV[17])
        inputtext += 'S   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[18], DV[18])
        inputtext += '      {} {}\n'.format(AV[19], DV[19])
        inputtext += '      {} {}\n'.format(AV[20], DV[20])
        inputtext += 'P   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[21], DV[21])
        inputtext += '      {} {}\n'.format(AV[22], DV[22])
        inputtext += '      {} {}\n'.format(AV[23], DV[23])
        inputtext += 'S   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[24], DV[24])
        inputtext += 'P   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[25], DV[25])
        inputtext += '****\n\n\n'

    elif a.Z == 17:

        DV = ['0.0018330', '0.0140340', '0.0690970', '0.2374520', '0.4830340', '0.3398560',
              '-0.0022974', '-0.0307140', '-0.1125280', '0.0450160', '0.5893530', '0.4652060',
              '0.0039894', '0.0303180', '0.1298800', '0.3279510', '0.4535270', '0.2521540',
              '-0.2518300', '0.0615890', '1.0601800', '-0.0142990', '0.3235720', '0.7435070',
              '1.0000000', '1.0000000']
        inputtext += 'S   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[0], DV[0])
        inputtext += '      {} {}\n'.format(AV[1], DV[1])
        inputtext += '      {} {}\n'.format(AV[2], DV[2])
        inputtext += '      {} {}\n'.format(AV[3], DV[3])
        inputtext += '      {} {}\n'.format(AV[4], DV[4])
        inputtext += '      {} {}\n'.format(AV[5], DV[5])
        inputtext += 'S   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[6], DV[6])
        inputtext += '      {} {}\n'.format(AV[7], DV[7])
        inputtext += '      {} {}\n'.format(AV[8], DV[8])
        inputtext += '      {} {}\n'.format(AV[9], DV[9])
        inputtext += '      {} {}\n'.format(AV[10], DV[10])
        inputtext += '      {} {}\n'.format(AV[11], DV[11])
        inputtext += 'P   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[12], DV[12])
        inputtext += '      {} {}\n'.format(AV[13], DV[13])
        inputtext += '      {} {}\n'.format(AV[14], DV[14])
        inputtext += '      {} {}\n'.format(AV[15], DV[15])
        inputtext += '      {} {}\n'.format(AV[16], DV[16])
        inputtext += '      {} {}\n'.format(AV[17], DV[17])
        inputtext += 'S   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[18], DV[18])
        inputtext += '      {} {}\n'.format(AV[19], DV[19])
        inputtext += '      {} {}\n'.format(AV[20], DV[20])
        inputtext += 'P   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[21], DV[21])
        inputtext += '      {} {}\n'.format(AV[22], DV[22])
        inputtext += '      {} {}\n'.format(AV[23], DV[23])
        inputtext += 'S   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[24], DV[24])
        inputtext += 'P   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[25], DV[25])
        inputtext += '****\n\n\n'

    elif a.Z == 18:

        DV = ['0.00182526', '0.01396860', '0.06870730', '0.23620400', '0.48221400', '0.34204300',
              '-0.00215972', '-0.02907750', '-0.11082700', '0.02769990', '0.57761300', '0.48868800',
              '0.00380665', '0.02923050', '0.12646700', '0.32351000', '0.45489600', '0.25663000',
              '-0.2555920', '0.0378066', '1.0805600', '-0.01591970', '0.32464600', '0.74399000',
              '1.0000000', '1.0000000']
        inputtext += 'S   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[0], DV[0])
        inputtext += '      {} {}\n'.format(AV[1], DV[1])
        inputtext += '      {} {}\n'.format(AV[2], DV[2])
        inputtext += '      {} {}\n'.format(AV[3], DV[3])
        inputtext += '      {} {}\n'.format(AV[4], DV[4])
        inputtext += '      {} {}\n'.format(AV[5], DV[5])
        inputtext += 'S   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[6], DV[6])
        inputtext += '      {} {}\n'.format(AV[7], DV[7])
        inputtext += '      {} {}\n'.format(AV[8], DV[8])
        inputtext += '      {} {}\n'.format(AV[9], DV[9])
        inputtext += '      {} {}\n'.format(AV[10], DV[10])
        inputtext += '      {} {}\n'.format(AV[11], DV[11])
        inputtext += 'P   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[12], DV[12])
        inputtext += '      {} {}\n'.format(AV[13], DV[13])
        inputtext += '      {} {}\n'.format(AV[14], DV[14])
        inputtext += '      {} {}\n'.format(AV[15], DV[15])
        inputtext += '      {} {}\n'.format(AV[16], DV[16])
        inputtext += '      {} {}\n'.format(AV[17], DV[17])
        inputtext += 'S   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[18], DV[18])
        inputtext += '      {} {}\n'.format(AV[19], DV[19])
        inputtext += '      {} {}\n'.format(AV[20], DV[20])
        inputtext += 'P   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[21], DV[21])
        inputtext += '      {} {}\n'.format(AV[22], DV[22])
        inputtext += '      {} {}\n'.format(AV[23], DV[23])
        inputtext += 'S   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[24], DV[24])
        inputtext += 'P   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[25], DV[25])
        inputtext += '****\n\n\n'

    elif a.Z == 19:

        DV = ['', '', '', '', '', '',
              '', '', '', '', '', '',
              '', '', '', '', '', '',
              '', '', '', '', '', '',
              '1.0000000', '1.0000000']
        inputtext += 'S   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[0], DV[0])
        inputtext += '      {} {}\n'.format(AV[1], DV[1])
        inputtext += '      {} {}\n'.format(AV[2], DV[2])
        inputtext += '      {} {}\n'.format(AV[3], DV[3])
        inputtext += '      {} {}\n'.format(AV[4], DV[4])
        inputtext += '      {} {}\n'.format(AV[5], DV[5])
        inputtext += 'S   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[6], DV[6])
        inputtext += '      {} {}\n'.format(AV[7], DV[7])
        inputtext += '      {} {}\n'.format(AV[8], DV[8])
        inputtext += '      {} {}\n'.format(AV[9], DV[9])
        inputtext += '      {} {}\n'.format(AV[10], DV[10])
        inputtext += '      {} {}\n'.format(AV[11], DV[11])
        inputtext += 'P   6 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[12], DV[12])
        inputtext += '      {} {}\n'.format(AV[13], DV[13])
        inputtext += '      {} {}\n'.format(AV[14], DV[14])
        inputtext += '      {} {}\n'.format(AV[15], DV[15])
        inputtext += '      {} {}\n'.format(AV[16], DV[16])
        inputtext += '      {} {}\n'.format(AV[17], DV[17])
        inputtext += 'S   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[18], DV[18])
        inputtext += '      {} {}\n'.format(AV[19], DV[19])
        inputtext += '      {} {}\n'.format(AV[20], DV[20])
        inputtext += 'P   3 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[21], DV[21])
        inputtext += '      {} {}\n'.format(AV[22], DV[22])
        inputtext += '      {} {}\n'.format(AV[23], DV[23])
        inputtext += 'S   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[24], DV[24])
        inputtext += 'P   1 1.00    0.0000000000\n'
        inputtext += '      {} {}\n'.format(AV[25], DV[25])
        inputtext += '****\n\n\n'

    file=open(FileName+'.gjf','w')
    file.write(inputtext)
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
    a.TextNum += 1
    Alpha_text = "EN"+str(a.TextNum)

    #for i in range(len(AV)):
    #    Alpha_text += "_" + str(AV[i])
    Energy = GetEnergyA(a.ElementName.strip() + Alpha_text, AV)
    return Energy


def MinimizeAlphas():
    a.a0 = np.array(a.AlphaValues)
    if a.AlphaValueRanges != None:
        a.a_r = np.array(a.AlphaValueRanges)
        a.a_r = np.reshape(a.a_r, (len(a.AlphaValues), 2))   #Ranges for the values to be changed, array of min max pairs
    print('CG>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
    print(bcolors.OKBLUE, '\nStart of program: Minimize energy.\n', bcolors.ENDC)
    a.Result = minimize(FunctionA, a.a0, method='Nelder-Mead', options={'disp': True})
    #a.Result = minimize(FunctionA, a.a0, method='CG', options={'gtol': a.Limit, 'disp': True})
    #a.Result = minimize(FunctionA, a.a0, jac=GetGradient, method='L-BFGS-B', options={'gtol': a.Limit, 'disp': True})
    #a.Result = minimize(FunctionA, a.a0, jac=GetGradient, bounds=a.a_r ,method='TNC', options={'disp': True})
    #a.Result = differential_evolution(FunctionA, a.a_r)

    print('\nThe results are: {}\n'.format(a.Result.x))
    print(bcolors.OKBLUE, '\nEnd of program: Minimize energy.\n', bcolors.ENDC)

def Main(arguments):
    EnergyFileI, EnergyFileF, sto = Initiate(arguments)


    if   a.MinMethod == 'en':
        #Calculating the initial energy
        a.E0 = GetEnergyA(EnergyFileI, a.AlphaValues)
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
        if len(a.Ranges) == 2 * len(a.AlphaValues):
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
        result = minimize(Function, a.x0, method='SLSQP', bounds=a.x_r, options={'ftol': a.Limit, 'disp': True})
        pass
    
    elif a.MinMethod == 'TR':
        ##result = minimize(Function, x0, jac=GetGradient, hess=GetHessian, method='trust-ncg', options={'disp': True})
        pass
    
    elif a.MinMethod == 'GA':
        if len(a.Ranges) == 2 * len(a.AlphaValues):
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
