#!/usr/bin/env python
import numpy as np
import argparse, sys, os, subprocess, joblib, math
from scipy.optimize import minimize#, differential_evolution
import random


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
    colorslist = None
    NumA = None
    NumD = None
    NumS = None
    PRINT = None

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
    parser.add_argument('-i','--InputFileType', required=False,type=str,  help='Input File Style',              default='STO',
                        choices=['STO','A','C','AC','S'])
    parser.add_argument('-b','--BasisSet',      required=False,type=str,  help='Basis set',                        default='6-31G',
                        choices=['6-31G', '6-311G', '6-31G(d,p)'])

    parser.add_argument('-P','--GaussianProc',  required=False,type=int,  help='Number of processors for Gaussian',default=1)
    parser.add_argument('-p','--ParallelProc',  required=False,type=int,  help='Total number of processors used',  default=1)
    parser.add_argument('-s','--Scales',        required=False,type=float,help='Initial scale values',             nargs='+')
    parser.add_argument('-r','--Ranges',        required=False,type=float,help='Range of each scale value',        nargs='+')
    parser.add_argument('-D','--Delta',         required=False,type=float,help='The value of Delta',               default=0.001)
    parser.add_argument('-l','--Limit',         required=False,type=float,help='Error limit',                      default=1.0e-4)
    parser.add_argument('-a','--AlphaValues',   required=False,type=float,help='Alpha values',                     nargs='+')
    parser.add_argument('-d','--Coeffs',        required=False,type=float,help='Coefficient values',               nargs='+')
    parser.add_argument('-A','--AlphaValueRanges',required=False,type=float,help='Ranges for alpha values',        nargs='+')

    arguments = parser.parse_args()

    a.Element = arguments.Element
    a.Z = arguments.Element
    a.Charge = arguments.Charge
    a.OptMethod = arguments.OptMethod
    a.MinMethod = arguments.MinMethod
    a.InputFileType = arguments.InputFileType
    a.BasisSet = arguments.BasisSet
    a.GaussianProc = arguments.GaussianProc
    a.ParallelProc = arguments.ParallelProc
    a.Ranges = arguments.Ranges

    a.colorslist = [bcolors.HEADER,bcolors.OKBLUE,bcolors.WARNING,bcolors.FAIL,bcolors.ENDC,bcolors.BOLD]

    a.Delta = arguments.Delta
    a.Limit = arguments.Limit
    a.AlphaValues = arguments.AlphaValues
    a.AlphaValueRanges = arguments.AlphaValueRanges

    a.ElementName = GetElementName()

    a.NumS, a.NumA, a.NumD = NumberSAD()

    ## files names  ##
    a.fileName = str(a.Element) + '_' + GetElementSymbol().strip()
    a.GuessFile = 'STOGuess_' + a.fileName + '.txt'
    a.EnergyFileI = 'EnergyI_' + a.fileName
    a.EnergyFileF = 'EnergyF_' + a.fileName
    
    a.PRINT = False 
    return(arguments)

# """ Starting the program """"

def Initiate(arguments):    
    print ("Test element is {}".format(a.ElementName))
    print ("Basis set is {}".format(a.BasisSet))
    print ("Level of theory is {}".format(a.OptMethod))
    print ("The value of Delta is {}".format(a.Delta))
    print ("The cutoff is {}".format(a.Limit))
    returnVarScales()

def InitialScales():
    if (a.InputFileType == "STO") :
        sto = GetSTO()
        stoLen = len(sto)

        if arguments.Scales is not None:
            if stoLen == len(arguments.Scales):
                print("The guess values ", arguments.Scales)
                a.IScales = arguments.Scales
            else:
                print(bcolors.FAIL,"\nSTOP STOP: number of guess values should be ", stoLen,bcolors.ENDC)
                sys.exit()
        #elif os.path.isfile(a.GuessFile):
        #        Scales=[]
        #        File = open(a.GuessFile, 'r')
        #        for line in File:
        #            Scales.append(float(line.rstrip('\n')))
        #        File.close()
        #        a.IScales = Scales
        #        print("The guess values (From the File) are ", a.IScales)
        else:
                a.IScales = [1.0] * stoLen
                print("The guess values (Default Values) are ", a.IScales)

    elif (a.InputFileType in ["AC","A","S","C"]) :
# InitScales
        if arguments.Scales is not None:
            if a.NumS == len(arguments.Scales):
                a.IScales = arguments.Scales
            else:
                print(bcolors.FAIL,"\nSTOP STOP: number of guess values should be ", a.NumS,bcolors.ENDC)
                sys.exit()
        else:
            a.IScales = [1.0] * a.NumS
# InitAlphas            
        if arguments.AlphaValues is not None:
            if a.NumA == len(arguments.AlphaValues):
                a.Alphas = arguments.AlphaValues
            else:
                print(bcolors.FAIL,"\nSTOP STOP: number of alpha values should be ", a.NumA, bcolors.ENDC)
                sys.exit()
        else:
             a.Alphas = DefaultAC()[0]
# InitCoeffs
        if arguments.Coeffs is not None:
            if a.NumD == len(arguments.Coeffs):
                a.Coeffs = arguments.Coeffs
            else:
                print(bcolors.FAIL,"\nSTOP STOP: number of coefficient values should be ", a.NumD ,bcolors.ENDC)
                sys.exit()
        else:
            a.Coeffs = DefaultAC()[1]
           
def returnVarScales():

    InitialScales()

    if (a.InputFileType == "STO") :
        a.Scales = a.IScales
    elif (a.InputFileType == "AC"):
        a.Scales = a.Alphas + a.Coeffs
    elif (a.InputFileType == "A"):
        a.Scales = a.Alphas
    elif (a.InputFileType == "C"):
        a.Scales = a.Coeffs
    elif (a.InputFileType == "S"):
        a.Scales = a.IScales 
    

    a.NumberOfScales = len(a.Scales)
    #Numpy array of the scales to be changed, this will be passed in to functions
    a.x0 = np.array(a.Scales)

    if a.Ranges != None:
        a.x_r = np.array(a.Ranges)
        a.x_r = np.reshape(a.x_r, (a.NumberOfScales, 2))   #Ranges for the values to be changed, array of min max pairs

##### Functions related to generating the Input file for scales

def WriteScales(Scales):
    File = open(a.GuessFile,'w')
    for val in Scales:
        File.write(str(val) + '\n')
    File.close()

def NumberSAD():
    if a.Z in [1, 2]:
        NumS = 2; NumA = 4; NumD = 3
    elif a.Z in range(3,5):
        NumS = 3; NumA = 10; NumD = 9
    elif a.Z in range(5,11):
        NumS = 5; NumA = 14; NumD = 12
    elif a.Z in range(11,19):
        NumS = 7; NumA = 26; NumD = 24
    else:
        print('The atomic number is more than 18')
        sys.exit()
    return NumS, NumA, NumD

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

def GenerateInputGen(Scale_values):
    sto = GetSTO()
    inputtext = ''
    for index, sto_out in enumerate(sto):
        inputtext += sto_out + ' ' + str(Scale_values[index])+'\n'
    inputtext += '****\n\n'
    return inputtext

def returnBasisSetsInput(Scale_values):
    inputtext = ""
    if (a.InputFileType == "STO") :
        inputtext += GenerateInputGen(Scale_values)
    elif (a.InputFileType == "AC") :
        AV = Scale_values[:a.NumA+1]
        DV = Scale_values[a.NumA:]
        SV = a.IScales
        inputtext += GenerateInputGenAlpha(SV,DV, AV)
    elif (a.InputFileType == "A") :
        AV = Scale_values
        DV = a.Coeffs
        SV = a.IScales
        inputtext += GenerateInputGenAlpha(SV,DV, AV)
    elif (a.InputFileType == "C") :
        AV = a.Alphas
        DV = Scale_values
        SV = a.IScales
        inputtext += GenerateInputGenAlpha(SV,DV, AV)
    elif (a.InputFileType == "S") :
        AV = a.Alphas
        DV = a.Coeffs
        SV = Scale_values
        inputtext += GenerateInputGenAlpha(SV,DV, AV)
    else:
        print("{} method is not available".format(a.InputFileType))
        sys.exit()
    return inputtext


def GenerateInput(Scale_values):
    inputtext = '%NPROCS=' + str(a.GaussianProc) + '\n' 
    inputtext += '# ' + a.OptMethod + '/gen gfinput\n'
    inputtext += "\n" + a.ElementName.strip() + "\n\n"
    inputtext += "{} {}\n".format(a.Charge, GetElementMultiplicity())
    inputtext += GetElementSymbol().strip() + "\n\n"
    inputtext += GetElementSymbol().strip() + ' 0\n'
    inputtext += returnBasisSetsInput(Scale_values)
    return inputtext
   
def DefaultAC():
    if a.Z in [1, 2]:
        alpha = [2.227660584, 0.405771156, 0.10981751, 
                 0.270949809]
        coffes = [ 0.154328967, 0.535328142, 0.444634542]
    elif a.Z in range(3,5):
        alpha = [23.10303149, 4.235915534, 1.185056519, 0.407098898, 0.158088415, 0.06510954,
                 2.581578398, 0.15676221, 0.060183323, 	
                 0.101215108]
        coffes = [0.009163596, 0.049361493, 0.168538305, 0.37056280, 0.41649153, 0.130334084, 
                 -0.059944749, 0.596038540, 0.458178629]
    elif a.Z in range(5,11):
        alpha = [23.10303149, 4.235915534, 1.185056519, 0.407098898, 0.158088415, 0.06510954,
                 2.581578398, 0.15676221, 0.060183323, 	
                 0.919237900, 0.23591945, 0.080098057,
                 0.101215108,
                 0.175966689]
        coffes = [0.009163596, 0.049361493, 0.168538305, 0.37056280, 0.41649153, 0.130334084, 
                 -0.059944749, 0.596038540, 0.458178629,
                  0.162394855, 0.566170886, 0.422307175]
    elif a.Z in range(11,19):
        print('Not available')
        pass



    return alpha, coffes
    
def GenerateInputGenAlpha(SV,DVV, AV):
    inputtext=''
    DV = []
    if a.Z in [1, 2]:
        inputtext += 'S   3 {0:.12f}    \n'.format(SV[0])
        DV = normalization(DVV[0:3],AV[0:3])[1]
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[0], DV[0])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[1], DV[1])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[2], DV[2])
        inputtext += 'S   1 {0:.12f}    \n'.format(SV[1])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[3], 1.000)
        inputtext += '****\n\n\n'

    elif a.Z in range(3,5):

        inputtext += 'S   6 {0:.12f}    \n'.format(SV[0])
        DV[0:6] = normalization(DVV[0:6],AV[0:6])[1].tolist()
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[0], DV[0])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[1], DV[1])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[2], DV[2])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[3], DV[3])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[4], DV[4])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[5], DV[5])
        inputtext += 'S   3 {0:.12f}    \n'.format(SV[1])
        DV[6:9] = normalization(DVV[6:9],AV[6:9])[1].tolist()
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[6], DV[6])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[7], DV[7])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[8], DV[8])
        inputtext += 'S   1 {0:.12f}    \n'.format(SV[2])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[9], 1.0000)
        inputtext += '****\n\n\n'

    elif a.Z in range(5,11):

        inputtext += 'S   6 {0:.12f}    \n'.format(SV[0])
        DV[0:6] = normalization(DVV[0:6],AV[0:6])[1].tolist()
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[0], DV[0])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[1], DV[1])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[2], DV[2])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[3], DV[3])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[4], DV[4])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[5], DV[5])
        inputtext += 'S   3 {0:.12f}    \n'.format(SV[1])
        DV[6:9] = normalization(DVV[6:9],AV[6:9])[1].tolist()
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[6], DV[6])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[7], DV[7])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[8], DV[8])
        inputtext += 'P   3 {0:.12f}    \n'.format(SV[2])
        DV[9:12] = normalization(DVV[9:12],AV[9:12])[1].tolist()
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[9], DV[9])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[10], DV[10])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[11], DV[11])
        inputtext += 'S   1 {0:.12f}    \n'.format(SV[3])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[12], 1.000)
        inputtext += 'P   1 {0:.12f}    \n'.format(SV[4])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[13], 1.000)
        inputtext += '****\n\n\n'

    elif a.Z in range(11,19):

        inputtext += 'S   6 {0:.12f}    \n'.format(SV[0])
        DV[0:6] = normalization(DVV[0:6],AV[0:6])[1]
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[0], DV[0])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[1], DV[1])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[2], DV[2])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[3], DV[3])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[4], DV[4])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[5], DV[5])
        inputtext += 'S   6 {0:.12f}    \n'.format(SV[1])
        DV[6:12] = normalization(DVV[6:12],AV[6:12])[1]
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[6], DV[6])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[7], DV[7])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[8], DV[8])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[9], DV[9])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[10], DV[10])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[11], DV[11])
        inputtext += 'P   6 {0:.12f}    \n'.format(SV[2])
        DV[12:18] = normalization(DVV[12:18],AV[12:18])[1]
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[12], DV[12])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[13], DV[13])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[14], DV[14])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[15], DV[15])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[16], DV[16])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[17], DV[17])
        inputtext += 'S   3 {0:.12f}    \n'.format(SV[3])
        DV[18:21] = normalization(DVV[18:21],AV[18:21])[1]
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[18], DV[18])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[19], DV[19])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[20], DV[20])
        inputtext += 'P   3 {0:.12f}    \n'.format(SV[4])
        DV[21:24] = normalization(DVV[21:24],AV[21:24])[1]
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[21], DV[21])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[22], DV[22])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[23], DV[23])
        inputtext += 'S   1 {0:.12f}    \n'.format(SV[5])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[24], 1.0000)
        inputtext += 'P   1 {0:.12f}    \n'.format(SV[6])
        inputtext += '      {0:.12f}   {1:.12f}\n'.format(AV[25], 1.0000)
        inputtext += '****\n\n\n'

    return inputtext
#### Functions related to energy 

def Function(Scales):
    Scales_text = "_"+str(a.TextNum)
    Energy = Get_Energy(a.ElementName.strip() + Scales_text, Scales)
    a.TextNum += 1
    return Energy

def EnergyParallel(Title, scales, index):
    Title = Title+'_'+a.ElementName.strip()+'_'+a.BasisSet.strip()+'_scale_'+str(index+1)
    Energy = Get_Energy(Title, scales)
    CurrentColor = random.choice(a.colorslist)
    sys.stdout.write("{}   {}      ; Energy: {} {}\r".format(CurrentColor ,Title, Energy,bcolors.ENDC))
    sys.stdout.flush()
    return(index, Energy)

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
        if a.PRINT: print('Scale Values: {}; Energy: {}'.format(Scale_values, EnergyNUM))
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
            if a.PRINT: print('Scale Values: {}; Energy: {}'.format(Scale_values, EnergyNUM))
            return EnergyNUM
        else:
            print('Scale Values: {}; Energy: ----------'.format(Scale_values))
            print(bcolors.FAIL,"\n STOP STOP: Gaussian job did not terminate normally", bcolors.ENDC)
            print(bcolors.FAIL,"File Name: ", FileName, bcolors.ENDC, "\n\n GOOD LUCK NEXT TIME!!!")
            sys.exit(0)
            return EnergyNUM
    
def Main(arguments):
    Initiate(arguments)

    a.E0 = Get_Energy(a.EnergyFileI, a.Scales)
    if   a.MinMethod == 'en':
        #Calculating the initial energy
        print('\n\n\nEnergy = {} Hartee'.format(a.E0))
        print('End of program: Calculate single energy with given scales.')
    
    elif a.MinMethod == 'own':
        print('{} method is not available'.format(a.MinMethod))
        pass
    
    elif a.MinMethod == 'comb':
        print('{} method is not available'.format(a.MinMethod))
        pass
    
    elif a.MinMethod == 'scan':
        print('{} method is not available'.format(a.MinMethod))
        pass
    
    elif a.MinMethod == 'scan2D':
        print('{} method is not available'.format(a.MinMethod))
        pass
    
    elif a.MinMethod == 'NM':
        a.PRINT = True
        print(bcolors.OKBLUE, '\nStart of program: Minimize energy using Nelder-Mead algorithm from scipy.optimize.minimize python package.\n', bcolors.ENDC)
        a.Result = minimize(Function, a.x0, method='Nelder-Mead', options={'disp': True})
        print('\nThe results are: {}\n'.format(a.Result.x))
        Function(a.Result.x)
        print(bcolors.OKBLUE, '\nEnd of program: Minimize energy using Nelder-Mead algorithm from scipy.optimize.minimize python package.\n', bcolors.ENDC)
    
    elif a.MinMethod == 'CG':
        a.PRINT = True
        print(bcolors.OKBLUE, '\nStart of program: Minimize energy using conjugate gradient algorithm from scipy.optimize.minimize python package.\n', bcolors.ENDC)
        a.Result = minimize(Function, a.x0, method='CG', options={'gtol': a.Limit, 'disp': True})
        print('\nThe results are: {}\n'.format(a.Result.x))
        Function(a.Result.x)
        print(bcolors.OKBLUE, '\nEnd of program: Minimize energy using conjugate gradient algorithm from scipy.optimize.minimize python package.\n', bcolors.ENDC)
    
    elif a.MinMethod == 'LBF':
        #a.PRINT = False
        print(bcolors.OKBLUE, '\nStart of program: Minimize energy using L-BFGS-B algorithm from scipy.optimize.minimize python package.\n', bcolors.ENDC)
        a.Result = minimize(Function, a.x0, jac=GetGradient, method='L-BFGS-B', options={'gtol': a.Limit, 'disp': True})
        print('\nThe results are: {}\n'.format(a.Result.x))
        Function(a.Result.x)
        print(bcolors.OKBLUE, '\nEnd of program: Minimize energy using L-BFGS-B algorithm from scipy.optimize.minimize python package.\n', bcolors.ENDC)
    
    elif a.MinMethod == 'TNC':
        #a.PRINT = False
        if len(a.Ranges) == 2 * len(a.Scales):
            print(bcolors.OKBLUE, '\nStart of program: Minimize energy using truncated Newton (TNC) algorithm from scipy.optimize.minimize python package.\n', bcolors.ENDC)
            a.Result = minimize(Function, a.x0, jac=GetGradient, bounds=a.x_r ,method='TNC', options={'disp': True})
            print('\nThe results are: {}\n'.format(a.Result.x))
            Function(a.Result.x)
            print(bcolors.OKBLUE, '\nEnd of program: Minimize energy using truncated Newton (TNC) algorithm from scipy.optimize.minimize python package.\n', bcolors.ENDC)
        else:
            a.Warnings.append('Ranges (min / max) for each scale value must be given for this method with the option "-r".\nlen(R) = 2 * len(S) condition not met!')
            ErrorTermination()

    elif a.MinMethod == 'NCG':
        ###result = minimize(Function, x0, jac=GetGradient, method='Newton-CG', options={'xtol': a.Limit, 'disp': True})
        print('{} method is not available'.format(a.MinMethod))
        pass
    
    elif a.MinMethod == 'SLS':
        #a.PRINT = False
        result = minimize(Function, a.x0, method='SLSQP', bounds=a.x_r, options={'ftol': a.Limit, 'disp': True})
        pass
    
    elif a.MinMethod == 'TR':
        ##result = minimize(Function, x0, jac=GetGradient, hess=GetHessian, method='trust-ncg', options={'disp': True})
        print('{} method is not available'.format(a.MinMethod))
        pass
    
    elif a.MinMethod == 'GA':
        if len(a.Ranges) == 2 * len(a.Scales):
            print(bcolors.OKBLUE, '\nStart of program: Minimize energy using differential_evolution algorithm from scipy.optimize python package.\n', bcolors.ENDC)
            a.Result = differential_evolution(Function, a.x_r)
            print('\nThe results are: {}\n'.format(a.Result.x))
            Function(a.Result.x)
            print(bcolors.OKBLUE, '\nEnd of program: Minimize energy using differential_evolution algorithm from scipy.optimize python package.\n', bcolors.ENDC)
        else:
            a.Warnings.append('Ranges (min / max) for each scale value must be given for this method with the option "-r".\nlen(R) = 2 * len(S) condition not met!')
            ErrorTermination()

    elif a.MinMethod == 'all':
#     Initial values:
        Ctrl   =1000.0
        RGS    =100.0
        E0     =0.0
        Rho    =0.0
        skip   =0
        counter=0
        Scales = a.x0 
#     Calculating the initial energy:
        if E0 == 0.0:
            E0 = Get_Energy(a.EnergyFileI, Scales)
            DEnergy=np.absolute(E0)
            print("Eo =",'% 12.6f' % E0, " --  Initial DEnergy =", '% 12.6f' % DEnergy)
      
#     Generate Gaussian input files and run them, then calculate Hessian:
            Hessian = GetHessian(Scales)
      
#     >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> while loop <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        while RGS > a.Limit:
            counter += 1 
      
#     Generate Gaussian input files and run them, then calculate Gradient:
            Gradient = GetGradient(np.array(Scales))
      
#     Update Hessian -------------------------------------
            if skip == 0: G0 = Gradient
            if skip == 1:
                dG         = Gradient - G0
                zd         = dG - np.dot(Hessian, dX)
                norm_dX    = np.linalg.norm(dX)
                norm_dG    = np.linalg.norm(dG)
                norm_zd    = np.linalg.norm(zd)
                condition1 = (np.dot(np.transpose(zd), dX)).tolist()[0][0] / (norm_zd * norm_dX)
                condition2 = (np.dot(np.transpose(dG), dX)).tolist()[0][0] / (norm_dG * norm_dX)
#     1) Murtagh-Sargent, symmetric rank one (SR1) update:
                if   condition1 < -0.1:
                    print('\n-----------1) Murtagh-Sargent, symmetric rank one (SR1) update')
                    #print('z_zT', np.dot(zd, np.transpose(zd)))
                    #print('zT_dX', np.dot(np.transpose(zd), dX))
                    Hessian = Hessian + ((np.dot(zd, np.transpose(zd))) / (np.dot(np.transpose(zd), dX)))
#     2) Broyden-Fletcher-Goldfarb-Shanno (BFGS) update:
                elif condition2 >  0.1:
                    print('\n-----------2) Broyden-Fletcher-Goldfarb-Shanno (BFGS) update')
                    #print('OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO')
                    #print('dG_dGT', (np.dot(dG, np.transpose(dG))))
                    #print('dGT_dX', (np.dot(np.transpose(dG), dX)))
                    dGdGt = ((np.dot(dG, np.transpose(dG))) / (np.dot(np.transpose(dG), dX)))
                    HxxtH  = (np.dot(np.dot(np.dot(Hessian, dX), np.transpose(dX)), Hessian)) /  np.dot(np.dot(np.transpose(dX), Hessian), dX)
                    #print('change', dGdGt - HxxtH)
                    Hessian = Hessian + (dGdGt - HxxtH)
#     3) Powell-symmetric-Broyden (PSB) update:
                else:
                    print('\n-----------3) Powell-symmetric-Broyden (PSB) update')
                    #print('dXzT + zdXT', np.dot(dX, np.transpose(zd)) + np.dot(zd, np.transpose(dX)))
                    #print('dXTdX', np.dot(np.transpose(dX), dX))
                    dG_Hx = (  np.dot(dX, np.transpose(zd)) + np.dot(zd, np.transpose(dX)) ) / (np.dot(np.transpose(dX), dX))
                    #print('np.dot(np.transpose(dX), zd)', np.dot(np.transpose(dX), zd))
                    #print('np.dot(dX, np.transpose(dX))', np.dot(dX, np.transpose(dX)))
                    #print('*', np.dot(np.transpose(dX), zd).tolist()[0][0] *  np.dot(dX, np.transpose(dX)) )
                    #print('np.dot(np.transpose(dX), dX)', np.dot(np.transpose(dX), dX))
                    #print('square', np.dot(np.transpose(dX), dX) * np.dot(np.transpose(dX), dX))
                    dxtzdxx = np.dot(np.transpose(dX), zd).tolist()[0][0] * np.dot(dX, np.transpose(dX)) / (np.dot(np.transpose(dX), dX) * np.dot(np.transpose(dX), dX))
                    #print('change', dG_Hx - dxtzdxx)
                    Hessian = Hessian + (dG_Hx - dxtzdxx)
      
            #    print("dG",dG)
            #    print("zd",zd)
            #    print("norm_dX",norm_dX)
            #    print("norm_dG",norm_dG)
            #    print("norm_zd",norm_zd)
            #    print("condition1",condition1)
            #    print("condition2",condition2)
            #    print("dGdGt",dGdGt)
            #    print("xHHx",xHHx)
            #    print("dG_Hx",dG_Hx)
            #    print("xzdxx",xzdxx)
      
#     Calculate Hessian eigenvalues:
            eW, eV = np.linalg.eig(Hessian)
            ew = min(eW)
      
#     Print to the output:
            print()
            print(bcolors.OKBLUE,"Step", counter, "---------------",bcolors.ENDC)
            print()
            print("   Eigenvalues of Hes:","[",', '.join('%8.6f' % i for i in eW),"]")
            print("   Minimum eigenvalue:", '%8.6f' % ew)
            print()
      
#     Control the criteria of the trust region:
            if DEnergy <= 0.0750 and (Rho < 1.250 and  Rho > 0.90): Ctrl = Ctrl/10.0
            if a.Z > 10 and DEnergy <= 1.0 and (Rho < 1.250 and  Rho > 0.90): Ctrl = Ctrl/10.0
            if Ctrl > 10.0 and DEnergy < 1.0e-4 and RGS < 0.01: Ctrl = 1.0
            if Ctrl < 1.0 and (Rho > 1.250 or (Rho < 0.90 and Rho > 0.250)): Ctrl = 1.0
            if Ctrl < 1.0 and (Rho < 1.250 and Rho > 0.90): 
                Ctrl = 0.250
                if RGS < 0.0050 and DEnergy < 1.0e-3: 
                    Ctrl = Ctrl/10.0
                    if RGS < 0.0009: Ctrl = Ctrl/20.0
            if Rho < 0.250: Ctrl = Ctrl * 10.0
            if Ctrl > 1000.0: Ctrl = 1000.0
      
#     Make dx = -inv(H - λI)g if, at least, one of the Hesian eigen values < zero, else dx = -inv(H)g: 
            if ew < 9.0e-3:
                Lambda = (- ew) + Ctrl
                ShiftedHessian = Hessian + Lambda * np.identity(len(Gradient))
                dX = -np.dot(ShiftedHessian.I, Gradient)
                dXList = np.transpose(dX).tolist()[0]
                LastScales = Scales.copy()
                Scales=[float(i) + float(j) for i, j in zip(Scales, dXList)]
      
                '''
                if min(Scales) < 0:
                    # Generate Gaussian input files and run them, then calculate Hessian:
                    Hessian = GetHessian(np.array(LastScales))
      
                    eW, eV = np.linalg.eig(Hessian)
                    ew = min(eW)
      
                    if ew < 9.0e-3:
                        Lambda = (- ew) + Ctrl
                        ShiftedHessian = Hessian + Lambda * np.identity(len(Gradient))
                        dX = -np.dot(ShiftedHessian.I, Gradient)
                        dXList = np.transpose(dX).tolist()[0]
                        Scales=[float(i) + float(j) for i, j in zip(LastScales, dXList)]
                    else:
                        dX = -np.dot(Hessian.I, Gradient)
                        dXList = np.transpose(dX).tolist()[0]
                        Scales=[float(i) + float(j) for i, j in zip(LastScales, dXList)]
                '''
      
                NEnergy=Get_Energy(a.EnergyFileF,Scales)
                DEnergy=E0-NEnergy
                if -(np.dot(np.transpose(Gradient), dX) + 0.5 * np.dot(np.dot(np.transpose(dX), Hessian), dX)).tolist()[0][0] == 0.0:
                    Rho = 0.0
                else:
                    Rho = (DEnergy / -(np.dot(np.transpose(Gradient), dX) + 0.5 * np.dot(np.dot(np.transpose(dX), Hessian), dX)).tolist()[0][0])
                print(bcolors.OKGREEN, "\n  * Ctrl:",Ctrl, bcolors.ENDC)
                print("   lambda:    ",'% 12.6f' % float(Lambda))
                print("   Rho:       ",'% 12.6f' % float(Rho))
            else:
                dX = -np.dot(Hessian.I, Gradient)
                dXList = np.transpose(dX).tolist()[0]
                LastScales = Scales.copy()
                Scales=[float(i) + float(j) for i, j in zip(Scales, dXList)]
      
                #'''
                if min(Scales) < 0:
                    # Generate Gaussian input files and run them, then calculate Hessian:
                    Hessian = GetHessian(LastScales)
      
                    eW, eV = np.linalg.eig(Hessian)
                    ew = min(eW)
      
                    if ew < 9.0e-3:
                        Lambda = (- ew) + Ctrl
                        ShiftedHessian = Hessian + Lambda * np.identity(len(Gradient))
                        dX = -np.dot(ShiftedHessian.I, Gradient)
                        dXList = np.transpose(dX).tolist()[0]
                        Scales=[float(i) + float(j) for i, j in zip(LastScales, dXList)]
                    else:
                        dX = -np.dot(Hessian.I, Gradient)
                        dXList = np.transpose(dX).tolist()[0]
                        Scales=[float(i) + float(j) for i, j in zip(LastScales, dXList)]
                #'''
      
                NEnergy=Get_Energy(a.EnergyFileF, Scales)
                DEnergy=E0-NEnergy
                if -(np.dot(np.transpose(Gradient), dX) + 0.5 * np.dot(np.dot(np.transpose(dX), Hessian), dX)).tolist()[0][0] == 0.0:
                    Rho = 0.0
                else:
                    Rho = (DEnergy / -(np.dot(np.transpose(Gradient), dX) + 0.5 * np.dot(np.dot(np.transpose(dX), Hessian), dX)).tolist()[0][0])
                print("\n   lambda:    ",'% 12.6f' % float(ew))
                print("   Rho:       ",'% 12.6f' % float(Rho))
      
#     calculate the root for the sum of squares of gradient components:  
            Sum_G_Sq = 0.0
            for i in range(len(Gradient)):
                Sum_G_Sq += (Gradient[i] * Gradient[i]) 
            RGS = math.sqrt(Sum_G_Sq)
      
#     Print to the output:
            print(bcolors.BOLD,"  RGS:       ",'% 12.6f' % float(RGS), bcolors.ENDC)      
            ColoRR=bcolors.OKGREEN
            if DEnergy < 0.0: ColoRR=bcolors.FAIL
            print()   
            print(ColoRR,"  New Scale:","[",' '.join('%10.6f' % i for i in Scales),"]") #"   ",np.array(Scales))
            print("   Initial E:","", '% 12.6f' % float(E0))
            print("   Final E  :","", '% 12.6f' % float(NEnergy))
            print("   Delta E  :","", '% 12.6f' % float(DEnergy), bcolors.ENDC)
            print()   
      
#     Saving the new E0, G0, and new scale values:
            WriteScales(Scales)
            E0 = NEnergy
            G0 = Gradient
            skip = 1
      
#     Print to the output if the convergence criteria met, and exit: 
            if (RGS <= a.Limit):
                a.x0 = np.array(Scales)
                print(bcolors.OKGREEN,"STOP:", bcolors.ENDC)
                print("The gradient is", "[",', '.join('%8.6f' % i for i in Gradient),"]", "and RGS =",'% 8.6f' % float(RGS))     
                print(bcolors.OKBLUE,"\n                        -- Optimization Terminated Normally --", bcolors.ENDC)
                print()
# ------------------------------------------------------ Exit ---------------------------------------------------------------
    else:
        a.Warnings.append('This method is unknown')
        ErrorTermination()

    #End of Main() function
    print
    if a.MinMethod == 'all': 
        print("\nThe full basis set is: \n\n{}".format(returnBasisSetsInput(a.x0))) 
    elif a.MinMethod in ['NM','CG','LBF','TNC','GA']:
        print("\nThe full basis set is: \n\n{}".format(returnBasisSetsInput(a.Result.x))) 


#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#""" Element Functions """

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

#""" Basis sets functions """

def GetBasisSetCoreValence():
    if   a.Z < 19                         : return ('6','31')
    elif a.Z in [19,20,31,32,33,34,35,36] : return ('6','6')
    else                                  : return ('','')

def normalization(DV,AV):
    lenDV = len(DV)
    lenAV = len(AV)
    if lenDV != lenAV:
        print('AV != DV')
        sys.exit()
    D_product = np.asarray([dv_i*dv_j for dv_i in DV for dv_j in DV])
    A_product = np.asarray([(a_i*a_j)/(a_i+a_j)**2 for a_i in AV for a_j in AV])
    norm = np.dot(D_product, np.power(A_product,0.75)) * 2.0 * np.sqrt(2.0)
    Corr_DV = DV / np.sqrt(norm)
    return norm, Corr_DV

#""" Gradiant Functions """

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

def GetGradient(Scales):
    print(bcolors.OKBLUE, '\nCalculating Gradient: ', bcolors.ENDC, '\n')
    Gradient_scales, Sorted_Gradient_scales = GetGradientScales(Scales)

    #Parallel    
    ll=joblib.Parallel(n_jobs=a.ParallelProc)(joblib.delayed(EnergyParallel)('Grad',scales,index)
        for index,scales in enumerate(Sorted_Gradient_scales))
    GradientEnergyDictionary={} 
    GradientEnergyDictionary={t[0]:round(t[1], 15) for t in ll}
    print(" "*100)

    """
    #Serial
    GradientEnergyDictionary={}
    p = []
    for index, scales in enumerate(Sorted_Gradient_scales):
        A, B =EnergyParallel('Grad',scales,index)
        p.append([A, B])
    GradientEnergyDictionary={t[0]:round(t[1], 15) for t in p}
    """


    GradientList=[]
    for val in range(0, len(GradientEnergyDictionary), 2):
        GradientList.append(round((float(GradientEnergyDictionary[val]) - float(GradientEnergyDictionary[val + 1])) / (2.0 * a.Delta), 15))
    
    Gradient = np.transpose(np.matrix(GradientList))

    return(Gradient)

#""" Hessian Functiong """

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

def GetHessian(Scales):
    len_hess = int((len(Scales) * (len(Scales) + 1)) * 2 - len(Scales) * 2)
    print(bcolors.OKBLUE, '\nCalculating Hessian: {} runs'.format(len_hess), bcolors.ENDC, '\n')
    Indices, Diagonal = CreateIndices()
    E2PScales, E2MScales, EPPScales, ENPScales, EPNScales, ENNScales, Sorted_Hessian_scales = GetHessianScales(Indices, Scales)

    #Parallel
    ll=joblib.Parallel(n_jobs=a.ParallelProc)(joblib.delayed(EnergyParallel)('Hess',scales,index) 
        for index,scales in enumerate(Sorted_Hessian_scales))
    HessianEnergyDictionary={}
    HessianEnergyDictionary={t[0]:round(t[1], 15) for t in ll}
     
    """
    #Serial
    HessianEnergyDictionary={}
    p = []
    for index, scales in enumerate(Sorted_Hessian_scales):
        A, B = EnergyParallel('Hess',scales,index)
        p.append([A, B])
    HessianEnergyDictionary={t[0]:round(t[1], 15) for t in p}
    """
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

#""" Program Functions """

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


# """ This should be in the end to mstrt the program """

if __name__ == "__main__":
    arguments = Arguments()
    Main(arguments)
    NormalTermination()

