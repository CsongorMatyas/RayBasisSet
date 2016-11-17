#!/usr/bin/env python
import numpy as np
import argparse, sys, os, subprocess, joblib, math
#from scipy.optimize import minimize

__author__ = "Raymond Poirier's Group - Ahmad Alrawashdeh, Ibrahim Awad, Csongor Matyas"

def Arguments():
    parser = argparse.ArgumentParser(description='Basis Sets project')
    parser.add_argument('-e','--Element',     required=True, type=int,  help='Input element atomic number'                           )
    parser.add_argument('-c','--Charge',      required=False,type=int,  help='The charge',                            default=0      )
    parser.add_argument('-m','--Method',      required=False,type=str,  help='Level of theory',                       default="UHF"  )
    parser.add_argument('-b','--BasisSet',    required=False,type=str,  help='Basis set',                             default="6-31G")
    parser.add_argument('-P','--GaussianProc',required=False,type=int,  help='Number of processors for Gaussian',     default=1      )
    parser.add_argument('-p','--ParallelProc',required=False,type=int,  help='Total number of processors used',       default=1      )
    parser.add_argument('-s','--Scales',      required=False,type=float,help='Initial scale values',                  nargs='+'      )
    parser.add_argument('-D','--Delta',       required=False,type=float,help='The value of Delta',                    default=0.001  )
    parser.add_argument('-l','--Limit',       required=False,type=float,help='Error limit',                           default=1.0e-4 )
    parser.add_argument('-G','--Gamma',       required=False,type=float,help='Base gamma coefficient value',          default=0.1    )

    arguments = parser.parse_args()
    return(arguments)

class bcolors:
    HEADER    = '\033[95m'
    OKBLUE    = '\033[94m'
    OKGREEN   = '\033[92m'
    WARNING   = '\033[93m'
    FAIL      = '\033[91m'
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'

np.set_printoptions(precision=6)
##################################################################################################################################################
#Get functions
##################################################################################################################################################

def GetElementSymbol(Z):
    if Z < 1:
        print('Error: the atomic number is less than one (Z < 1)\nProgram Exit')
        sys.exit(0)
    elif Z > 92:
        print('Error: the atomic number is greater than 92 (Z > 92)\nProgram Exit')
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
        print('Error: the atomic number is less than one (Z < 1)\nProgram Exit')
        sys.exit(0)
    elif Z > 92:
        print('Error: the atomic number is greater than 92 (Z > 92)\nProgram Exit')
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

def GetElementGroupPeriod(Z):          # Valid for representative elements, only.
    if Z == 0:
        print('Error: the atomic number is less than one (Z < 1)\nProgram Exit')
        sys.exit(0)
    ElementOrder = [2, 8, 8, 18, 18, 36, 36]
    Period = 1
    Group  = 0
    for i in range(len(ElementOrder)):
        if Group < Z - ElementOrder[i]:
            Group  += ElementOrder[i]
            Period += 1
    Group = Z - Group
    if (Z > 30 and Z < 37) or (Z > 48  and Z <  55): Group = Group - 10
    if (Z > 80 and Z < 87) or (Z > 112 and Z < 119): Group = Group - 24
    return(Group, Period)
   
def GetElementMultiplicity(Z, Charge):
    N_el = Z - Charge
    if   N_el in [0,2,4,10,12,18,20,36,38,54,56,86]            :return 1
    elif N_el in [1,3,5,9,11,13,17,19,31,35,37,49,53,55,81,85] :return 2
    elif N_el in [6,8,14,16,32,34,50,52,82,84]                 :return 3
    elif N_el in [7,15,33,51,83]                               :return 4

def GetElementCoreValence(Z):    # have to be extended 
    if Z in [0,1,2]                          : return ([],['1S'])
    elif Z in [3,4]                          : return (['1S'],['2S'])
    elif Z in [5,6,7,8,9,10]                 : return (['1S'],['2S','2P'])
    elif Z in [11,12]                        : return (['1S','2S','2P'],['3S'])
    elif Z in [13,14,15,16,17,18]            : return (['1S','2S','2P'],['3S','3P'])
    elif Z in [19,20]                        : return (['1S','2S','2P','3S','3P'],['4S'])
    elif Z in [21,22,23,24,25,26,27,28,29,30]: return([],[])
    elif Z in [31,32,33,34,35,36]            : return (['1S','2S','2P','3S','3P','3D'],['4S','4P'])
    else                                     : return([],[])
   
def GetBasisSetCoreValence(Z):
    if   Z < 19                         : return ('6','31')
    elif Z in [19,20,31,32,33,34,35,36] : return ('6','6')
    else                                : return ('','')

def GetSTO(Z, BasisSet):
    STO=[]
    Core, Valence = GetBasisSetCoreValence(Z)
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
    FirstLine = '# ' + Method + '/gen gfinput\n'
    return FirstLine

def GenerateTitle(Z, Scaling_factors):
    all_factors = ''
    Element = GetElementName(Z).strip()
    Title = "\n" + Element + "\n\n"
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
def Get_Energy(FileName, CPU, Z, Charge, Method, BasisSet, Scales):
    file=open(FileName+'.gjf','w')
    file.write(GenerateInput(CPU, Z, Charge, Method, BasisSet, Scales) + '\n\n')
    file.close()
    
    # Subprocess.call('GAUSS_SCRDIR="/nqs/$USER"\n', shell=True)
    subprocess.call('g09 < '+ FileName + '.gjf > ' + FileName + '.out\n', shell=True)
    Energy = subprocess.check_output('grep "SCF Done:" ' + FileName + '.out | tail -1 | awk \'{ print $5 }\'', shell=True)
    Energy = Energy.decode('ascii').rstrip('\n')
    if Energy != "":
        EnergyNUM=float(Energy)
        return EnergyNUM
    else:
        file=open(FileName+'.gjf','w')
        file.write(GenerateInput(CPU, Z, Charge, Method, BasisSet, Scales) + '\n\n')
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

def GetGradientScales(Delta, Scales):
    Gradient_scales = []
    Sorted_Gradient_scales = []
    for i in range(len(Scales)):
        plus = Scales[:]
        plus[i] = round(Scales[i] + Delta, 15)
        minus = Scales[:]
        minus[i] = round(Scales[i] - Delta, 15)
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

def CreateE2Scales(Nr_of_scales, Delta, Scales):
    E2Scales = []
    for i in range(Nr_of_scales):
        iScales = np.zeros(Nr_of_scales).tolist()
        for j in range(Nr_of_scales):
            iScales[j] = Scales[j]
        iScales[i] = iScales[i] + 2 * Delta
        E2Scales.append(iScales)
    return(E2Scales)

def CreateEEScales(Nr_of_scales, Delta1, Delta2, Scales, Indices):
    EEScales = []
    for (i, j) in Indices:
        ijScales = np.zeros(Nr_of_scales).tolist()
        for k in range(Nr_of_scales):
            ijScales[k] = Scales[k]
        ijScales[i] = ijScales[i] + Delta1
        ijScales[j] = ijScales[j] + Delta2
        EEScales.append(ijScales)
    return(EEScales)

def GetHessianScales(Nr_of_scales, Delta, Scales, Indices):
    E2PScales = CreateE2Scales(Nr_of_scales, Delta, Scales)
    E2MScales = CreateE2Scales(Nr_of_scales, -Delta, Scales)
    EPPScales = CreateEEScales(Nr_of_scales, Delta, Delta, Scales, Indices)
    ENPScales = CreateEEScales(Nr_of_scales, -Delta, Delta, Scales, Indices)
    EPNScales = CreateEEScales(Nr_of_scales, Delta, -Delta, Scales, Indices)
    ENNScales = CreateEEScales(Nr_of_scales, -Delta, -Delta, Scales, Indices)

    Sorted_Hessian_scales = []
    Sorted_Hessian_scales.extend(E2PScales)
    Sorted_Hessian_scales.extend(E2MScales)
    Sorted_Hessian_scales.extend(EPPScales)
    Sorted_Hessian_scales.extend(ENPScales)
    Sorted_Hessian_scales.extend(EPNScales)
    Sorted_Hessian_scales.extend(ENNScales)
    return(E2PScales, E2MScales, EPPScales, ENPScales, EPNScales, ENNScales, Sorted_Hessian_scales)

def EnergyParallel(Title, CPU, Z, Charge, Method, BasisSet, sto_out, index, ElementName):
    Title = Title+'_'+ElementName.strip()+'_'+BasisSet.strip()+'_scale_'+str(index+1)
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
            Scales = arguments.Scales
        else:
            print(bcolors.FAIL,"\nSTOP: number of guess values should be ", stoLen,bcolors.ENDC)
            sys.exit()
    elif os.path.isfile(GuessFile):
            Scales=[]
            File = open(GuessFile, 'r')
            for line in File:
                Scales.append(float(line.rstrip('\n')))
            File.close()
            print("The guess values (From the File) are ", "[",''.join('%12.6f' % i for i in Scales),"]")
    else:
            Scales = [1.0] * stoLen
            print("The guess values (Default Values) are ", "[",''.join('%12.6f' % i for i in Scales),"]")
    Nr_of_scales = len(Scales)

    return(Z, ElementName, CPU, Limit, Delta, GuessFile, EnergyFileI, EnergyFileF, sto, Scales, Nr_of_scales)


def WriteScales(GuessFile, Scales):
    File = open(GuessFile,'w')
    for val in Scales:
        File.write(str(val) + '\n')
    File.close()

def GetGradientEnergies(arguments, CPU, Z, ElementName, Delta, Nr_of_scales, Sorted_Gradient_scales):
    ll=joblib.Parallel(n_jobs=arguments.ParallelProc)(joblib.delayed(EnergyParallel)('Grad',CPU,Z,arguments.Charge,arguments.Method,arguments.BasisSet,sto_out,index,ElementName)
        for index,sto_out in enumerate(Sorted_Gradient_scales))
    GradientEnergyDictionary={} 
    GradientEnergyDictionary={t[0]:round(t[1], 15) for t in ll}

    GradientList=[]
    for val in range(0, len(GradientEnergyDictionary), 2):
        GradientList.append(round((float(GradientEnergyDictionary[val]) - float(GradientEnergyDictionary[val + 1])) / (2.0 * Delta), 15))
    
    Gradient = np.transpose(np.matrix(GradientList))

    return(GradientEnergyDictionary, GradientList, Gradient)

def GetHessianEnergies(arguments, CPU, Z, ElementName, Delta, Nr_of_scales, Sorted_Hessian_scales, EPPScales, E0):
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

    HessianList = np.zeros((Nr_of_scales, Nr_of_scales)).tolist()
    
    for i in range(Nr_of_scales):
        for j in range(Nr_of_scales):
            if i == j:
                HessianList[i][i] = HessianDiagonal[i]
                continue
            elif i < j:
                HessianList[i][j] = HessianUpT[i * (Nr_of_scales - i - 1) + j - 1]
                continue
            elif i > j:
                HessianList[i][j] = HessianUpT[j * (Nr_of_scales - j - 1) + i - 1]
                continue
            else:
                print("Wrong value!")
    Hessian = np.zeros((len(HessianList), len(HessianList)))
    Hessian = np.matrix(Hessian)

    for i in range(len(HessianList)):
        for j in range(len(HessianList)):
            Hessian[i, j] = HessianList[i][j]

    return(HessianEnergyDictionary, HessianEnergies, HessianList, Hessian)

def Main():
    arguments = Arguments()
    Z, ElementName, CPU, Limit, Delta, GuessFile, EnergyFileI, EnergyFileF, sto, Scales, Nr_of_scales = Initiate(arguments)
    WriteScales(GuessFile, Scales)

# Initial values:
    Ctrl   =1000.0
    RGS    =100.0
    E0     =0.0
    Rho    =0.0
    skip   =0
    counter=0

# Calculating the initial energy:
    if E0 == 0.0:
        E0 = Get_Energy(EnergyFileI,CPU,Z,arguments.Charge,arguments.Method,arguments.BasisSet,Scales)
        DEnergy=np.absolute(E0)
        print("Eo =",'% 12.6f' % E0, " --  Initial DEnergy =", '% 12.6f' % DEnergy)

# Generate Gaussian input files and run them, then calculate Hessian:
    Indices, Diagonal = CreateIndices(Nr_of_scales)
    E2PScales, E2MScales, EPPScales, ENPScales, EPNScales, ENNScales, Sorted_Hessian_scales = GetHessianScales(Nr_of_scales, Delta, Scales, Indices)
    HessianEnergyDictionary, HessianEnergies, HessianList, Hessian = GetHessianEnergies(arguments, CPU, Z, ElementName, Delta, Nr_of_scales, Sorted_Hessian_scales, EPPScales, E0)

#    for i in range(1,87):
#       Group, Period = GetElementGroupPeriod(i) 
#       print("Z: ",i, "group: ",Group, "Period",Period)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> while loop <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    while RGS > Limit:
        counter += 1 

# Generating scales for the gradient:
        Indices, Diagonal = CreateIndices(Nr_of_scales)
        Gradient_scales, Sorted_Gradient_scales = GetGradientScales(Delta, Scales)
# Generate Gaussian input files and run them, then calculate Gradient:
        GradientEnergyDictionary, GradientList, Gradient = GetGradientEnergies(arguments, CPU, Z, ElementName, Delta, Nr_of_scales, Sorted_Gradient_scales)

# Update Hessian -------------------------------------
        if skip == 0: G0 = Gradient
        if skip == 1:
            dG         = Gradient - G0
            zd         = dG - np.dot(Hessian, dX)
            norm_dX    = np.linalg.norm(dX)
            norm_dG    = np.linalg.norm(dG)
            norm_zd    = np.linalg.norm(zd)
            condition1 = (np.dot(np.transpose(zd), dX)).tolist()[0][0] / (norm_zd * norm_dX)
            condition2 = (np.dot(np.transpose(dG), dX)).tolist()[0][0] / (norm_dG * norm_dX)
# 1) Murtagh-Sargent, symmetric rank one (SR1) update:
            if   condition1 < -0.1:
                print('-----------1) Murtagh-Sargent, symmetric rank one (SR1) update')
                #print('z_zT', np.dot(zd, np.transpose(zd)))
                #print('zT_dX', np.dot(np.transpose(zd), dX))
                Hessian = Hessian + ((np.dot(zd, np.transpose(zd))) / (np.dot(np.transpose(zd), dX)))
# 2) Broyden-Fletcher-Goldfarb-Shanno (BFGS) update:
            elif condition2 >  0.1:
                print('-----------2) Broyden-Fletcher-Goldfarb-Shanno (BFGS) update')
                #print('OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO')
                #print('dG_dGT', (np.dot(dG, np.transpose(dG))))
                #print('dGT_dX', (np.dot(np.transpose(dG), dX)))
                dGdGt = ((np.dot(dG, np.transpose(dG))) / (np.dot(np.transpose(dG), dX)))
                HxxtH  = (np.dot(np.dot(np.dot(Hessian, dX), np.transpose(dX)), Hessian)) /  np.dot(np.dot(np.transpose(dX), Hessian), dX)
                #print('change', dGdGt - HxxtH)
                Hessian = Hessian + (dGdGt - HxxtH)
# 3) Powell-symmetric-Broyden (PSB) update:
            else:
                print('-----------3) Powell-symmetric-Broyden (PSB) update')
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

# Calculate Hessian eigenvalues:
        eW, eV = np.linalg.eig(Hessian)
        ew = min(eW)

# Print to the output:
        print()
        print(bcolors.OKBLUE,"Step", counter, "---------------",bcolors.ENDC)
        print()
        print("   Eigenvalues of Hes:","[",', '.join('%8.6f' % i for i in eW),"]")
        print("   Minimum eigenvalue:", '%8.6f' % ew)
        print()

# Control the criteria of the trust region:
        if DEnergy <= 0.0750 and (Rho < 1.250 and  Rho > 0.90): Ctrl = Ctrl/10.0
        if Z > 10 and DEnergy <= 1.0 and (Rho < 1.250 and  Rho > 0.90): Ctrl = Ctrl/10.0
        if Ctrl > 10.0 and DEnergy < 1.0e-4 and RGS < 0.01: Ctrl = 1.0
        if Ctrl < 1.0 and (Rho > 1.250 or (Rho < 0.90 and Rho > 0.250)): Ctrl = 1.0
        if Ctrl < 1.0 and (Rho < 1.250 and Rho > 0.90): 
            Ctrl = 0.250
            if RGS < 0.0050 and DEnergy < 1.0e-3: 
                Ctrl = Ctrl/2.0
                if RGS < 0.0009: Ctrl = Ctrl/20.0
        if Rho < 0.250: Ctrl = Ctrl * 10.0
        if Ctrl > 1000.0: Ctrl = 1000.0

# Make dx = -inv(H - λI)g if, at least, one of the Hesian eigen values < zero, else dx = -inv(H)g: 
        if ew < 9.0e-3:
            Lambda = (- ew) + Ctrl
            ShiftedHessian = Hessian + Lambda * np.identity(len(Gradient))
            dX = -np.dot(ShiftedHessian.I, Gradient)
            dXList = np.transpose(dX).tolist()[0]
            LastScales = Scales.copy()
            Scales=[float(i) + float(j) for i, j in zip(Scales, dXList)]

            #'''
            if min(Scales) < 0:
                # Generate Gaussian input files and run them, then calculate Hessian:
                Indices, Diagonal = CreateIndices(Nr_of_scales)
                E2PScales, E2MScales, EPPScales, ENPScales, EPNScales, ENNScales, Sorted_Hessian_scales = GetHessianScales(Nr_of_scales, Delta, LastScales, Indices)
                HessianEnergyDictionary, HessianEnergies, HessianList, Hessian = GetHessianEnergies(arguments, CPU, Z, ElementName, Delta, Nr_of_scales, Sorted_Hessian_scales, EPPScales, E0)

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

            NEnergy=Get_Energy(EnergyFileF,CPU,Z,arguments.Charge,arguments.Method,arguments.BasisSet,Scales)
            DEnergy=E0-NEnergy
            if -(np.dot(np.transpose(Gradient), dX) + 0.5 * np.dot(np.dot(np.transpose(dX), Hessian), dX)).tolist()[0][0] == 0.0:
                Rho = 0.0
            else:
                Rho = (DEnergy / -(np.dot(np.transpose(Gradient), dX) + 0.5 * np.dot(np.dot(np.transpose(dX), Hessian), dX)).tolist()[0][0])
            print(bcolors.OKGREEN, "  * Ctrl:",Ctrl, bcolors.ENDC)
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
                Indices, Diagonal = CreateIndices(Nr_of_scales)
                E2PScales, E2MScales, EPPScales, ENPScales, EPNScales, ENNScales, Sorted_Hessian_scales = GetHessianScales(Nr_of_scales, Delta, LastScales, Indices)
                HessianEnergyDictionary, HessianEnergies, HessianList, Hessian = GetHessianEnergies(arguments, CPU, Z, ElementName, Delta, Nr_of_scales, Sorted_Hessian_scales, EPPScales, E0)

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

            NEnergy=Get_Energy(EnergyFileF,CPU,Z,arguments.Charge,arguments.Method,arguments.BasisSet,Scales)
            DEnergy=E0-NEnergy
            if -(np.dot(np.transpose(Gradient), dX) + 0.5 * np.dot(np.dot(np.transpose(dX), Hessian), dX)).tolist()[0][0] == 0.0:
                Rho = 0.0
            else:
                Rho = (DEnergy / -(np.dot(np.transpose(Gradient), dX) + 0.5 * np.dot(np.dot(np.transpose(dX), Hessian), dX)).tolist()[0][0])
            print("   lambda:    ",'% 12.6f' % float(ew))
            print("   Rho:       ",'% 12.6f' % float(Rho))

# calculate the root for the sum of squares of gradient components:  
        Sum_G_Sq = 0.0
        for i in range(len(GradientList)):
            Sum_G_Sq += (GradientList[i] * GradientList[i]) 
        RGS = math.sqrt(Sum_G_Sq)

# Print to the output:
        print(bcolors.BOLD,"  RGS:       ",'% 12.6f' % float(RGS), bcolors.ENDC)      
        ColoRR=bcolors.OKGREEN
        if DEnergy < 0.0: ColoRR=bcolors.FAIL
        print()   
        print(ColoRR,"  New Scale:","[",' '.join('%10.6f' % i for i in Scales),"]") #"   ",np.array(Scales))
        print("   Initial E:","", '% 12.6f' % float(E0))
        print("   Final E  :","", '% 12.6f' % float(NEnergy))
        print("   Delta E  :","", '% 12.6f' % float(DEnergy), bcolors.ENDC)
        print()   

# Saving the new E0, G0, and new scale values:
        WriteScales(GuessFile, Scales)
        E0 = NEnergy
        G0 = Gradient
        skip = 1

# Print to the output if the convergence criteria met, and exit: 
        if (RGS <= Limit):
            print(bcolors.OKGREEN,"STOP:", bcolors.ENDC)
            print("The gradient is", "[",', '.join('%8.6f' % i for i in GradientList),"]", "and RGS =",'% 8.6f' % float(RGS))      
            print(bcolors.OKBLUE,"\n                        -- Optimization Terminated Normally --")
            print()
# ------------------------------------------------------ Exit ---------------------------------------------------------------
if __name__ == "__main__":
    Main()
sys.exit(0) 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
