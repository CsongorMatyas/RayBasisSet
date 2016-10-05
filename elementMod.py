from sys import exit
import subprocess

def GetElemSym(z):
    if z<1:
        print('Error: the atomic number is less than one (Z<1)\nProgram Exit ):')
        exit(0)
    element=['H ',                                                                       'He', 
    'Li','Be',                                                  'B ','C ','N ','O ','F ','Ne', 
    'Na','Mg',                                                  'Al','Si','P ','S ','Cl','Ar', 
    'K ','Ca','Sc','Ti','V ','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr', 
    'Rb','Sr','Y ','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I ','Xe', 
    'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf', 
    'Ta','W ','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U ']
    return element[z-1]


def GetElemNam(z):
    if z<1:
        print('Error: the atomic number is less than one (Z<1)\nProgram Exit ):')
        exit(0)
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
    return element[z-1]

def GetElemGrPe(z):
    ElemOrd=[2,8,8,18,18,36,36]
    period=1
    group=0
    for eleord in ElemOrd:
        if group < z-eleord:
            group+=eleord
            period+=1
    group=z-group
    return group,period

   
def GetElemMult(z,charge):
    g,p=GetElemGrPe(z)
    g=g-charge
    if g in [1,3,7,17]:
        return 2
    elif g in [2,8,18]:
        return 1
    elif g in [4,6,16]:
        return 3
    elif g in [5,15]:
        return 4

def GetElemCorVal(z):
    ElemAto=[['1S'],['2S','2P'],['3S','3P'],['4S','3D','4P'],['5S','4D','5P'],['6S','4F','5D','6P'],['7S','5F','6D','7P']]
    g,p=GetElemGrPe(z)
    AtomicCore=[]
    for sto in range(0,p-1):
        for psto in ElemAto[sto]:
            AtomicCore.append(psto)
    AtomicValance=[]
    for vsto in ElemAto[p-1]:
        AtomicValance.append(vsto)
    return AtomicCore,AtomicValance

def GetBasisCorVal(BasisSet):
    core=BasisSet[:BasisSet.find('-')]
    valance=BasisSet[BasisSet.find('-')+1:BasisSet.find('G')]  #<<<<<<<<<<Needing to use RE :(
    return core, valance

def GetSTO(Z,basis):
    STO=[]
    core,valance=GetBasisCorVal(basis)
    AtomicCore,AtomicValance=GetElemCorVal(Z)
    #count=0
    for corevalue in core:
        for atomiccore in AtomicCore:
            STO.append('STO '+ atomiccore+ ' '+str(corevalue))
            #count+=1
    for valancevalue in valance:
        for atomicvalance in AtomicValance:
            STO.append('STO '+ atomicvalance +' '+str(valancevalue))
            #count+=1
    return STO

##### Input file

def gen_first_line(method):
	first_line = '# opt freq ' + method + '/gen gfprint\n'
	return first_line

def gen_title(Z, scaling_factors):
	t_scaling_factor = ''
	atom=GetElemNam(Z).strip()
	for t in range(len(scaling_factors)):
		t_scaling_factor = t_scaling_factor + str(scaling_factors[t]) + '_'

	title = "\n" + atom + "_" + t_scaling_factor + "\n\n"
	return title

def gen_charge_multiplicity(Z, charge):
	charge_multiplicity = "{} {}\n".format(charge, GetElemMult(Z,charge))
	return charge_multiplicity

def gen_z_matrix(Z):
	z_matrix = GetElemSym(Z).strip() + "\n\n"
	return z_matrix

def gen_cartesian_coord(Z):
	cart_coord = GetElemSym(Z).strip()+' 0\n'
	return cart_coord
	
def returnInput(cpu, Z, charge, method, basis, scaling_factors):
    outputtext='%NPROCS='+str(cpu)+'\n'+gen_first_line(method)
    outputtext+=gen_title(Z, scaling_factors)
    outputtext+=gen_charge_multiplicity(Z, charge)
    outputtext+=gen_z_matrix(Z)
    outputtext+=gen_cartesian_coord(Z)
    sto=GetSTO(Z,basis)
    for index,sto_out in enumerate(sto):
        outputtext+=sto_out+' '+str(scaling_factors[index])+'\n'
    outputtext+='****\n\n'
    return outputtext
    
def Get_Energy(fileName,cpu,Z,charge,theory,basis,guessScale):
    # calculate the energy
    file=open(fileName+'.gjf','w')
    file.write(returnInput(cpu,Z,charge,theory,basis,guessScale)+'\n\n')
    file.close()
    subprocess.call('g09 < '+fileName+'.gjf > '+fileName+'.out\n',shell=True)
    Energy=subprocess.check_output('grep "SCF Done:" '+fileName+'.out | tail -1|awk \'{ print $5 }\'', shell=True)
    return float(Energy.decode('ascii').rstrip('\n'))
    
# Generate .sh file to run it on the Cluster (serial)
'''
file=open(title+'.sh','w')
file.write('#$ -S /bin/csh\n')
file.write('#$ -cwd\n')
file.write('#$ -j y\n')
file.write('#$ -l h_rt=1:00:00\n')
#file.write('#$ -l h_vmem=7500M\n')
#file.write('#$ -pe openmp 4\n')
file.write('\n\n')
file.write('setenv GAUSS_SCRDIR /nqs/$USER\n')
file.write('module load gaussian\n')
for ijob in job:
    file.write('g09 < '+ijob+'.gjf > '+ijob+'.out\n')
    file.write('grep "SCF Done:  E(" '+ijob+'.out | tail -1|awk \'{ print $5 }\' >> "'+GradFile+'"\n')
file.write('\n\n')
file.write('# bash to extract energy\n')
file.close()
'''

#HE-2.85516042615
