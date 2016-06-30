#!/usr/bin/python
import sys
import argparse
import sqlite3

parser = argparse.ArgumentParser()

parser.add_argument('-o', '--output', nargs='?', type=argparse.FileType('w'), default=sys.stdout, help='file name to output to (string)')
parser.add_argument('-i', '--input', nargs='?', type=argparse.FileType('r'), default=sys.stdout, help='file name to input from (string)')

args = parser.parse_args()



method = 'uhf'
atom = 'O'
charge = '0'
multiplicity = '3'
scaling_factors = [1.00, 1.00, 1.00, 1.00, 1.00]
list_basis_set = ['     0\nSTO 1S 6 ', 'STO 2S 3 ', 'STO 2P 3 ', 'STO 2S 1 ', 'STO 2P 1 ', '****\n']


def gen_first_line(method):
	first_line = '# opt freq ' + method + '/gen gfprint\n'
	return first_line

def gen_title(atom, scaling_factors):
	t_scaling_factor = ''
	for t in range(len(scaling_factors)):
		t_scaling_factor = t_scaling_factor + str(scaling_factors[t]) + '_'

	title = "\n" + atom + "_" + t_scaling_factor + "\n\n"
	return title

def gen_charge_multiplicity(charge, multiplicity):
	charge_multiplicity = "{} {}\n".format(charge, multiplicity)
	return charge_multiplicity

def gen_z_matrix(atom):
	z_matrix = atom + "\n\n"
	return z_matrix

def gen_cartesian_coord():
	cart_coord = "\n"
	return cart_coord

def gen_basis_sets(atom, list_basis_set, scaling_factors):
	basis_sets = str(atom) + list_basis_set[0] + str(scaling_factors[0]) + '\n' + list_basis_set[1] + str(scaling_factors[1]) + '\n' + list_basis_set[2] + str(scaling_factors[2]) + '\n' + list_basis_set[3] 
	basis_sets = basis_sets	+ str(scaling_factors[3]) + '\n' + list_basis_set[4] + str(scaling_factors[4]) + '\n' + list_basis_set[5] +'\n' 
	return basis_sets

first_line = gen_first_line(method)
title = gen_title(atom, scaling_factors)
charge_multiplicity = gen_charge_multiplicity(charge, multiplicity)
z_matrix = gen_z_matrix(atom)
cart_coord = gen_cartesian_coord()
basis_sets = gen_basis_sets(atom, list_basis_set, scaling_factors)

args.output.write(first_line + title + charge_multiplicity + z_matrix + basis_sets)

