from scipy import interpolate
from scipy.optimize import fsolve
from scipy.integrate import quad
from math import floor, ceil, sqrt
import numpy as np
import os

def data_loader(filename):
	data_file = open(filename, 'r')
	file_lines = data_file.readlines()
	data_file.close()
	data = []
	for line in file_lines:
		data.append(line.split(","))
	data = np.transpose(data)
	return data

def load_data_into_lists(filename):
	data_file = open(filename, 'r')
	file_lines = data_file.readlines()
	data_file.close()
	energies = []
	XANES = []
	XMCD = []
	for line in file_lines[3:]:
		energies.append(float(line.split()[0]))
		XANES.append(float(line.split()[1]))
		XMCD.append(float(line.split()[2]))
	return energies, XANES, XMCD

def take_step(folder_name):
	return float(folder_name.split("/")[1].split("_")[0])

def get_edge_position(energies, XANES, edge_estimate):
	first_derivative = np.gradient(XANES, energies)
	second_derivative = np.gradient(first_derivative, energies)
	second_derivative_function = interpolate.interp1d(energies, second_derivative, kind = 'linear')
	return fsolve(second_derivative_function, edge_estimate), second_derivative_function

def get_pre_edge_line(energies, XANES_function, e0, pre_edge_fit_range):
	pre_data_domain = (e0[0] - energies[-1])*pre_edge_fit_range
	pre_edge_domain = np.linspace(energies[-1], energies[-1] + pre_data_domain, 1000)
	pre_edge_line = np.polyfit(pre_edge_domain, XANES_function(pre_edge_domain), 1)
	return pre_edge_line

def get_post_edge_line(energies, XANES_function, e0, post_edge_fit_range):
	post_data_domain = (energies[0] - e0[0])*post_edge_fit_range
	post_edge_domain = np.linspace(energies[0] - post_data_domain, energies[0], 1000)
	post_edge_line = np.polyfit(post_edge_domain, XANES_function(post_edge_domain), 1)
	return post_edge_line

def calculate_normalization_factor(pre_edge_line, post_edge_line, e0):
	return (post_edge_line[0]-pre_edge_line[0])*e0 + post_edge_line[1] - pre_edge_line[1]

def XANES_data_workup(energies, XANES, edge_approximation, pre_edge_fit_range = 1/2, post_edge_fit_range = 2/3):
	#interpolate the XANES data into a function
	XANES_function = interpolate.interp1d(energies, XANES, kind = 'linear')
	#solve for the absorption edge position, e0, and get the second_derivative_function
	e0, second_derivative_function = get_edge_position(energies, XANES, edge_approximation)
	#fit the pre-edge line to the first fraction of the data and the post-edge line to the last fraction of data
	pre_edge_line = get_pre_edge_line(energies, XANES_function, e0, pre_edge_fit_range)
	post_edge_line = get_post_edge_line(energies, XANES_function, e0, post_edge_fit_range)
	#calculate the normalization factor
	normalization_factor = calculate_normalization_factor(pre_edge_line, post_edge_line, e0)
	return e0, XANES_function, second_derivative_function, pre_edge_line, post_edge_line, normalization_factor

def XMCD_statistics_calculator(XMCD_function_list, integration_range, background_function=None):
	XMCD_integration = []
	XMCD_statistics = []
	if background_function != None:
		for i in range(len(XMCD_function_list)):
			background_subtraction_domain = np.linspace(integration_range[0] - 0.01, integration_range[1] + 0.01, 1000)
			XMCD_function_list[i] = interpolate.interp1d(background_subtraction_domain, XMCD_function_list[i](background_subtraction_domain) - background_function(background_subtraction_domain), kind = 'linear')
	for XMCD_function in XMCD_function_list:
		XMCD_integration.append(quad(XMCD_function, integration_range[0], integration_range[1])[0] * 1000)
	XMCD_statistics.append(np.mean(XMCD_integration))
	XMCD_statistics.append(np.std(XMCD_integration))
	return XMCD_statistics, XMCD_integration





folder_list = []
for folder in os.scandir():
	if folder.is_dir():
		folder_list.append(folder.path)

statistics_file = open("XMCD_statistics.csv", 'w')
statistics_file.write("Conditions,Average_XMCD,XMCD_standard_deviation\n")
background_subtracted_statistics_file = open("XMCD_statistics_background_subtracted.csv", 'w')
background_subtracted_statistics_file.write("Conditions,Average_XMCD,XMCD_standard_deviation\n")
folder_list.sort(key=take_step)

for folder_name in folder_list:
	XMCD_function_list = []
	background = []
	for filename in os.listdir(folder_name):
		if filename[-4:] == ".dat":
			print(filename)

			#interpolate the XANES data into a function
			energies, XANES, XMCD = load_data_into_lists(folder_name + "/" + filename)

			#Do the XANES data workup
			e0, XANES_function, second_derivative_function, pre_edge_line, post_edge_line, normalization_factor = XANES_data_workup(energies, XANES, 13.427)

			#Scale the XMCD data by the normalization_factor
			XMCD = XMCD / normalization_factor
			XMCD_function = interpolate.interp1d(energies, XMCD, kind = 'linear')
			XMCD_function_list.append(XMCD_function)

		if filename[-4:] == ".csv":
			background = data_loader(folder_name + "/" + filename)
			background_floats = []
			background_floats.append([float(i) for i in background[0][1:]])
			background_floats.append([float(i) for i in background[1][1:]])
			background_function = interpolate.interp1d(background_floats[0], background_floats[1], kind = 'linear')



	#Evaluate the integration statistics of the XMCD for each field value
#	XMCD_integration_list = XMCD_integrator(XMCD_function_list)
	XMCD_statistics, XMCD_integration = XMCD_statistics_calculator(XMCD_function_list, [13.427, 13.441])

	#plot the rms and variance for the given absolute field value
	statistics_file.write(folder_name + ",")
	for value in XMCD_statistics:
		statistics_file.write(str(value) + ",")
	statistics_file.write('\n')


	if len(background) != 0:
		XMCD_statistics, XMCD_integration = XMCD_statistics_calculator(XMCD_function_list, [13.427, 13.441], background_function=background_function)
		background_subtracted_statistics_file.write(folder_name + ",")
		for value in XMCD_statistics:
			background_subtracted_statistics_file.write(str(value) + ",")
		background_subtracted_statistics_file.write('\n')


		

statistics_file.close()