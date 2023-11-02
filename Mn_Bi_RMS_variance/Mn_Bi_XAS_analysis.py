import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import fsolve
from math import floor, ceil
import numpy as np
import os

def plotter(x, y, title, x_label, y_label, settings, x_axis_increment='', y_axis_increment='', x_tick_positions = [], y_tick_positions = []):
	settings_dict = {
		"SI": {
			"Figsize": (9, 8),
			"x_labels_font_size": 24,
			"x_label_pad": 10,
			"x_tick_labels_font_size": 18,
			"x_tick_labels_pad": 10,
			"y_labels_font_size": 24,
			"y_label_pad": -25,
			"y_tick_labels_font_size": 18,
			"y_tick_labels_pad": 5,
			"title_font_size": 40,
			"title_pad": 15,
			"linewidth": 6,
			"linestyle": '-',
			"marker": '',
			"x_tick_length": 8,
			"y_tick_length": 8,
			"border_thickness": 3,
			"move_bottom": 0.12,
			"move_top": 0.9,
			"move_left": 0.18,
			"move_right": 0.95,
			"show_grid": False,
			"colors": [[7/255, 107/255, 255/255], [208/255, 170/255, 96/255]]
		}
	}

	settings = settings_dict.get(settings, {})

	if x_axis_increment == '':
		x_axis_increment, x_decimal_place = calculate_axis_increment([x], x_tick_positions)

	if y_axis_increment == '':
		y_axis_increment, y_decimal_place = calculate_axis_increment(y, y_tick_positions)

	if x_decimal_place:
		x_decimal_place = calculate_decimal_place(x_axis_increment)

	if y_decimal_place:
		y_decimal_place = calculate_decimal_place(y_axis_increment)

	x_tick_positions = x_tick_positions or generate_tick_positions([x], x_axis_increment)

	y_tick_positions = y_tick_positions or generate_tick_positions(y, y_axis_increment)

	x_tick_labels = generate_tick_labels(x_tick_positions, x_axis_increment, x_decimal_place)

	y_tick_labels = generate_tick_labels(y_tick_positions, y_axis_increment, y_decimal_place)

	fig, ax = plt.subplots(figsize=settings.get("Figsize", (9, 8)))
	for i in range(len(y)):
		ax.plot(x, y[i], marker=settings.get("marker", ''), linestyle=settings.get("linestyle", '-'), linewidth=settings.get("linewidth", 6), color=settings.get("colors", [7/255, 107/255, 255/255])[i])
	set_plot_labels(ax, x_label, y_label, title, settings)
	set_plot_ticks(ax, x_tick_positions, y_tick_positions, x_tick_labels, y_tick_labels, settings)
	set_plot_limits(ax, x_tick_positions[0], x_tick_positions[-1], y_tick_positions[0], y_tick_positions[-1])
	set_plot_tick_params(ax, settings)
	set_plot_spines(ax, settings)

	fig.subplots_adjust(bottom = settings.get("move_bottom", 0.12), top = settings.get("move_top", 0.9), left = settings.get("move_left", 0.18), right = settings.get("move_right", 0.95))
	plt.grid(settings.get("show_grid", False))
	return fig, ax

def find_min_and_max(x):
	max_data = x[0][0]
	min_data = x[0][0]
	for data_list in x:
		if max(data_list) > max_data:
			max_data = max(data_list)
		if min(data_list) < min_data:
			min_data = min(data_list)
	return min_data, max_data

def calculate_axis_increment(x, tick_positions):
	if tick_positions == []:
		min_data, max_data = find_min_and_max(x)
		axis_increment = (max_data - min_data)/6
	else:
		axis_increment = tick_positions[1] - tick_positions[0]
	decimal_place = calculate_decimal_place(axis_increment)
	axis_increment = round(axis_increment*10**(-decimal_place))*10**decimal_place
	return axis_increment, decimal_place

def calculate_decimal_place(axis_increment):
	if "." in str(axis_increment) and "e" not in str(axis_increment):
		decimal_place = 0
		for letter in str(axis_increment).split(".")[1]:
			decimal_place = decimal_place + 1
			if letter != "0":
				break
		decimal_place = -decimal_place
	elif "e" in str(axis_increment):
		decimal_place = float(str(axis_increment).split("e")[1])
	else:
		decimal_place = len(str(axis_increment).split(".")[0]) - 1
	return decimal_place

def generate_tick_positions(x, axis_increment):
	min_data, max_data = find_min_and_max(x)
	tick_positions = range(floor(min_data/axis_increment), ceil(max_data/axis_increment) + 1)
	return [position * axis_increment for position in tick_positions]

def generate_tick_labels(tick_positions, axis_increment, decimal_place):
	if decimal_place > 2 or decimal_place < -2:
		tick_labels = [str(position * 10 ** (-decimal_place)).split(".")[0] for position in tick_positions]
	else:
		tick_labels = [f"{position:.{abs(decimal_place)}f}" if "." in str(axis_increment) else str(position) for position in tick_positions]
	if decimal_place > 2 or decimal_place < -2:
		decimal_place = int(decimal_place)
		tick_labels[-1] += f' x 10$^{{\\mathbf{{ {decimal_place} }}}}$'
	return tick_labels

def set_plot_labels(ax, x_label, y_label, title, settings):
	ax.set_xlabel(x_label, fontweight='bold', fontsize=settings.get("x_labels_font_size", 24), fontfamily='Arial', labelpad=settings.get("x_label_pad", 10))
	ax.set_ylabel(y_label, fontweight='bold', fontsize=settings.get("y_labels_font_size", 24), fontfamily='Arial', labelpad=settings.get("y_label_pad", -25))
	ax.set_title(title, fontweight='bold', fontsize=settings.get("title_font_size", 40), fontfamily='Arial', pad=settings.get("title_pad", 15))

def set_plot_ticks(ax, x_tick_positions, y_tick_positions, x_tick_labels, y_tick_labels, settings):
	ax.set_xticks(x_tick_positions)
	ax.set_yticks(y_tick_positions)
	ax.set_xticklabels(x_tick_labels, fontname='Arial', fontweight='bold', fontsize=settings.get("x_tick_labels_font_size", 18))
	ax.set_yticklabels(y_tick_labels, fontname='Arial', fontweight='bold', fontsize=settings.get("y_tick_labels_font_size", 18))

def set_plot_limits(ax, x_min, x_max, y_min, y_max):
	ax.set_xlim(x_min, x_max)
	ax.set_ylim(y_min, y_max)

def set_plot_tick_params(ax, settings):
	ax.tick_params(axis='x', length=settings.get("x_tick_length", 8), width=settings.get("border_thickness", 3), pad=settings.get("x_tick_labels_pad", 10))
	ax.tick_params(axis='y', length=settings.get("y_tick_length", 8), width=settings.get("border_thickness", 3), pad=settings.get("y_tick_labels_pad", 5))

def set_plot_spines(ax, settings):
	for spine in ['top', 'bottom', 'left', 'right']:
		ax.spines[spine].set_linewidth(settings.get("border_thickness", 3))

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

def plot_XANES(energies, e0, XANES, XANES_function, second_derivative_function, pre_edge_line, post_edge_line, filename, XMCD):
	pre_edge_plot_domain = np.linspace(energies[-1], e0[0], 1000)
	post_edge_plot_domain = np.linspace(e0[0], energies[0], 1000)
	plt.plot(energies, XANES)
	plt.plot(energies, XANES_function(energies))
	plt.plot(energies, second_derivative_function(energies)/100000)
	plt.axvline(x = e0)
	plt.plot(pre_edge_plot_domain, pre_edge_line[0]*pre_edge_plot_domain + pre_edge_line[1])
	plt.plot(post_edge_plot_domain, post_edge_line[0]*post_edge_plot_domain + post_edge_line[1])
	plt.xlabel('Energy (keV)')
	plt.ylabel('Mu(E)')
	plt.title("XANES " + filename.split(".")[0])
	plt.savefig(filename.split(".")[0] + ".png", dpi=400)
	plt.clf()
	plt.plot(energies, XMCD)
	plt.xlabel('Energy (keV)')
	plt.ylabel('XMCD')
	plt.title("XMCD " + filename.split(".")[0])
	plt.savefig(filename.split(".")[0] + "_XMCD.png", dpi=400)
	plt.clf()

def get_rms_and_std_function(XMCD_function_list, evaluation_domain):
	std_list = []
	rms_list = []
	for energy in evaluation_domain:
		XMCD_list = []
		for XMCD_function in XMCD_function_list:
			XMCD_list.append(XMCD_function(energy))
		XMCD_array = np.array(XMCD_list)
		std_list.append(np.std(XMCD_array))
		rms_list.append(np.sqrt(np.mean(XMCD_array**2)))
	rms_function = interpolate.interp1d(evaluation_domain, rms_list, kind = 'linear')
	std_function = interpolate.interp1d(evaluation_domain, std_list, kind = 'linear')
	return rms_function, std_function

def plot_rms_and_std(rms_function, std_function, evaluation_domain, folder_name):
	fig1, ax1 = plotter(evaluation_domain[:-100], [rms_function(evaluation_domain[:-100])], "RMS " + folder_name.split("/")[-1].split("_")[0] + " " + folder_name.split("/")[-1].split("_")[1], "Energy (keV)", "RMS XMCD", "SI", y_tick_positions = [0.0, 0.0004, 0.0008, 0.0012000000000000001, 0.0016, 0.002, 0.0024000000000000002, 0.0028])
	plt.savefig(folder_name.split("/")[-1] + "_rms.png", dpi=400)
	plt.close()
	fig1, ax1 = plotter(evaluation_domain[:-100], [std_function(evaluation_domain[:-100])], "Standard Deviation " + folder_name.split("/")[-1].split("_")[0] + " " + folder_name.split("/")[-1].split("_")[1], "Energy (keV)", "XMCD standard deviation", "SI", y_tick_positions = [0.0, 0.0004, 0.0008, 0.0012000000000000001, 0.0016, 0.002, 0.0024000000000000002, 0.0028])
	plt.savefig(folder_name.split("/")[-1] + "_std.png", dpi=400)
	plt.close()

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







folder_list = []
for folder in os.scandir():
	if folder.is_dir():
		folder_list.append(folder.path)

for folder_name in folder_list:
	pos_folder = folder_name + "/pos/"
	neg_folder = folder_name + "/neg/"
	XMCD_function_list = []
	start_energy = 0
	end_energy = 1000000000
	for filename in os.listdir(pos_folder):
		if filename[-4:] == ".dat":
			print(filename)

			#interpolate the XANES data into a function
			energies, XANES, XMCD = load_data_into_lists(pos_folder + filename)

			#Do the XANES data workup
			e0, XANES_function, second_derivative_function, pre_edge_line, post_edge_line, normalization_factor = XANES_data_workup(energies, XANES, 13.427)

			#plot all the relevant XANES details for each plot
			plot_XANES(energies, e0, XANES, XANES_function, second_derivative_function, pre_edge_line, post_edge_line, filename, XMCD)

			#save energies if relevant
			if start_energy < energies[-1]:
				start_energy = energies[-1]

			if end_energy > energies[0]:
				end_energy = energies[0]

			#Scale the XMCD data by the normalization_factor
			XMCD = XMCD / normalization_factor
			XMCD_function = interpolate.interp1d(energies, XMCD, kind = 'linear')
			XMCD_function_list.append(XMCD_function)

	for filename in os.listdir(neg_folder):
		if filename[-4:] == ".dat":
			print(filename)

			#interpolate the XANES data into a function
			energies, XANES, XMCD = load_data_into_lists(neg_folder + filename)

			#Do the XANES data workup
			e0, XANES_function, second_derivative_function, pre_edge_line, post_edge_line, normalization_factor = XANES_data_workup(energies, XANES, 13.427)

			#plot all the relevant XANES details for each plot
			plot_XANES(energies, e0, XANES, XANES_function, second_derivative_function, pre_edge_line, post_edge_line, filename, XMCD)

			#save energies if relevant
			if start_energy < energies[-1]:
				start_energy = energies[-1]

			if end_energy > energies[0]:
				end_energy = energies[0]

			#Scale the XMCD data by the normalization_factor and multiply by negative 1 for the negative field
			XMCD = XMCD / normalization_factor * -1
			XMCD_function = interpolate.interp1d(energies, XMCD, kind = 'linear')
			XMCD_function_list.append(XMCD_function)


	#Evaluate the rms and std of the XMCD for a given absolute field value
	evaluation_domain = np.linspace(start_energy, end_energy, 1000)
	rms_function, std_function = get_rms_and_std_function(XMCD_function_list, evaluation_domain)


	#plot the rms and std for the given absolute field value
	plot_rms_and_std(rms_function, std_function, evaluation_domain, folder_name)