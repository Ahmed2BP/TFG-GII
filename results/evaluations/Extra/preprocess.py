# Script para preprocesar un fichero de evaluación de un modelo/agente

import csv
import sys
import math


nombreArchivo = ""

if(len(sys.argv) == 1):
    print("Introduzca el nombre del archivo que desea preprocesar:\n")
    input(nombreArchivo)

else:
    nombreArchivo = sys.argv[1]

with open(nombreArchivo) as csv_file:
    next(csv_file)
    csv_reader = csv.reader(csv_file, delimiter=',')
    num_evaluations = 0
    mean_power_consumption = 0
    mean_comfort_violation = 0
    mean_reward = 0

    for row in csv_reader:
        mean_reward += float(row[2])
        mean_power_consumption += float(row[4])
        mean_comfort_violation += float(row[9])
        num_evaluations += 1

    mean_comfort_violation = mean_comfort_violation / num_evaluations
    mean_power_consumption = mean_power_consumption / num_evaluations
    mean_reward = mean_reward / num_evaluations

    dev_comfort_violation = 0
    dev_power_consumption = 0
    dev_reward = 0


with open(nombreArchivo) as csv_file:
    next(csv_file)
    csv_reader = csv.reader(csv_file, delimiter=',')

    for row in csv_reader:
        dev_reward += pow(float(row[2]) - mean_reward, 2)
        dev_power_consumption += pow(float(row[4]) - mean_power_consumption, 2)
        dev_comfort_violation += pow(float(row[9]) - mean_comfort_violation, 2)

    dev_comfort_violation = math.sqrt(dev_comfort_violation / num_evaluations)
    dev_power_consumption = math.sqrt(dev_power_consumption / num_evaluations)
    dev_reward = math.sqrt(dev_reward / num_evaluations)

    print("\n\n" + nombreArchivo.split('-')[12][:-5] + " (evaluado con OccupRew)")
    print('\nmean_comfort_violation = ' + str(round(mean_comfort_violation, 5)) + "\t\ttypical dev = " + str(dev_comfort_violation))
    print('mean_power_consumption = ' + str(round(mean_power_consumption, 5)) + "\t\ttypical dev = " + str(dev_power_consumption))
    print('mean_reward = ' + str(round(mean_reward, 5)) + "\t\t\t\t\ttypical dev = " + str(dev_reward) + "")
