import matplotlib.pyplot as plt
import csv


def save_to_file(array, filename, writetype):
    with open(filename, writetype) as f:
        csv.writer(f, delimiter=',').writerow(array)


def plot_and_save(array, xname, yname, plotname):
    fig = plt.figure()
    plt.plot(list(range(len(array))), array)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.savefig(plotname)
    plt.close(fig)


def array_pad_zero(array, xlimit):
    arr_len = len(array)
    if (xlimit > arr_len):
        array.append([0 for i in range(xlimit - arr_len)])
    return array


def read_plot_save(filename, xlimit, xname, yname, plotname):
    with open(filename, newline='') as f:
        data = list(csv.reader(f))
    index = 0
    for array in data:
        plot_and_save(array_pad_zero(list(map(float, array)), xlimit), xname, yname, plotname + str(index) + '.png')
        index += 1

