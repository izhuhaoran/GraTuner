import sqlite3
import json

from opentuner.utils import stats_matplotlib as stats
import matplotlib.pyplot as plt
import os

def save_all_configurations_to_json():
    conn = sqlite3.connect('/home/zhuhaoran/AutoGraph/AutoGraph/graphit/autotune/opentuner.db/node8.db')  # 假设你的数据库连接对象为conn
    cursor = conn.cursor()

    query = "SELECT id, data FROM configuration"
    cursor.execute(query)

    for row in cursor.fetchall():
        config_id = row[0]
        data = row[1]
        json_data = json.loads(data)

        file_name = f"/home/zhuhaoran/AutoGraph/AutoGraph/graphit/autotune/opentuner.db/config_{config_id}.json"
        with open(file_name, 'w') as file:
            json.dump(json_data, file, indent=4)

    cursor.close()
    conn.close()
    
def matplotlibplot_file(labels, xlim=None, ylim=None, disp_types=['median']):
    """
    Arguments,
      labels: List of labels that need to be included in the plot
      xlim: Integer denoting the maximum X-coordinate in the plot
      ylim: Integer denoting the maximum Y-coordinate in the plot
      disp_types: List of measures that are to be displayed in the plot
    Returns,
      A figure object representing the required plot
    """

    figure = plt.figure()
    values = stats.get_values(labels)
    for label in values:
        (mean_values, percentile_values) = values[label]
        for disp_type in disp_types:
            cols = None
            data = percentile_values

            if disp_type == 'median':
                cols = [11]
            elif disp_type == 'mean':
                cols = [1]
                data = mean_values
            elif disp_type == 'all_percentiles':
                cols = list(range(1, 22))

            plotted_data = [[] for x in range(len(cols))]

            x_indices = []
            for data_point in data[1:]:
                x_indices.append(int(data_point[0]))
                for i in range(0, len(cols)):
                    plotted_data[i].append(float(data_point[cols[i]]))
            args = []
            for to_plot in plotted_data:
                args.append(x_indices)
                args.append(to_plot)

            plt.plot(*args, label='%s(%s)' % (label, disp_type))

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.xlabel('Autotuning Time (seconds)')
    plt.ylabel('Execution Time (seconds)')
    plt.legend(loc='upper right')
    plt.show()
    return figure

# os.chdir('/home/zhuhaoran/AutoGraph/AutoGraph/graphit/autotune/opentuner.db')

# labels = stats.get_all_labels()

# xlim = 5000
# ylim = 10
# disp_types = ['median']

# fig = matplotlibplot_file(labels, xlim, ylim, disp_types)

# # plt.show(fig)

save_all_configurations_to_json()