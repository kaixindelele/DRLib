"""
"@Author    : kaixindelele,
"@Contact   : 
CSDN:	https://blog.csdn.net/hehedadaq;
知乎: 	https://www.zhihu.com/people/heda-he-28
"@Describe  : 对于多任务、多设置实验的成功率可视化，
            读取文本的问题有点复杂,我就直接把数据手动输入到字典中。
"""

import matplotlib.pyplot as plt
import numpy as np

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()
barFontSize = 16
xTicksFontSize = 16
yTicksFontSize = 16
yLabelFontSize = 16
legendFontSize = 16
titleFontSize = 16
errorBarSize = 5


def plot_success(exp_data_dict, 
                #  task_names, exp_names,
                 title_str='Success Rate (%)', y_label_str="Success Rate (%)"):
    """
    :param exp_data_dict: [name:list(1,2,3)]
    :param task_names: ['Reaching task', 'Lifting task', ]
    :param exp_names: ['PNO', 'PNR', 'POR']
    :param title_str: 'Fig'
    :param y_label_str: 'Success Rate (%)'
    :return: None
    """
    bar_width = 0.6
    # 有i个实验，在不同任务中的位置。
    x_axis_tick = np.arange(len(exp_data_dict))
    values = []
    labels = []
    for key, value in exp_data_dict.items():
        values.append(value[0])
        labels.append(key)
    print("x_axis_tick:", x_axis_tick)
    print("values:", values)
    print("labels:", labels)
    for index in range(len(exp_data_dict)):
        rect_mean = plt.bar(x=x_axis_tick[index],
                            height=values[index],
                            width=bar_width,
                            align="center",
                            label=labels[index],
                            )
    
    # 给legend赋值字体大小
    # plt.legend(loc=0, numpoints=1)
    plt.legend(loc='upper center',
               borderaxespad=0.,
               )
    leg = plt.gca().get_legend()
    text = leg.get_texts()
    plt.setp(text, fontsize=legendFontSize)
    # 给每个柱状图都标上均值，按照规律来。
    for i, y in enumerate(values):
        plt.text(x_axis_tick[i], 1*y + 1.5, '%s' % round(y, 2), ha='center', FontSize=barFontSize)
    
    plt.xticks(x_axis_tick, labels, FontSize=xTicksFontSize)
    # 确保y轴标签处于0-100，每隔20一个，
    y_ticks = np.arange(0, 120, 20)
    print("y_ticks:", y_ticks)
    plt.yticks(y_ticks, FontSize=yTicksFontSize)
    plt.ylabel(y_label_str, FontSize=yLabelFontSize)

    # plt.title(title_str, FontSize=titleFontSize)
    plt.show()


def main():
    data_dict = dict()
    # 确保成功率是%格式！不能小于1
    data_dict = {"dense": [10.0/15.0 * 100.0],
                 "dense2sparse": [14.0/15.0 * 100.0],
                 "sparse": [0.0/15.0 * 100.0],
                 }
    plot_success(data_dict,
                 )


if __name__ == "__main__":
    main()

