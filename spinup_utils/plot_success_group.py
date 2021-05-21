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


def plot_success(exp_data_dict, task_names, exp_names,
                 title_str='Figure', y_label_str="Success Rate (%)"):
    """
    :param exp_data_dict: [name:list(1,2,3)]
    :param task_names: ['Reaching task', 'Lifting task', ]
    :param exp_names: ['PNO', 'PNR', 'POR']
    :param title_str: 'Fig'
    :param y_label_str: 'Success Rate (%)'
    :return: None
    """
    # 按照实验名迭代，对每个任务画一个柱子
    for exp_index, exp_name in enumerate(exp_names):
        total_mean = []
        total_std = []
        for task_name in task_names:
            mean_list = []
            for m in exp_data_dict[exp_name+'_'+task_name]:
                success_num = m
                mean_list.append([np.round(float(success_num)/1.0, 2)])
            mean = np.mean(np.array(mean_list))
            std = np.std(np.array(mean_list))
            total_mean.append(mean)
            total_std.append(std)

        bar_width = 0.5
        # 有i个实验，在不同任务中的位置。
        x = np.arange(len(task_names)) * (len(exp_names)+1) * bar_width + exp_index * bar_width
        print("x:", x)
        rect_mean = plt.bar(x=x,
                            height=total_mean,
                            width=bar_width,
                            align="center",
                            label=exp_name,
                            )
        rect_std = plt.errorbar(x=x,
                                y=total_mean,
                                yerr=total_std,
                                fmt='o',        # 中心点形状
                                ecolor='r',     # 竖线颜色
                                color='b',      # 横线颜色
                                elinewidth=2,   # 线宽
                                capsize=errorBarSize,   # 横线长度
                                )
        # 给legend赋值字体大小
        plt.legend(loc=0, numpoints=1)
        leg = plt.gca().get_legend()
        text = leg.get_texts()
        plt.setp(text, fontsize=legendFontSize)
        # 给每个柱状图都标上均值，按照规律来。
        for i, y in enumerate(total_mean):
            plt.text(x[i], 1*y + 1.5, '%s' % round(y, 2), ha='center', FontSize=barFontSize)
    # 在x的基础上，加了半个宽度
    x_center_list = [bar_width*(i*(len(exp_names)+1)+(len(exp_names))/2) for i in range(len(task_names))]
    print(x_center_list)
    np.arange(len(task_names)) * (len(exp_names) + 1)

    plt.xticks(x_center_list, task_names, FontSize=xTicksFontSize)
    plt.yticks(FontSize=yTicksFontSize)
    plt.ylabel(y_label_str, FontSize=yLabelFontSize)

    plt.title(title_str, FontSize=titleFontSize)
    plt.show()


def main():
    data_dict = dict()
    # 不同的任务，每个任务有不同的实验设置，中间用_连起来
    data_dict = {"PNO_Reaching task": [77.8, 20, 80],
                 "PNR_Reaching task": [90.0, 89, 29],

                 "PNO_Lifting task": [8.2, 30, 90],
                 "PNR_Lifting task": [69.2, 20, 102],
                 }
    task_names = ['Reaching task', 'Lifting task']
    exp_names = ['PNO', 'PNR']
    plot_success(data_dict,
                 task_names=task_names,
                 exp_names=exp_names)


if __name__ == "__main__":
    main()

