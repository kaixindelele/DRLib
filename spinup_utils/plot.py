"""
相比于原始的plot.py文件，增加了如下的功能：
1.可以直接在pycharm或者vscode执行，也可以用命令行传参；
2.按exp_name排序，而不是按时间排序；
3.固定好每个exp_name的颜色；
4.可以调节曲线的线宽，便于观察；
5.保存图片到本地，便于远程ssh画图~
6.自动显示全屏
7.图片自适应
8.针对颜色不敏感的人群,可以在每条legend上注明性能值,和性能序号
seaborn版本0.8.1
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()


def plot_data(data, xaxis='Epoch', value="TestEpRet",
              condition="Condition1", smooth=1,
              linewidth=4,
              rank=True,
              performance=True,
              **kwargs):
    performance_rank_dict = {}
    condition2_list = []
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            condition2_list.append(datum["Condition2"].values[0])
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            datum[value] = smoothed_x
            # add mean performance to performance_rank{dict}
            print("rank-add:", datum[condition].values[0])
            if datum[condition].values[0] not in performance_rank_dict.keys():
                performance_rank_dict[datum[condition].values[0]] = np.mean(smoothed_x[-len(smoothed_x)//10:])
            else:
                performance_rank_dict[datum[condition].values[0]] += np.mean(smoothed_x[-len(smoothed_x)//10:])
    # concern the multi-seeds:
    for key in performance_rank_dict.keys():
        seed_num = sum([1 for cond in condition2_list if key in cond])
        performance_rank_dict[key] /= seed_num

    # value list 获取性能值排序序号
    performance_list = []
    performance_rank_keys = []
    for key, val in performance_rank_dict.items():
        print(key, val)
        performance_list.append(val)
        performance_rank_keys.append(key)

    # 获取列表排序序号,一定要argsort2次~
    performance_rank_list = np.argsort(np.argsort(-np.array(performance_list)))
    performance_rank_sort_dict = {performance_rank_keys[index]: performance_rank_list[index]
                                  for index in range(len(performance_rank_list))}
    print("performance_rank_list:", performance_rank_list)

    # 修改data[condition]的名字
    for index, datum in enumerate(data):
        origin_key = datum[condition].values[0]
        if performance:
            p = performance_rank_dict[origin_key]
            datum[condition] = 'P-' + str(np.round(p, 3)) + "-" + datum[condition]
        if rank:
            rank_value = performance_rank_sort_dict[origin_key]
            datum[condition] = 'Rank-' + str(rank_value) + "-" + datum[condition]

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    sns.set(style="darkgrid", font_scale=1.75, )
    # # data按照lenged排序；
    data.sort_values(by='Condition1', axis=0)

    sns.tsplot(data=data,
               time=xaxis,
               value=value,
               unit="Unit",
               condition=condition,
               ci='sd',
               linewidth=linewidth,
               color=sns.color_palette("Paired", len(data)),
               # palette=sns.color_palette("hls", 8),
               **kwargs)
    """
    If you upgrade to any version of Seaborn greater than 0.8.1, switch from 
    tsplot to lineplot replacing L29 with:
        sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)
    Changes the colorscheme and the default legend style, though.        

    plt.legend()
        loc:图例位置,可取('best', 'upper right', 'upper left', 'lower left', 'lower right', 
            'right', 'center left', 'center , right', 'lower center', 'upper center', 'center')
            若是使用了bbox_to_anchor,则这项就无效了
        fontsize: int或float或{'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'},字体大小；
        frameon: 是否显示图例边框,
        ncol: 图例的列的数量,默认为1,
        title: 为图例添加标题
        shadow: 是否为图例边框添加阴影,
        markerfirst: True表示图例标签在句柄右侧,false反之,
        markerscale: 图例标记为原图标记中的多少倍大小,
        numpoints: 表示图例中的句柄上的标记点的个数,一般设为1,
        fancybox: 是否将图例框的边角设为圆形
        framealpha: 控制图例框的透明度
        borderpad: 图例框内边距
        labelspacing: 图例中条目之间的距离
        handlelength: 图例句柄的长度
        bbox_to_anchor: (横向看右,纵向看下),如果要自定义图例位置或者将图例画在坐标外边,用它,
            比如bbox_to_anchor=(1.4,0.8),这个一般配合着ax.get_position(),
            set_position([box.x0, box.y0, box.width*0.8 , box.height])使用
    """
    plt.legend(loc='upper center',
               ncol=1,
               handlelength=6,
               mode="expand",
               borderaxespad=0.,
               )
    """
    For the version of the legend used in the Spinning Up benchmarking page, 
    swap L38 with:
    plt.legend(loc='upper center', ncol=6, handlelength=1,
               mode="expand", borderaxespad=0., prop={'size': 13})
    """

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.tight_layout(pad=0.5)


def get_datasets(logdir, condition=None):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger.
    Assumes that any file "progress.txt" is a valid hit.
    """
    global exp_idx
    global units
    datasets = []
    roots = []
    exp_names = []
    for root, _, files in os.walk(logdir):
        if 'progress.txt' in files:
            exp_name = None
            try:
                config_path = open(os.path.join(root, 'config.json'))
                config = json.load(config_path)
                if 'exp_name' in config:
                    exp_name = config['exp_name']
                exp_names.append(exp_name)
                roots.append(root)
            except Exception as e:
                print("e:", e)
                print('No file named config.json')
    # just leave one seed:
    # roots_names_dict = {exp_names[index]: roots[index] for index in range(len(exp_names))}
    # exp_name(str) --> roots(list) with diff seeds
    roots_names_dict = {exp_names[index]: roots for index in range(len(exp_names))}
    for key, value in roots_names_dict.items():
        print(key, value)
    # 按照实验名排序
    roots_names_list = sorted(roots_names_dict.items(), key=lambda x: x[0])
    print("roots_names_list:", roots_names_list)
    roots_names_dict = {tup[0]: tup[1] for tup in roots_names_list}
    print("roots_names_dict:", roots_names_dict)

    for exp_name, roots in roots_names_dict.items():
        for root in roots:
            condition1 = condition or exp_name or 'exp'
            condition2 = condition1 + '-' + str(exp_idx)
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1

            try:
                exp_data = pd.read_table(os.path.join(root, 'progress.txt'))
                line_num = len(exp_data)
                print('line num:{}, read from {}'.format(line_num,
                                                         os.path.join(root, 'progress.txt')))
            except:
                print('Could not read from %s' % os.path.join(root, 'progress.txt'))
                continue
            performance = 'TestEpRet' if 'TestEpRet' in exp_data else 'AverageTestEpRet'
            exp_data.insert(len(exp_data.columns), 'Unit', unit)
            exp_data.insert(len(exp_data.columns), 'Condition1', condition1)
            exp_data.insert(len(exp_data.columns), 'Condition2', condition2)
            exp_data.insert(len(exp_data.columns), 'Performance', exp_data[performance])
            datasets.append(exp_data)
    # # 默认按照时间顺序获取文件夹数据
    # print("-"*10, 'sorted by time', '-'*10)
    # for root, _, files in os.walk(logdir):
    #     if 'progress.txt' in files:
    #         exp_name = None
    #         try:
    #             config_path = open(os.path.join(root, 'config.json'))
    #             config = json.load(config_path)
    #             if 'exp_name' in config:
    #                 exp_name = config['exp_name']
    #         except:
    #             print('No file named config.json')
    #         condition1 = condition or exp_name or 'exp'
    #         condition2 = condition1 + '-' + str(exp_idx)
    #         exp_idx += 1
    #         if condition1 not in units:
    #             units[condition1] = 0
    #         unit = units[condition1]
    #         units[condition1] += 1
    #
    #         try:
    #             exp_data = pd.read_table(os.path.join(root, 'progress.txt'))
    #             line_num = len(exp_data)
    #             print('line num:{}, read from {}'.format(line_num,
    #                                                      os.path.join(root, 'progress.txt')))
    #         except:
    #             print('Could not read from %s' % os.path.join(root, 'progress.txt'))
    #             continue
    #         # performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'TestEpRet'
    #         # performance = 'AverageEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet'
    #         performance = 'TestSuccess' if 'TestSuccess' in exp_data else 'AverageEpRet'
    #         exp_data.insert(len(exp_data.columns),'Unit',unit)
    #         exp_data.insert(len(exp_data.columns),'Condition1',condition1)
    #         exp_data.insert(len(exp_data.columns),'Condition2',condition2)
    #         exp_data.insert(len(exp_data.columns),'Performance',exp_data[performance])
    #         datasets.append(exp_data)
    return datasets


def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is,
           pull data from it;
        2) if not, check to see if the entry is a prefix for a
           real directory, and pull data from that.
    """
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1] == os.sep:
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x: osp.join(basedir, x)
            prefix = logdir.split(os.sep)[-1]
            print("basedir:", basedir)
            listdir = os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])

    """
    Enforce selection rules, which check logdirs for certain substrings.
    Makes it easier to look at graphs from particular ablations, if you
    launch many jobs at once with similar names.
    """
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [log for log in logdirs if all(not (x in log) for x in exclude)]

    # Verify logdirs
    print('Plotting from...\n' + '=' * DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '=' * DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs
    assert not (legend) or (len(legend) == len(logdirs)), \
        "Must give a legend title for each set of experiments."

    # Load data from logdirs
    data = []
    if legend:
        for log, leg in zip(logdirs, legend):
            data += get_datasets(log, leg)
    else:
        for log in logdirs:
            data += get_datasets(log)
    return data


def make_plots(all_logdirs, legend=None,
               xaxis=None, values=None,
               count=False,
               font_scale=1.5, smooth=1,
               linewidth=4,
               select=None, exclude=None,
               estimator='mean',
               rank=True,
               performance=True,
               ):
    data = get_all_datasets(all_logdirs, legend, select, exclude)
    values = values if isinstance(values, list) else [values]
    condition = 'Condition2' if count else 'Condition1'
    estimator = getattr(np, estimator)  # choose what to show on main curve: mean? max? min?
    for value in values:
        plt.figure()
        plot_data(data, xaxis=xaxis, value=value,
                  condition=condition, smooth=smooth, estimator=estimator,
                  linewidth=linewidth, rank=rank, performance=performance)

    # 默认最大化图片
    manager = plt.get_current_fig_manager()
    try:
        # matplotlib3.3.4 work
        manager.resize(*manager.window.maxsize())
    except:
        # matplotlib3.2.1//2.2.3 work
        manager.window.showMaximized()
    fig = plt.gcf()
    fig.set_size_inches((16, 9), forward=False)

    select_str = ''
    exclude_str = ''
    print("select:", select)
    print("select_str:", select_str)
    if select is not None and type(select) is list:
        for s_str in select:
            select_str += s_str
    if exclude is not None and type(exclude) is list:
        for s_str in exclude:
            exclude_str += s_str
    print("select_str:", select_str)
    try:
        # 如果非远程，则显示图片
        plt.show()
    except:
        pass
    fig.savefig(all_logdirs[0] + 'ep_reward_'+select_str+exclude_str+'.png',
                bbox_inches='tight',
                dpi=300)
    # plt.savefig(all_logdirs[0] + 'ep_reward.png',
    #            bbox_inches='tight',
    #            dpi=300,
    #            )



def main():
    import argparse
    parser = argparse.ArgumentParser()
    import sys
    # 如果是命令行启动,调用下面的语句,必须要输入数据路径!
    if len(sys.argv) > 1:        
        print("run in command: \n argv:", sys.argv, '\n', '-' * 30)        
        parser.add_argument('logdir', nargs='*')
        # other nargs        
        parser.add_argument('--select', nargs='*',
                            help='在当前路径下,选择特定关键词,不能是下一个文件夹,'
                                 '在idle中不能是字符串,在终端,不用加双引号,多个关键词可以用空格隔开')
        parser.add_argument('--exclude', nargs='*',
                            help='同select')
    else:
        # 如果是idle启动,用于debug,则需要将路径加入到下面的语句!
        print("run in pycharm\n", '-' * 30)        
        parser.add_argument('--logdir', '-r', type=list,
                            default=[
                                "/home/dongkun/spinup/DRLib/spinup_utils/HER_DRLib_rew_PP_fork_pos/2",
                            ])
        # other nargs
        parser.add_argument('--select', default=[], )
        parser.add_argument('--exclude', default=[], )
        
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--xaxis', '-x', default='TotalEnvInteracts',
                        help='选择什么为横坐标,默认为TotalEnvInteracts')
    parser.add_argument('--value', '-y', default='Performance', nargs='*',
                        help='选择特定变量为性能指标,默认为AverageTestEpRet')
    parser.add_argument('--count', action='store_true',
                        help='是否显示每个随机种子,加--count为显示')
    # parser.add_argument('--count', default="False")
    parser.add_argument('--smooth', '-s', type=int, default=20,
                        help='滑动平均,20看起来会更平滑些')
    parser.add_argument('--linewidth', '-lw', type=float, default=4,
                        help='实验线宽,粗点容易分清')
    parser.add_argument('--rank', type=bool, default=True,
                        help='是否在legend上显示性能排序')
    parser.add_argument('--performance', type=bool, default=True,
                        help='是否在legend上显示性能值')
    parser.add_argument('--est', default='mean')
    args = parser.parse_args()
    print("args:", args)
    make_plots(args.logdir, args.legend, args.xaxis, args.value, args.count,
               smooth=args.smooth, select=args.select, exclude=args.exclude,
               estimator=args.est,
               linewidth=args.linewidth,
               rank=args.rank,
               performance=args.performance)


if __name__ == "__main__":
    main()


