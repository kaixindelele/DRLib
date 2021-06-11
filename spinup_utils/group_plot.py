import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np
COLORS = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3ff']

FontSize = 19
DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()


def plot_data(data,
              dir_index,
              ax,
              legend_flag,
              xaxis='Epoch',
              value="AverageEpRet",
              condition="Condition1",
              smooth=1, color='r',
              nrow=1,
              **kwargs):
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            datum[value] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    sns.set(style="darkgrid", font_scale=2)
    # value = "AverageEpisodeReward"
    sns.tsplot(data=data,
               time=xaxis,
               value=value, unit="Unit",
               condition=condition,
               legend=legend_flag,
               ci='sd',
               color=COLORS,
               linewidth=4,
               ax=ax,
               **kwargs)
    # true 将会在x轴显示abc.
    reach_flag = False
    # 下面是根据第一层第二层，以及每一列，设置x/y轴标签，title等细节，根据需要自己定制。
    if nrow == 1:
        x_label = ''
        # 设置x轴的单位，可以注释掉
        ax.set_xlabel(x_label, fontsize=FontSize)
    elif nrow == 2:
        if reach_flag:
            x_label = xaxis + '\n\n(' + chr(dir_index+97) +')'
        else:
            x_label = xaxis
        ax.set_xlabel(x_label, fontsize=FontSize)
    if nrow == 1:
        if dir_index == 0:
            # 这个TD3和SAC是定制的，可以删掉或者重写，甚至不需要y轴标签的话，整句话注释掉
            ax.set_ylabel(value+"-TD3", fontsize=FontSize+4)
            # 是否给子图加标题，不同层和列的需求不一样，根据需求添加
            title_str = 'title in first'
        else:
            title_str = 'title in second ' + str(dir_index*5) + ' Degrees'
            ax.set_ylabel('', fontsize=0)    
        ax.set_title(title_str)
    else:
        if dir_index == 0:
            # ax.set_ylabel(value+"-SAC-AUTO", fontsize=FontSize+4)
            ax.set_ylabel(value+"-SAC", fontsize=FontSize+4)
        else:
            ax.set_ylabel('', fontsize=FontSize+4)    
    # plt.legend(bbox_to_anchor=(0.05, 1), loc=1, borderaxespad=0.)
    ax.tick_params(
        axis='x',  # changes apply to the x-axis
        labelsize=FontSize + 3
    )
    ax.tick_params(
        axis='y',  # changes apply to the x-axis
        labelsize=FontSize + 3
    )
    # 下面的是将legend放在最后一张子图的左下角
    # plt.legend(loc='lower right', ncol=2, handlelength=2,
    #            borderaxespad=0., prop={'size': FontSize})
    # 下面是将legend放在整张图的下面居中，更好看一些
    plt.legend(bbox_to_anchor=(0.2, -0.3), ncol=4, handlelength=1,
               borderaxespad=0., prop={'size': FontSize})

    xscale = np.max(np.asarray(data[xaxis])) > 5e3    
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), labelsize=FontSize)
        if nrow == 2:
            ax.xaxis.offsetText.set_fontsize(FontSize + 3)
        else:
            ax.xaxis.offsetText.set_fontsize(0)    
    # 加栅格，和栅格的透明度
    ax.grid(True, alpha=0.4)
    
    # plt.tight_layout(pad=0.5)


def get_datasets(logdir, condition=None):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger.
    Assumes that any file "progress.txt" is a valid hit.
    """
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        if 'progress.txt' in files:
            exp_name = None
            try:
                config_path = open(os.path.join(root, 'config.json'))
                config = json.load(config_path)
                if 'exp_name' in config:
                    exp_name = config['exp_name']
            except:
                print('No file named config.json')
            condition1 = condition or exp_name or 'exp'
            # condition2 代表其他随机种子
            condition2 = condition1 + '-' + str(exp_idx)
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1

            try:
                exp_data = pd.read_table(os.path.join(root,'progress.txt'))
            except Exception as e:
                print("read e:", e)
                print('Could not read from %s' % os.path.join(root, 'progress.txt'))
                continue
            performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet'
            exp_data.insert(len(exp_data.columns), 'Unit', unit)
            exp_data.insert(len(exp_data.columns), 'Condition1', condition1)
            exp_data.insert(len(exp_data.columns), 'Condition2', condition2)
            exp_data.insert(len(exp_data.columns), 'Performance', exp_data[performance])
            datasets.append(exp_data)
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
            listdir = os.listdir(basedir)
            # logdirs += sorted([fulldir(x) for x in listdir if prefix in x])
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])
            for log in logdirs:
                print("logdirs_list:", log)

    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        print("exclude:", exclude)
        print("logdirs:", logdirs)
        logdirs = [log for log in logdirs if all(not(x in log) for x in exclude)]

    # Verify logdirs
    print('Plotting from...\n' + '='*DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '='*DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs
    assert not(legend) or (len(legend) == len(logdirs)), \
        "Must give a legend title for each set of experiments."

    # Load data from logdirs
    data = []
    if legend:
        for log, leg in zip(logdirs, legend):
            data += get_datasets(log, leg)
    else:
        # 修改一下 legend的名称，这个可以根据自己的需要重新写。
        # log是一个字符串，如果某些关键词在log中，直接将legend修改为你需要的字符串
        for log in logdirs:
            print("log:", log)
            if 'dense2sparse' in log:
                leg = 'Dense2Sparse'
            if 'oracle' in log:
                leg = 'Oracle'
            if 'dense' in log and 'sparse' not in log:
                leg = 'Dense'
            if 'dense' not in log and 'sparse' in log:
                leg = 'Sparse'
            if 'real' in log:
                leg = 'Real'
                continue

            data += get_datasets(log, leg)
            # data += get_datasets(log)
    return data


def make_plots(all_logdirs,
               dir_index,
               ax,
               legend_flag=True,
               legend=None, xaxis=None, values=None, count=False,
               font_scale=1.5, smooth=1, select=None, exclude=None, estimator='mean', nrow=1):
    data = get_all_datasets(all_logdirs,
                            legend,
                            select,
                            exclude)
    values = values if isinstance(values, list) else [values]
    condition = 'Condition2' if count else 'Condition1'
    # choose what to show on main curve: mean? max? min?
    estimator = getattr(np, estimator)

    for value in values:
        plot_data(data,
                  dir_index=dir_index,
                  ax=ax,
                  legend_flag=legend_flag,
                  xaxis=xaxis, value=value,
                  condition=condition, smooth=smooth,
                  estimator=estimator, nrow=nrow)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    import sys
    sys.path.append(r'../')
    # 设定好子图的路径，目标是2*3张子图，即两层，每层3列
    # 第一层的三个路径：
    sac_camera0 = r'F:\Train-Reach-camera-0-sac\2'
    sac_camera5 = r'F:\Train-Reach-camera-5-sac\2'
    sac_camera10 = r'F:\Train-Reach-camera-10-sac\2'
    
    # 第二层的三个路径
    td3_camera0 = r'F:\Train-Reach-camera-0\2'
    td3_camera5 = r'F:\Train-Reach-camera-5\2'
    td3_camera10 = r'F:\Train-Reach-camera-10\2'

    parser.add_argument('--logdir', '-r', default=[sac_camera0,
                                                   sac_camera5,
                                                   sac_camera10,
                                                   td3_camera0,
                                                   td3_camera5,
                                                   td3_camera10,
                                                   ])

    # parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--xaxis', '-x', default='TotalEnvInteracts')
    # parser.add_argument('--value', '-y', default='AverageQ2', nargs='*')
    parser.add_argument('--value', '-y', default='AverageTestEpRet', nargs='*')
    parser.add_argument('--count', default=False)
    parser.add_argument('--smooth', '-s', type=int, default=20)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--est', default='mean')
    args = parser.parse_args()

    print("legend:", args.legend)
    # 设定好 row(行) 为2
    nrows = 2
    # 列 数目直接可以算出来
    ncols = len(args.logdir)//nrows
    # 调节分辨率和图像大小
    figsize = (4 * nrows, 4 * ncols)
    f, axarr = plt.subplots(nrows, ncols,
                            sharex=False,
                            sharey=True,
                            squeeze=False,
                            figsize=figsize)
    # 这里迭代六张图的路劲，前三个是第一层的，后三个是第二层的。
    for logdir_index, logs in enumerate(args.logdir):      
        # 确定只有最后一张子图有legend
        # 也可以直接设定legend的位置在整张图的下面居中
        if logdir_index == len(args.logdir)-1:
            legend_flag = True
        else:
            legend_flag = False
        # 如果是第一层的（logdir_index=0,1,2），做如下处理：
        if logdir_index < 3:
            make_plots([logs],
                       logdir_index,
                       axarr[0][logdir_index],
                       legend_flag,
                       args.legend, args.xaxis, args.value, args.count,
                       smooth=args.smooth, select=args.select, exclude=args.exclude,
                       estimator=args.est,
                       nrow=1,
                       )
        # 如果是第二层的图（logdir_index=3,4,5），做如下处理：
        if logdir_index > 2:
            make_plots([logs],
                       logdir_index-3,
                       axarr[1][logdir_index-3],
                       legend_flag,
                       args.legend, args.xaxis, args.value, args.count,
                       smooth=args.smooth, select=args.select, exclude=args.exclude,
                       estimator=args.est,
                       nrow=2,
                       )
    # plt.legend(loc='best')
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
    plt.show()


if __name__ == "__main__":
    main()
