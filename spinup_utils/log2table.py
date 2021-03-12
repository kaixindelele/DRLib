import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np
import warnings
warnings.filterwarnings("ignore")

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

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()


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
                config_path = open(os.path.join(root,'config.json'))
                config = json.load(config_path)
                if 'exp_name' in config:
                    exp_name = config['exp_name']
            except:
                print('No file named config.json')
            condition1 = condition or exp_name or 'exp'
            condition2 = condition1 + '-' + str(exp_idx)
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1

            try:
                exp_data = pd.read_table(os.path.join(root,'progress.txt'))
            except:
                print('Could not read from %s'%os.path.join(root,'progress.txt'))
                continue
            # performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'TestEpRet'
            # performance = 'AverageEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet'
            performance = 'TestSuccess' if 'TestSuccess' in exp_data else 'AverageEpRet'
            exp_data.insert(len(exp_data.columns),'Unit',unit)
            exp_data.insert(len(exp_data.columns),'Condition1',condition1)
            exp_data.insert(len(exp_data.columns),'Condition2',condition2)
            exp_data.insert(len(exp_data.columns),'Performance',exp_data[performance])
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
        if osp.isdir(logdir) and logdir[-1]==os.sep:
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x : osp.join(basedir, x)
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
        for log in logdirs:
            data += get_datasets(log)
    # print("data:", data)
    return data


def make_plots(all_logdirs, legend=None,
               xaxis=None, values=None,
               count=False,
               font_scale=1.5, smooth=1,
               select=None, exclude=None,
               estimator='mean'):
    data = get_all_datasets(all_logdirs, legend, select, exclude)
    values = values if isinstance(values, list) else [values]
    condition = 'Condition2' if count else 'Condition1'
    estimator = getattr(np, estimator)      # choose what to show on main curve: mean? max? min?
    for value in values:
        data2table(data, value=value, exclude='_FetchPush', select='PER')
        data2bar(data, value=value, exclude='_FetchPush', select='PER')


def data2bar(data, value="EpochTime", exclude='_FetchPush', select='PER'):
    mean_list = [np.mean(d[value]) for d in data]
    origin_exp_list = [d["Condition2"][0] for d in data]
    exp_list = []
    for exp in origin_exp_list:
        if select in exp:
            exp_list.append(exp[:exp.find(exclude)] + '_' + exp[exp.find(select):exp.find(select)+3])
        else:
            exp_list.append(exp[:exp.find(exclude)])
    print("exp_list:", exp_list)
    print("mean_list:", mean_list)

    # 画图    
    fig, ax = plt.subplots()
    exp_list.reverse()
    b = ax.barh(range(len(exp_list)), mean_list, color='#6699CC')
    # plt.rc('font', family='SimHei', weight='bold')
    # 为横向水平的柱图右侧添加数据标签。
    for rect in b:
        w = rect.get_width()
        ax.text(w, rect.get_y()+rect.get_height()/2, '%0.3f' %
                float(w), ha='left', va='center', FontSize=barFontSize)
    # 设置Y轴纵坐标上的刻度线标签。
    ax.set_yticks(range(len(exp_list)))
    ax.set_yticklabels(exp_list, FontSize=xTicksFontSize)
    
    # 不要X横坐标上的label标签。
    plt.xticks(FontSize=xTicksFontSize)
    
    plt.title(value, loc='center', fontsize=titleFontSize,
              fontweight='bold', color='red')
    plt.show()
    plt.savefig("barh.jpg", dpi=200, bbox_inches='tight')


def data2table(data, value="EpochTime", exclude='_FetchPush', select='PER'):
    mean_list = [np.mean(d[value]) for d in data]
    origin_exp_list = [d["Condition2"][0] for d in data]
    exp_list = []
    for exp in origin_exp_list:
        if select in exp:
            exp_list.append(exp[:exp.find(exclude)]+ '_' +exp[exp.find(select):exp.find(select)+3])
        else:
            exp_list.append(exp[:exp.find(exclude)])
    print("exp_list:", exp_list)
    print("mean_list:", mean_list)

    key_lens = [len(key) for key in exp_list]
    max_key_len = max(15, max(key_lens))
    while min(mean_list) < 10:
        mean_list = [mean * 10 for mean in mean_list]
    max_value = int(max(mean_list) + 5)
    # keystr = '%' + '%d' % max_key_len
    # 左对齐
    keystr = '%' + '-%d' % max_key_len
    fmt = "| " + keystr + "s | %-"+str(max_value)+"s |"
    n_slashes = max_value + 7 + max_key_len
    print("-" * n_slashes)
    print(fmt % ("exp_name", value))
    print("-" * n_slashes)
    for i in range(len(mean_list)):
        val = mean_list[i]
        # valstr = "%8.5g" % val if hasattr(val, "__float__") else val
        # 小数位数来表示值的大小，形成柱状图的效果。
        
        lens = "%8."+str(int(val))+"g"        
        valstr = lens % val if hasattr(val, "__float__") else val                    
        if len(valstr) < int(val):
            valstr += '0'* (int(val) - len(valstr))
        print(fmt % (exp_list[i], valstr))


def main():
    import argparse
    parser = argparse.ArgumentParser()

    target_dir = '/home/lyl/robot_code/DRLib/HER_DRLib_exps/2021-03'
    parser.add_argument('--logdir', '-r', default=[target_dir])
    # parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--xaxis', '-x', default='Epoch')
    parser.add_argument('--value', '-y', default=['EpochTime'], nargs='*')
    parser.add_argument('--count', action='store_true')
    # parser.add_argument('--count', default="False")
    parser.add_argument('--smooth', '-s', type=int, default=1)
    parser.add_argument('--select', default=["GPU",], nargs='*')
    parser.add_argument('--exclude', default=['CPU',], nargs='*')
    parser.add_argument('--est', default='mean')
    args = parser.parse_args()
    make_plots(args.logdir, args.legend, args.xaxis, args.value, args.count,
               smooth=args.smooth, select=args.select, exclude=args.exclude,
               estimator=args.est)


if __name__ == "__main__":
    main()
