import shutil
import os
import pandas as pd
min_line_num = 20


def get_progress_line_num(root):
    line_num = 0
    try:
        exp_data = pd.read_table(os.path.join(root, 'progress.txt'))
        line_num = len(exp_data)
        print('line num:{}, read from {}'.format(line_num,
                                                 os.path.join(root, 'progress.txt')))
    except:
        print('line num:{}, can not read from {}'.format(line_num,
                                                 os.path.join(root, 'progress.txt')))

    return line_num


# 定义一个删除空文件和非指定类型文件的函数
def delete_null_dir(parent):
    # 如果是文件夹的话，那么进入下面的循环
    if os.path.isdir(parent):
        # print("进入删除模式：")，这里你不清楚，直接可以print变量，看看到底是啥
        # 如p是打开parent这个目录里面的文件和文件夹。
        for p in os.listdir(parent):
            # 这是一个递归还是嵌套？反正就是可以一次性扫光你根目录下，所有的文件和文件夹。
            # d是将路径和新的文件夹名联合起来，如果新的路径d是文件夹，再次调用这个函数
            d = os.path.join(parent, p)
            if os.path.isdir(d):
                delete_null_dir(d)
    # os.listdir(parent)拿到文件夹里的所有东西，如果为空，就是空文件夹
    # 所以这个判断，就是删除所有为空的文件夹。
    if not os.listdir(parent):
        print(parent)
        os.rmdir(parent)
        print("删除成功！")


def delete_no_checkpoints(parent):
    if os.path.isdir(parent):
        document = []
        for p in os.listdir(parent):
            try:
                document.append(p)
            except:
                print("not document~")
            d = os.path.join(parent, p)
            # print(d)
            if os.path.isdir(d):
                delete_no_checkpoints(d)
        print("----")

        print("document:", document)

        if len(document) > 0:
            old_path_name = parent.split("\\")[-1]
            print("old_path_name:", old_path_name)
            # change = input("是否需要删除(y/n)？")
            # if (change == 'y'):
            try:
                # 判断后缀是否在集合里，如果没有后缀，那么就是文件夹了
                save_keys = ['.pt', '.pth', 'checkpoint', '.pkl']
                entry_keys = ['progress.txt']
                delete_flag = False
                for entry_key in entry_keys:
                    if entry_key in document:
                        line_num = get_progress_line_num(root=old_path_name)
                        for value in document:
                            save_value = [value for save_key in save_keys if save_key in value]
                        if len(save_value) > 1:
                            delete_flag = False
                            break
                        else:
                            delete_flag = True
                if line_num < min_line_num:
                    delete_flag = True
                if delete_flag:
                    input("确定是否执行删除？无法恢复！")
                    for doc in document:
                        os.remove(os.path.join(old_path_name, doc))
                    shutil.retree(old_path_name)
                    print("old_path_name:", old_path_name)
                    print("删除成功！")
            except Exception as e:
                print("delete e:", e)


if __name__ == "__main__":  # 执行本文件则执行下述代码
    path = r'/home/lyl/robot_code/DRLib/spinup_utils/HER_DRLib_rew_PP_exps/'
    # delete_null_dir(path)
    flag = input("确定是否执行删除？无法恢复！无法恢复！无法恢复！是否确定！y/n")
    if flag == 'y':
        delete_no_checkpoints(path)
        delete_null_dir(path)

