import shutil
import os


#定义一个删除空文件和非指定类型文件的函数
def delete_null_dir(parent):
#如果是文件夹的话，那么进入下面的循环
    if os.path.isdir(parent):
        # print("进入删除模式：")，这里你不清楚，直接可以print变量，看看到底是啥
        # 如p是打开parent这个目录里面的文件和文件夹。
        for p in os.listdir(parent):
        # 这是一个递归还是嵌套？反正就是可以一次性扫光你根目录下，所有的文件和文件夹。
        # d是将路径和新的文件夹名联合起来，如果新的路径d是文件夹，再次调用这个函数
            d  = os.path.join(parent, p)
            if (os.path.isdir(d) == True):
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
            if (os.path.isdir(d) == True):
                delete_no_checkpoints(d)
        print("----")

        print("document:", document)

        if (len(document) > 0):
            old_path_name = parent.split("\\")[-1]
            print("old_path_name:", old_path_name)
            # change = input("是否需要删除(y/n)？")
            # if (change == 'y'):
            try:
                # 判断后缀是否在集合里，如果没有后缀，那么就是文件夹了
                if 'pth' not in document and 'pt' not in document and 'checkpoint' not in document and 'progress.txt' in document:
                    for doc in document:
                        os.remove(os.path.join(old_path_name, doc))
                    shutil.retree(old_path_name)
                    print("old_path_name:", old_path_name)
                    print("删除成功！")
            except Exception as e:
                print("delete e:", e)


if __name__ == "__main__":  # 执行本文件则执行下述代码
    path = r'/home/lyl/robot_code/DRLib/HER_DRLib_exps/'
    # delete_null_dir(path)
    flag = input("确定是否执行删除？无法恢复！y/n")
    if flag:
        delete_no_checkpoints(path)
        delete_null_dir(path)
