import sys


class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        print("filename:", filename)
        self.filename = filename
        self.add_flag = add_flag
        # self.log = open(filename, 'a+')

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass


def main():
    logger_kwargs = {'output_dir':"logger/"}
    try:
        import os
        os.mkdir(logger_kwargs['output_dir'])
    except:
        pass
    sys.stdout = Logger(logger_kwargs["output_dir"]+"print.log",
                        sys.stdout)
    
    print('print something')
    print("*" * 3)    
    import time
    time.sleep(2)
    print("other things")


if __name__ == '__main__':
    main()
