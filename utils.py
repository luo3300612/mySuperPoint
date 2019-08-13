import logging


class OutPutUtil:
    def __init__(self, terminal=True, log=True, log_file='./train.log'):
        self.log = log
        self.terminal = terminal
        if log:
            logging.basicConfig(level=logging.DEBUG,
                                filename=log_file,
                                filemode='a',
                                format='%(asctime)s - %(levelname)s: %(message)s')

    def speak(self, message):
        if self.terminal:
            print(message)
        if self.log:
            logging.info(message)
