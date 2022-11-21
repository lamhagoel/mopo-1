import os
import atexit


class Logger:
    def __init__(self, writer, output_file, output_fname="progress.txt"):
        self.writer = writer
        self.output_file = open(os.path.join(self.writer.get_logdir(), output_fname), 'w')
        atexit.register(self.output_file.close)
        
        self.output_file2 = output_file
        if self.output_file2 is not None:
            self.output_file2 = open(self.output_file2, 'w')
            atexit.register(self.output_file2.close)

    def record(self, tag, scalar_value, global_step, printed=True):
        self.writer.add_scalar(tag, scalar_value, global_step)
        if printed:
            info = f"{tag}: {scalar_value:.3f}"
            print("\033[1;32m [info]\033[0m: " + info)
            self.output_file.write(info + '\n')
            if self.output_file2 is not None:
                self.output_file2.write(info + '\n')

    def print(self, info):
        print("\033[1;32m [info]\033[0m: " + info)
        self.output_file.write(info + '\n')
        if self.output_file2 is not None:
            self.output_file2.write(info + '\n')
