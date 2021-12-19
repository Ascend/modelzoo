from sys import argv

sec_step = float(argv[1])
batch_size = float(argv[2])
print(1000 * batch_size / sec_step)