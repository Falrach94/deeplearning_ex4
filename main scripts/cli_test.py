import sys
import time

from utils.cli_table_builder import TableBuilder

from colorama import *

#init(strip=False)

def set_cursor_pos_txt(x, y):
    sys.stdout.write(f'\033[{str(x)};{str(y)}H')

#print('test')
#print('test2')
#print('test3')


t = "test\ntest2\ntest3\x1B[2A tester"
sys.stdout.write(t)
#print(t, end='', flush=True)

exit()

for i in range(5):
    builder = TableBuilder()

    builder.add_line(f'{i}', f'{i}', f'{i}')
    builder.new_block()
    builder.add_line(f'{i}', f'{i}', f'{i}')
    builder.new_block()
    builder.add_line(f'{i}', f'{i}', f'{i}')
    builder.print()
    time.sleep(1)
