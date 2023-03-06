import time

from utils.cli_table_builder import TableBuilder

#from colorama import *

def set_cursor_pos_txt(x, y):
    return f'\x1b[{str(x)};{str(y)}H'


print('abc')

print()

print(set_cursor_pos_txt(0, 0) + 'bcd')
print("abc" + set_cursor_pos_txt(10, 10) + 'def')

while True:
    pass

for i in range(5):
    builder = TableBuilder()

    builder.add_line(f'{i}', f'{i}', f'{i}')
    builder.new_block()
    builder.add_line(f'{i}', f'{i}', f'{i}')
    builder.new_block()
    builder.add_line(f'{i}', f'{i}', f'{i}')
    builder.print()
    time.sleep(1)
