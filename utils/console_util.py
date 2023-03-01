
def print_progress_bar(prefix, i, cnt, suffix, fill_char='█', bar_length=50):

    perc = i/cnt
    filled = round(perc*bar_length)
    not_filled = bar_length-filled
    bar = fill_char*filled + '-'*not_filled
    print(f'\r{prefix}:\t|{bar}| ({i}/{cnt} | {round(perc*100)}%) - {suffix}', end='', flush=True)

