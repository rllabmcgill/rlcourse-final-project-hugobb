import curses

def close_window(window):
    curses.nocbreak()
    window.keypad(0)
    curses.echo()
    curses.curs_set(1)
    curses.endwin()
