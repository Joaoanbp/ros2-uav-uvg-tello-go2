#!/usr/bin/env python3
"""Check X11 authorization and require an interactive Qt plotting backend."""

import ctypes
import os
import sys


def configure_live_plot():
    display = os.environ.get('DISPLAY')
    if not display:
        raise RuntimeError('DISPLAY não está definido; abra um terminal da sessão gráfica.')
    x11 = ctypes.CDLL('libX11.so.6')
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    x11.XCloseDisplay.restype = ctypes.c_int
    connection = x11.XOpenDisplay(display.encode())
    if not connection:
        raise RuntimeError(
            f'Sem acesso ao display {display}. Verifique a autorização X11 '
            '(DISPLAY/XAUTHORITY) ou execute ./scripts/zigzag.sh --shape circle no host.'
        )
    x11.XCloseDisplay(connection)

    import matplotlib
    # Explicit selection disables automatic fallback to Agg, whose timer cannot
    # drive the live plot or start the mission.
    matplotlib.use('Qt5Agg', force=True)
    import matplotlib.pyplot as plt
    return plt


def main():
    try:
        plt = configure_live_plot()
        fig = plt.figure()
        fig.canvas.draw()
        plt.close(fig)
    except (ImportError, RuntimeError, OSError) as error:
        print(f'Erro na interface gráfica: {error}', file=sys.stderr)
        return 1
    import matplotlib
    print(f'Interface gráfica pronta: {matplotlib.get_backend()}, '
          f'Matplotlib {matplotlib.__version__}.', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
