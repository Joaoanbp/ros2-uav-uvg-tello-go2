"""Regression checks for the silent noninteractive-backend failure."""

import os
import unittest
from unittest.mock import MagicMock, patch

from check_graphics import configure_live_plot, main


class GraphicsTests(unittest.TestCase):
    def test_missing_display_fails_instead_of_falling_back_to_agg(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'DISPLAY'):
                configure_live_plot()

    def test_unauthorized_display_fails_before_loading_matplotlib(self):
        x11 = MagicMock()
        x11.XOpenDisplay.return_value = None
        with patch.dict(os.environ, {'DISPLAY': ':0'}), \
                patch('check_graphics.ctypes.CDLL', return_value=x11):
            with self.assertRaisesRegex(RuntimeError, 'XAUTHORITY'):
                configure_live_plot()
        x11.XCloseDisplay.assert_not_called()

    def test_failed_graphical_preflight_returns_nonzero(self):
        with patch('check_graphics.configure_live_plot', side_effect=RuntimeError('Sem acesso ao display')), \
                patch('sys.stderr'):
            self.assertEqual(main(), 1)


if __name__ == '__main__':
    unittest.main()
