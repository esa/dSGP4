import subprocess
import sys
import textwrap
import unittest

WITHOUT_MATPLOTLIB = textwrap.dedent("""
    import sys

    class BlockMatplotlib:
        def find_spec(self, name, path=None, target=None):
            if name == 'matplotlib' or name.startswith('matplotlib.'):
                raise ImportError('matplotlib is blocked by the test')
            return None

    sys.meta_path.insert(0, BlockMatplotlib())

    import dsgp4
    assert 'matplotlib' not in sys.modules, 'import dsgp4 pulled in matplotlib'

    for plot_function in (dsgp4.plot_orbit, dsgp4.plot_tles):
        try:
            plot_function(None)
        except ImportError as e:
            assert 'dsgp4[plot]' in str(e), str(e)
        else:
            raise AssertionError(f'{plot_function.__name__} did not raise ImportError')
""")


class OptionalMatplotlibTestCase(unittest.TestCase):
    def test_import_and_plot_errors_without_matplotlib(self):
        result = subprocess.run([sys.executable, '-c', WITHOUT_MATPLOTLIB], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()

