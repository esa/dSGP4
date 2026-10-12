import subprocess
import sys
import textwrap
import unittest


class OptionalMatplotlibTestCase(unittest.TestCase):
    def test_import_without_matplotlib(self):
        # Run in a fresh interpreter where any attempt to import matplotlib fails.
        code = textwrap.dedent("""
            import sys
            class _Block:
                def find_spec(self, name, path=None, target=None):
                    if name == 'matplotlib' or name.startswith('matplotlib.'):
                        raise ImportError('matplotlib blocked for test')
                    return None
            sys.meta_path.insert(0, _Block())
            import dsgp4
            assert 'matplotlib' not in sys.modules
            try:
                dsgp4.plot_orbit(None)
            except ImportError as e:
                assert 'matplotlib' in str(e)
            else:
                raise AssertionError('plot_orbit should raise ImportError without matplotlib')
        """)
        r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
