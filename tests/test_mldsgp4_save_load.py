import os
import tempfile
import unittest

import dsgp4
import torch


class Mldsgp4SaveLoadTestCase(unittest.TestCase):
    def test_save_load_round_trip(self):
        model = dsgp4.mldsgp4(hidden_size=8)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.uniform_(-0.1, 0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mldsgp4.pth")
            model.save_model(path)
            self.assertTrue(os.path.isfile(path))

            restored = dsgp4.mldsgp4(hidden_size=8)
            restored.load_model(path)

        for name, parameter in model.state_dict().items():
            self.assertTrue(torch.equal(parameter, restored.state_dict()[name]))
        self.assertFalse(restored.training)


if __name__ == "__main__":
    unittest.main()
