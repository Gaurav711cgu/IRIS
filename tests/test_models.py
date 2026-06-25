import unittest
import torch
from src.models import GeneratorRRDB, PatchGANDiscriminator

class TestModels(unittest.TestCase):
    def test_generator_shape(self):
        """
        Verify that GeneratorRRDB accepts a 6-channel input
        and returns a 3-channel output scaled 2x in spatial dimensions.
        """
        batch_size = 2
        in_channels = 6
        h, w = 64, 64
        
        generator = GeneratorRRDB(in_channels=in_channels, out_channels=3)
        dummy_input = torch.randn(batch_size, in_channels, h, w)
        
        output = generator(dummy_input)
        
        self.assertEqual(output.shape, (batch_size, 3, 2 * h, 2 * w))
        self.assertTrue(torch.all(output >= 0.0))
        self.assertTrue(torch.all(output <= 1.0))

    def test_discriminator_shape(self):
        """
        Verify that PatchGANDiscriminator accepts a 3-channel input
        and outputs a grid of patch classifications.
        """
        batch_size = 2
        in_channels = 3
        h, w = 128, 128
        
        discriminator = PatchGANDiscriminator(in_channels=in_channels)
        dummy_input = torch.randn(batch_size, in_channels, h, w)
        
        output = discriminator(dummy_input)
        
        # In our architecture:
        # Layer 1: stride 2 -> 64x64
        # Layer 2: stride 2 -> 32x32
        # Layer 3: stride 2 -> 16x16
        # Layer 4: stride 1 -> 15x15 (padding 1, kernel 4)
        # Layer 5: stride 1 -> 14x14
        self.assertEqual(output.dim(), 4)
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], 1)
        self.assertTrue(output.shape[2] > 0)
        self.assertTrue(output.shape[3] > 0)
        print(f"PatchGAN output resolution for {h}x{w} input is: {output.shape[2]}x{output.shape[3]}")

if __name__ == '__main__':
    unittest.main()
