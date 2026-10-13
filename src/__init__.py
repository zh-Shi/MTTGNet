"""MTTGNet model, historical retrieval, training and evaluation modules.

The current weather configurations use mttgnet_test.MTTGNetTestWrapper
(arch: clean). Shared components and earlier configurations remain in mttgnet.
The final weather model disables prototype memory and retains a three-source
fusion gate with a zero-valued third source.
"""

__version__ = "1.0.0"
