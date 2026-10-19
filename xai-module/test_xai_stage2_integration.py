"""Compatibility entry point: the current protocol is tested by the stage 3 suite."""
if __name__ == '__main__':
    import unittest
    from test_xai_stage3_integration import PipelineTest
    unittest.main(verbosity=2)
