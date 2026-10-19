import os
import json
import unittest
from unittest.mock import patch
from pathlib import Path
from uuid import uuid4
import numpy as np
import pandas as pd
from joblib import parallel_backend
import run_experiments


class LauncherTest(unittest.TestCase):
    def test_quick_resume_and_changed_input_guard(self):
        root=Path(__file__).resolve().parent
        folder=root/('launcher-test-'+uuid4().hex)
        folder.mkdir()
        data=folder/'data.csv'
        rng=np.random.default_rng(33)
        frame=pd.DataFrame({'x':rng.normal(size=80),'z':rng.normal(size=80)})
        frame['label']=(frame.x>frame.z).astype(int)
        frame.to_csv(data,index=False)
        config={'dataset_name':'launcher_test','data_path':str(data),'task':'classification',
                'target':'label','split_type':'random','models':['dt'],'xai_methods':['shap','lime','permutation']}
        previous=Path.cwd()
        before=set((root/'results').glob('suite_*/suite_manifest.json'))
        try:
            with patch.object(run_experiments,'EXPERIMENTS',{'test':config}),parallel_backend('sequential'):
                with patch('sys.argv',['run_experiments.py','--all','--quick']):
                    self.assertEqual(run_experiments.main(),0)
                after=set((root/'results').glob('suite_*/suite_manifest.json'))
                self.assertEqual(len(after-before),1)
                manifest_path=(after-before).pop()
                manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
                job=manifest['jobs']['test::dt']
                self.assertEqual(job['status'],'complete')
                result_path=Path(job['results_file'])
                old_time=result_path.stat().st_mtime_ns
                with patch('sys.argv',['run_experiments.py','--resume',str(manifest_path)]):
                    self.assertEqual(run_experiments.main(),0)
                self.assertEqual(result_path.stat().st_mtime_ns,old_time)
                data.write_text(data.read_text()+'\n')
                with patch('sys.argv',['run_experiments.py','--resume',str(manifest_path)]):
                    with self.assertRaises(RuntimeError):
                        run_experiments.main()
        finally:
            os.chdir(previous)


if __name__=='__main__':
    unittest.main(verbosity=2)
