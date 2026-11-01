"""No-game regression checks for deployment admission and evidence retention."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import evaluate
import host_resources
import support

class ReleaseGuards(unittest.TestCase):
    def test_host_and_cgroup_keep_separate_ten_percent(self):
        gib=1024**3
        values={'/proc/meminfo':f'MemTotal: {256*gib//1024} kB\nMemAvailable: {20*gib//1024} kB\n',
                '/proc/self/cgroup':'0::/limited\n',
                '/sys/fs/cgroup/limited/memory.max':str(64*gib),
                '/sys/fs/cgroup/limited/memory.current':str(10*gib)}
        with patch.object(Path,'read_text',autospec=True,side_effect=lambda p,*a,**k:values[str(p)]), \
             patch.object(Path,'is_file',autospec=True,side_effect=lambda p:str(p) in values):
            result=evaluate.memory()
        self.assertLess(result['headroom'],0)
        self.assertEqual(len(result['layers']),2)
        self.assertAlmostEqual(result['headroom']/gib,-5.6)

    def test_shared_lock_admission_honors_actual_cpu_and_memory(self):
        records={os.getpid():{'start':'fixture'}}
        load={'reserved_cpu_slots':3,'future_growth_bytes':0}
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(host_resources,'scan',return_value=records), \
             patch.object(host_resources,'legacy_reservations',return_value=[]), \
             patch.object(host_resources,'account',return_value=load):
            lease=host_resources.HostLeases(Path(temp)/'private','offline-test')
            token,_=lease.acquire('job',10,100,100,cpu_capacity=4,memory_headroom=90)
            self.assertIsNone(token)
            load['reserved_cpu_slots']=0
            token,_=lease.acquire('job',10,100,100,cpu_capacity=64,memory_headroom=9)
            self.assertIsNone(token)
            self.assertEqual(list((Path(temp)/'private').glob('*.json')),[])

    def test_optimized_interpreter_rejected_and_child_override_removed(self):
        p=subprocess.run([sys.executable,'-O',str(ROOT/'evaluate.py'),'--help'],capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0)
        self.assertIn('not permitted',p.stderr)
        with patch.dict(os.environ,{'PYTHONOPTIMIZE':'2','DEEPSEEK_API_KEY':'offline-test-placeholder'}):
            env=support.native_environment(Path('/tmp/example'),Path('/tmp/example-root'))
        self.assertNotIn('PYTHONOPTIMIZE',env)
        self.assertNotIn('DEEPSEEK_API_KEY',env)

    def test_partial_failure_evidence_archives_without_final_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'raw';root.mkdir()
            (root/'combat_metrics_failed.json').write_text(json.dumps({'error':'synthetic interruption'}))
            (root/'root_console.log').write_text('synthetic interrupted output')
            result=evaluate.archive(root,Path(temp)/'failed.tar.gz')
            self.assertTrue(result['validated_by_reopening'])
            self.assertIn('combat_metrics_failed.json',result['files'])
            self.assertIn('root_console.log',result['files'])

if __name__=='__main__':unittest.main()
