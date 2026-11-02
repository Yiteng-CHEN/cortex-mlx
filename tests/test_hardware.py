import json
import threading
import unittest
import urllib.request

from neurolume.server.hardware import Hardware, parse_gpu, parse_vm_stat

IOREG = '''| |   "PerformanceStatistics" = {"Device Utilization %"=37,"In use system memory"=123,"Renderer Utilization %"=30}
| |   "PerformanceStatistics" = {"Device Utilization %"=5}'''
VM = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               30000.
Pages active:                            100000.
Pages inactive:                           90000.
Pages wired down:                         50000.
Pages occupied by compressor:             10000.
"""


def fake_run(cmd):
    return {"sysctl": "137438953472\n", "vm_stat": VM, "ioreg": IOREG}[cmd[0]]


class TestHardware(unittest.TestCase):
    def test_parsers(self):
        self.assertEqual(parse_gpu(IOREG), 0.37)
        self.assertIsNone(parse_gpu("nothing"))
        self.assertEqual(parse_vm_stat(VM), 160000 * 16384)

    def test_sample(self):
        h = Hardware(run=fake_run, active_model_bytes=lambda: 14_200_000_000)
        s = h.sample()
        self.assertEqual(s["gpu"], 0.37)
        self.assertEqual(s["mem_total_gb"], 137.4)
        self.assertAlmostEqual(s["mem_pct"], 0.019, places=3)
        self.assertEqual(s["model_gb"], 14.2)

    def test_degrades_to_none(self):
        s = Hardware(run=lambda c: "").sample()
        self.assertIsNone(s["gpu"])
        self.assertIsNone(s["mem_pct"])

    def test_endpoint(self):
        from neurolume.config import load_config
        from neurolume.server.app import serve
        httpd = serve(load_config(), port=0, block=False)
        httpd.neurolume_state.hardware = Hardware(run=fake_run)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        j = json.load(urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_address[1]}/api/hw"))
        self.assertEqual(j["gpu"], 0.37)
        httpd.shutdown()


if __name__ == "__main__":
    unittest.main()
