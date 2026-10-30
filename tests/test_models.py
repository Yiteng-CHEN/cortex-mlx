import unittest

from cortex.models.inspect import classify_layer, compress_pattern, summarize_layout
from cortex.models.train_probe import parse_lora_output, synthetic_examples


class TestInspect(unittest.TestCase):
    def test_qwen_style_layers(self):
        linear = classify_layer(["linear_attn.in_proj_qkv.weight", "linear_attn.conv1d.weight",
                                 "mlp.gate_proj.weight", "mlp.down_proj.weight"])
        full = classify_layer(["self_attn.q_proj.weight", "self_attn.o_proj.weight",
                               "mlp.gate_proj.weight"])
        self.assertEqual(linear, {"mixer": "linear_attention", "ffn": "mlp"})
        self.assertEqual(full, {"mixer": "attention", "ffn": "mlp"})

    def test_moe(self):
        moe = classify_layer(["self_attn.q_proj.weight", "mlp.gate.weight",
                              "mlp.switch_mlp.up_proj.weight"])
        self.assertEqual(moe["ffn"], "moe")

    def test_pattern(self):
        self.assertEqual(compress_pattern("LLLA" * 16), "(LLLA)x16")
        self.assertEqual(compress_pattern("A" * 40), "Ax40")
        self.assertEqual(compress_pattern("AALA"), "A2LA")
        layers = [{"mixer": m, "ffn": "mlp"} for m in ["linear_attention"] * 3 + ["attention"]]
        self.assertEqual(summarize_layout(layers * 2)["pattern"], "(LLLA)x2")


class TestProbe(unittest.TestCase):
    def test_parse_lora_output(self):
        log = ("Iter 1: Val loss 2.1\n"
               "Iter 5: Train loss 1.9, Learning Rate 1e-05, It/sec 0.40, Tokens/sec 700.1, "
               "Trained Tokens 8000, Peak mem 30.5 GB\n"
               "Iter 10: Train loss 1.7, Learning Rate 1e-05, It/sec 0.50, Tokens/sec 812.0, "
               "Trained Tokens 16000, Peak mem 31.2 GB\n")
        r = parse_lora_output(log)
        self.assertEqual(r["it_per_sec"], 0.5)
        self.assertEqual(r["seconds_per_step"], 2.0)
        self.assertEqual(r["peak_memory_gb"], 31.2)
        self.assertEqual(r["tokens_per_sec"], 812.0)

    def test_parse_empty(self):
        self.assertIsNone(parse_lora_output("boom")["seconds_per_step"])

    def test_synthetic_examples_size(self):
        ex = synthetic_examples(3, 1600)
        self.assertEqual(len(ex), 3)
        chars = len(ex[0]["messages"][0]["content"]) + len(ex[0]["messages"][1]["content"])
        self.assertGreater(chars // 4, 1000)


if __name__ == "__main__":
    unittest.main()
