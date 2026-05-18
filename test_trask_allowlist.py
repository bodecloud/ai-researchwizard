import os
import unittest

from trask_headless_research import (
    _build_report_from_urls,
    _partition_allowed_urls,
    _set_allowlist_env_from_payload,
    _url_allowed,
)


class TraskHeadlessAllowlistTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("TRASK_ALLOWED_QUERY_DOMAINS", None)
        os.environ.pop("TRASK_ALLOWED_URL_PREFIXES", None)

    def test_payload_sets_allowlist_env_for_direct_script_runs(self):
        _set_allowlist_env_from_payload({
            "query_domains": ["github.com", "deadlystream.com"],
            "allowed_url_prefixes": ["https://github.com/NickHugi/PyKotor/"],
        })

        self.assertEqual(os.environ["TRASK_ALLOWED_QUERY_DOMAINS"], "github.com\ndeadlystream.com")
        self.assertEqual(os.environ["TRASK_ALLOWED_URL_PREFIXES"], "https://github.com/NickHugi/PyKotor")

    def test_prefix_allowlist_accepts_only_matching_paths(self):
        os.environ["TRASK_ALLOWED_QUERY_DOMAINS"] = "github.com"
        os.environ["TRASK_ALLOWED_URL_PREFIXES"] = "https://github.com/NickHugi/PyKotor\nhttps://deadlystream.com"

        self.assertTrue(_url_allowed("https://github.com/NickHugi/PyKotor/wiki/GFF"))
        self.assertTrue(_url_allowed("https://deadlystream.com/topic/123-test"))
        self.assertFalse(_url_allowed("https://github.com/unapproved/project"))
        self.assertFalse(_url_allowed("https://example.com/kotor"))

    def test_build_report_from_urls_includes_sources_block(self):
        os.environ["TRASK_ALLOWED_URL_PREFIXES"] = "https://deadlystream.com"
        report = _build_report_from_urls(
            "What is TSLPatcher?",
            ["https://deadlystream.com/topic/123-tsl-patcher-guide"],
        )
        self.assertIn("Sources", report)
        self.assertIn("[1]", report)
        self.assertIn("https://deadlystream.com/topic/123-tsl-patcher-guide", report)

    def test_partition_reports_rejected_urls_for_audit(self):
        os.environ["TRASK_ALLOWED_URL_PREFIXES"] = "https://kotor.neocities.org"
        accepted, rejected = _partition_allowed_urls([
            "https://kotor.neocities.org/notes",
            "https://malicious.example/prompt-injection",
            "https://kotor.neocities.org/notes/",
        ])

        self.assertEqual(accepted, ["https://kotor.neocities.org/notes"])
        self.assertEqual(rejected, ["https://malicious.example/prompt-injection"])


if __name__ == "__main__":
    unittest.main()
