from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "alarm_filter.py"
SPEC = importlib.util.spec_from_file_location("alarm_filter", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
alarm_filter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(alarm_filter)


def record(detail: str, work_id: str | None, subfunction: str) -> dict[str, object]:
    params = {"subFunction": subfunction}
    if work_id is not None:
        params["workId"] = work_id
    return {
        "message": "platform request failed",
        "params": params,
        "detail": detail,
    }


def error_groups(result: dict[str, object]) -> list[dict[str, object]]:
    return [
        error
        for group in result["groups"]
        for child in group["children"]
        for error in child["errorGroups"]
    ]


class AlarmAggregationTests(unittest.TestCase):
    def test_masked_asset_names_merge_and_error_is_not_truncated(self) -> None:
        tail = "full stable tail " + "detail-" * 50
        def asset_error(masked_name: str, code: str, message: str) -> str:
            return (
                "failed to prepare reference images: ByteDance asset processing failed: "
                f'{{ "Id": "asset-12345678-dynamic", "Name": "{masked_name}", '
                '"AssetType": "Image", "GroupId": "group-1234567890123-random", '
                f'"Error": {{ "Code": "{code}", "Message": "{message}" }} }} '
                + tail
            )

        payload = {
            "data": [
                record(
                    asset_error("*" * 47 + "f", "Sensitive", "input rejected"),
                    "work-1",
                    "image",
                ),
                record(
                    asset_error("*" * 47 + "3", "Sensitive", "input rejected"),
                    "work-2",
                    "image",
                ),
                record(
                    asset_error("*" * 47 + "9", "Unavailable", "service unavailable"),
                    "work-3",
                    "image",
                ),
            ]
        }

        result = alarm_filter.aggregate(payload)
        groups = error_groups(result)
        sensitive_group = next(
            group for group in groups if '"Code": "Sensitive"' in group["error"]
        )

        self.assertEqual(len(groups), 2)
        self.assertEqual(sensitive_group["count"], 2)
        self.assertEqual(sensitive_group["workId"], "work-1")
        self.assertIn('<asset-name>', sensitive_group["error"])
        self.assertGreater(len(sensitive_group["error"]), 240)
        self.assertTrue(sensitive_group["error"].endswith(tail))
        self.assertFalse(sensitive_group["error"].endswith("…"))

    def test_signed_request_metadata_merges_but_error_causes_do_not(self) -> None:
        def signed_error(timestamp: str, sha256: str, code: str) -> str:
            return (
                "CreateAsset request failed status_code: 400 "
                "signed_headers: content-type;host;x-content-sha256;x-date "
                f"x_date: {timestamp} x_content_sha256: {sha256} response: "
                '{ "ResponseMetadata": { "RequestId": "dynamic", '
                f'"Error": {{ "Code": "{code}", '
                '"Message": "Frame rate must be between 23.8 FPS and 60 FPS." } } }'
            )

        payload = {
            "data": [
                record(
                    signed_error("20260716T010101Z", "a" * 64, "FpsTooLow"),
                    "low-1",
                    "video",
                ),
                record(
                    signed_error("20260716T020202Z", "b" * 64, "FpsTooLow"),
                    "low-2",
                    "video",
                ),
                record(
                    signed_error("20260716T030303Z", "c" * 64, "FpsTooHigh"),
                    "high-1",
                    "video",
                ),
            ]
        }

        groups = error_groups(alarm_filter.aggregate(payload))

        self.assertEqual(len(groups), 2)
        counts = {
            "FpsTooLow" if "FpsTooLow" in group["error"] else "FpsTooHigh": group["count"]
            for group in groups
        }
        self.assertEqual(counts, {"FpsTooLow": 2, "FpsTooHigh": 1})

    def test_exec_time_merges_and_later_work_id_fills_representative(self) -> None:
        payload = {
            "data": [
                record(
                    "API error. Response: {'code': 400, 'msg': 'same reason', "
                    "'exec_time': 0.1, 'ip': '1.2.3.4'}",
                    None,
                    "r2v",
                ),
                record(
                    "API error. Response: {'code': 400, 'msg': 'same reason', "
                    "'exec_time': 0.9, 'ip': '5.6.7.8'}",
                    "work-later",
                    "r2v",
                ),
            ]
        }

        groups = error_groups(alarm_filter.aggregate(payload))

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["count"], 2)
        self.assertEqual(groups[0]["workId"], "work-later")
        self.assertIn("<seconds>", groups[0]["error"])
        self.assertIn("<ip>", groups[0]["error"])


if __name__ == "__main__":
    unittest.main()
