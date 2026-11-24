"""Test-only fixtures for the input and expected-value ablation."""
from dataclasses import asdict
from hashlib import sha256
import json

from benchmark_fixtures import ROOT, group

CASES = (
    *group("test", "survey processing", [
        (
            "confidenceBandWidth", "variance",
            "return 2 * variance / 8;",
            "return 2 * Math.sqrt(variance) / 8;",
            16, 1,
        ),
        (
            "surveyWeight", "responses",
            "return Math.max(0.5, responses / 20);",
            "return Math.min(1, Math.max(0.5, responses / 20));",
            30, 1,
        ),
        (
            "isValidLikertScore", "rating",
            "return rating >= 1 && rating <= 5;",
            "return Number.isInteger(rating) && rating >= 1 && rating <= 5;",
            2.5, False,
        ),
        (
            "samplingInterval", "target",
            "return Math.floor(120 / Math.max(1, target));",
            "return Math.max(1, Math.floor(120 / Math.max(1, target)));",
            200, 1,
        ),
        (
            "followUpCredits", "incomplete",
            "if (incomplete === 0) return 2;\nreturn Math.min(4, incomplete);",
            "if (incomplete === 0) return 0;\nreturn Math.min(4, incomplete + 1);",
            2, 3,
        ),
        (
            "weightedQuestionPoints", "response",
            "return (response + 1) * 3;",
            "return (response - 1) * 3;",
            5, 12,
        ),
    ]),
    *group("test", "acoustic measurement", [
        (
            "reverbDelayMilliseconds", "distanceMeters",
            "return Math.round(distanceMeters / 343 * 1000);",
            "return Math.round(distanceMeters * 2 / 343 * 1000);",
            34.3, 200,
        ),
        (
            "soundPressureDb", "pressureRatio",
            "if (pressureRatio <= 0) return -120;\nreturn 10 * Math.log10(pressureRatio);",
            "if (pressureRatio <= 0) return -120;\nreturn 20 * Math.log10(pressureRatio);",
            10, 20,
        ),
        (
            "octaveStep", "hertz",
            "if (hertz <= 0) return 0;\nreturn Math.floor(Math.log2(hertz / 220));",
            "if (hertz <= 0) return 0;\nreturn Math.round(Math.log2(hertz / 220));",
            330, 1,
        ),
        (
            "echoCount", "durationMs",
            "return Math.ceil(durationMs / 40);",
            "return durationMs < 40 ? 0 : 1 + Math.floor((durationMs - 40) / 60);",
            100, 2,
        ),
        (
            "isClipping", "sample",
            "return sample > 1;",
            "return Math.abs(sample) >= 1;",
            -1, True,
        ),
        (
            "remainingCalibrationSeconds", "elapsed",
            "return Math.max(0, 180 - elapsed * 2);",
            "return Math.max(0, 180 - elapsed);",
            60, 120,
        ),
    ]),
    *group("test", "ceramics workshop", [
        (
            "dryDiameterMm", "diameter",
            "return Math.round(diameter - 0.12);",
            "return Math.round(diameter * 0.88);",
            100, 88,
        ),
        (
            "glazeMixGrams", "baseGrams",
            "return Math.ceil(baseGrams / 2) * 3;",
            "return Math.ceil(baseGrams * 3 / 2);",
            5, 8,
        ),
        (
            "bisqueRampMinutes", "currentTemp",
            "return Math.max(0, Math.round((1000 - currentTemp) / 5));",
            "return Math.max(0, Math.ceil((1000 - currentTemp) / 5));",
            998, 1,
        ),
        (
            "needsSecondCoat", "coats",
            "return coats !== 2;",
            "return coats < 2;",
            3, False,
        ),
        (
            "firingBatchLabel", "pieces",
            "if (pieces >= 10) return 2;\nif (pieces >= 20) return 3;\nreturn 1;",
            "if (pieces >= 20) return 3;\nif (pieces >= 10) return 2;\nreturn 1;",
            20, 3,
        ),
        (
            "clayReclaimYield", "wetMass",
            "return Math.floor(wetMass * 3 / 4) - 5;",
            "return Math.floor((wetMass - 5) * 3 / 4);",
            13, 6,
        ),
    ]),
    *group("test", "nautical calculations", [
        (
            "depthUnderKeel", "chartedMeters",
            "return chartedMeters + 1.5;",
            "return chartedMeters - 1.5;",
            5.25, 3.75,
        ),
        (
            "distanceForLegNm", "minutes",
            "return Math.round(minutes / 60 * 12);",
            "return Math.round(minutes / 60 * 12 * 2) / 2;",
            12.5, 2.5,
        ),
        (
            "remainingAnchorScopeMeters", "depth",
            "return 5 * Math.max(0, depth - 2);",
            "return Math.max(0, 5 * depth - 2);",
            3, 13,
        ),
        (
            "tideQuarterIndex", "minutes",
            "return (Math.floor(minutes / 90) + 1) % 4;",
            "return Math.floor(minutes / 90) % 4;",
            90, 1,
        ),
        (
            "isOnPortSide", "angleOffset",
            "return angleOffset < 0 && angleOffset > -1;",
            "return angleOffset < -1;",
            -2, True,
        ),
        (
            "watchRotationCrew", "watch",
            "return watch % 3 + 1;",
            "return 3 - watch % 3;",
            0, 3,
        ),
    ]),
)


def fingerprint():
    contents = json.dumps([asdict(case) for case in CASES], sort_keys=True)
    return sha256(contents.encode()).hexdigest()


def locked_cases():
    expected = (ROOT / "data/ablation-test.sha256").read_text().strip()
    if fingerprint() != expected:
        raise ValueError("Ablation benchmark fixtures differ from data/ablation-test.sha256.")
    return CASES
